import numpy as np
from os import path
from time import time
from datetime import timedelta

from tiger.io import write_image
from tiger.patches import PatchExtractor3D
from tiger.masks import BoundingBox, retain_largest_components, ConnectedComponents
from tiger.resampling import resample_mask, pad_or_crop_image, align_mask_with_image, resample_image, reorient_image
from tiger.screenshots import ScreenshotGenerator, ScreenshotCollage, VertebraColors

from experiments import Experiment
from datasets import Dataset, DatasetCollection, VolumeNotAvailableError
from networks import IterativeSegmentationNetworkDoubleMemoryState
from config import datadir, patch_shape, working_resolution
from utils import remove_small_components, remove_low_intensity_surface_voxels, extract_valid_patch_pairs


def retain_posterior_components(image, mask, fragment_size=500):
    # first remove small fragments
    mask = remove_small_components(image, mask, min_size=fragment_size)
    # get connected components. Save the largest. Retain all components posterior of the largest component
    if np.count_nonzero(mask) > 0:
        mask_components = ConnectedComponents(mask)
        largest_component = mask_components.components[-1]
        largest_component_Y_mean = np.where(largest_component.mask)[1].mean()
        retain_components = []
        for component in mask_components.components:
            if np.where(component.mask)[1].mean() > largest_component_Y_mean:
                retain_components.append(component)

        # Find lowest component in retain_components
        retain_components_Z_mean = [np.where(component.mask)[2].mean() for component in retain_components]
        if retain_components_Z_mean:
            lowest_component = retain_components_Z_mean.index(min(retain_components_Z_mean))
            # Only put lowest component in one mask
            retain_mask = largest_component.mask
            component = retain_components[lowest_component]
            retain_mask[component.mask] = True
        else:
            retain_mask = largest_component.mask
            retain_mask[component.mask] = True

        # Put all components that need to be retained in one mask
        # retain_mask = largest_component.mask
        # for component in retain_components:
        #     retain_mask[component.mask] = True

        mask = np.multiply(mask, retain_mask)

    return mask


def put_patch_in_full_mask(segmentation, predicted_segmentation, patch_center, scan, label, resample=True):
    # Create a temporary segmentation mask the same size as the resampled image
    segmentation_temp = np.zeros_like(scan.image, dtype='float32')

    # Put segmentation mask from the patch into the full sized output mask
    segmentation_patch, prediction_patch = extract_valid_patch_pairs(segmentation_temp,
                                                                     predicted_segmentation,
                                                                     patch_center)
    segmentation_patch[segmentation_patch == 0] = prediction_patch[segmentation_patch == 0]

    # Resample mask to original image dimensions
    if resample:
        segmentation_temp = pad_or_crop_image(
            resample_image(segmentation_temp, scan.header['spacing'], scan.original_header['spacing'], order=3),
            target_shape=scan.original_shape
        )

    segmentation[segmentation_temp > 0.5] = label

    return segmentation


class Module(Experiment):
    def __init__(self, argv):
        super().__init__()
        self.add_param('epoch', type=int)
        self.add_param('dataset', type=str, list=True)
        self.add_param('subset', type=str, default='all')
        self.add_param('scans', type=str, default='*')
        self.add_param('channel', type=int, default=0)
        self.add_param('channels', type=int, default=1)
        self.add_param('max_vertebrae', type=int, default=25)
        self.add_param('max_steps', type=int, default=10, help='Maximum number of iteration steps per vertebra')
        self.add_param('patch_center_convergence_threshold', type=int, default=2, help='Maximum offset between iterations to qualify as converged')
        self.add_param('min_size_init', type=int, default=1000, help='Minimal number of voxels needed to start the traversal')
        self.add_param('min_size_cont', type=int, default=500, help='Minimal number of voxels needed to continue the traversal')  # use 100 for MR
        self.add_param('min_label_init', type=float, default=1.5)
        self.add_param('min_label_cont', type=float, default=0.0)
        self.add_param('min_label_accept', type=float, default=0.75)
        self.add_param('min_fragment_size', type=int, default=500, help='Minimal number of voxels that a single fragment must have')  # use 100 for MR
        self.add_param('keep_only_largest_component', type=bool, help='Keeps in every iteration only the largest component, ignores min_fragment_size')
        self.add_param('keep_all_components', type=bool)
        self.add_param('labeling_penalties', type=bool)
        self.add_param('labeling_ensemble', type=bool)
        self.add_param('surface_erosion_threshold', type=int, default=200)
        self.add_param('disable_cropping', type=bool)
        self.add_param('remove_incomplete', type=bool)
        self.add_param('export_image', type=bool)
        self.add_param('export_orgres', type=bool)
        self.add_param('export_original', type=bool)  # not only original resolution, but also original orientation
        self.add_param('export_mask', type=bool)
        self.add_param('export_trail', type=bool)
        self.add_param('export_activation_maps', type=bool)
        self.add_param('only_vertebra', type=bool)
        self.add_param('vertebra_and_discs', type=bool)
        self.add_param('retain_posterior_components', type=bool)
        self.parse_and_restore(argv, execute_preserved_code=False)

        # Prepare screenshot generator
        coordinates = (0.4, 0.45, 0.5, 0.55, 0.6)
        lut = VertebraColors()
        self.screenshot_generator = ScreenshotCollage(
            [ScreenshotGenerator(c, axis=0, window_level=None, overlay_lut=lut) for c in coordinates],  # window/level is overwritten later
            spacing=3,
            margin=3
        )

    def execute(self):
        # Restore trained network
        # Assemble the network
        if self['only_vertebra']:
            n_output_channels = 1
            n_input_channels = 2
        elif self['vertebra_and_discs']:
            n_output_channels = 2
            n_input_channels = 3
        else:
            n_output_channels = 3
            n_input_channels = 3

        network = IterativeSegmentationNetworkDoubleMemoryState(self['filters'], n_input_channels, n_output_channels, self['traversal_direction'])
        network.restore(self.network_parameter_file(self['epoch']))

        # Prepare dataset
        data = DatasetCollection()
        for dataset_name in self['dataset']:
            data.add_dataset(Dataset(dataset_name, datadir, originals=True))

        # Iterate over all scans in the dataset
        if self['subset'] == 'all':
            subset = None
        else:
            subset = self['subset']

        for dataset, scan in data.items(subset):
            self.process_scan(dataset, scan, network)

    def process_scan(self, dataset, scan, network):
        # Print out the dataset and scan identifiers
        scan_identifier = scan.identifier
        if self['channel'] > 0:
            scan_identifier += '-{}'.format(self['channel'])

        print('{}/{}'.format(dataset.identifier, scan_identifier))

        # Check if this scan matches the scan selector
        if not scan.matches(self['scans']):
            print('  >> skipped')
            return

        if self['disable_cropping']:
            scan.cropping = False

        # Prepare screenshot generator (adjust window leveling depending on scan type)
        try:
            grayvalue_augmentation = self['grayvalue_augmentation']
        except KeyError:
            grayvalue_augmentation = False

        if scan.metadata['modality'] == 'CT' and not grayvalue_augmentation:
            window_level = (400, 50)
        else:
            window_level = None

        for generators in self.screenshot_generator.generators:
            for generator in generators:
                generator.window_level = window_level

        # Let's go ...
        start_time_overall = time()

        # Create an empty mask for the segmentation result. One same size as original image, one new resolution used for the memory state
        segmentation = np.zeros(scan.original_shape, dtype='int16')
        segmentation_new_res = np.zeros_like(scan.image, dtype='int16')
        segmentation_new_res_patches = PatchExtractor3D(segmentation_new_res, pad_value=0, dtype='float32')

        # Create an empty mask for the disc segmentation result
        segmentation_discs = np.zeros(scan.original_shape, dtype='int16')
        segmentation_discs_new_res = np.zeros_like(scan.image, dtype='int16')
        segmentation_discs_new_res_patches = PatchExtractor3D(segmentation_discs_new_res, pad_value=0, dtype='float32')

        # Create an empty mask for the SC segmentation result
        segmentation_sc = np.zeros(scan.original_shape, dtype='int16')

        # Create an empty list for the predicted labels
        labels = dict()

        # Initialize patch location in the corner of the image
        scan_shape = np.asarray(scan.shape, dtype=int)
        max_coords = scan_shape - 1

        if self['traversal_direction'] == 'up':
            starting_coordinates = np.array((0, 0, 0), dtype=int)
        else:
            starting_coordinates = max_coords

        patch_center = np.copy(starting_coordinates)

        # Traverse over the image, searching for the spine
        n_steps = 0
        n_steps_vertebra = 0
        n_detected_vertebrae = 0

        completely_visible_vertebrae = set()

        start_time_iteration = time()
        while n_detected_vertebrae < self['max_vertebrae']:
            n_steps += 1

            # Extract image patch at current position and correspond part of the segmentation mask
            image_patch = scan.image_patches.extract_cuboid(patch_center, patch_shape)
            segmentation_new_res_patch = segmentation_new_res_patches.extract_cuboid(patch_center, patch_shape)
            segmentation_discs_new_res_patch = segmentation_discs_new_res_patches.extract_cuboid(patch_center, patch_shape)

            # Feed image patch through network
            memory_state = segmentation_new_res_patch > 0
            memory_state_discs = segmentation_discs_new_res_patch > 0
            prediction, classification, labeling = network.segment_and_classify(image_patch, memory_state=memory_state, memory_state_discs=memory_state_discs, threshold_segmentation=False, channel=self['channel'])

            predicted_segmentation_vertebrae = prediction[0] > 0.5
            predicted_segmentation_discs = prediction[1] > 0.5

            predicted_segmentation_vertebrae_float = np.copy(prediction[0])
            predicted_segmentation_discs_float = np.copy(prediction[1])
            predicted_segmentation_sc_float = np.copy(prediction[2])

            predicted_completeness = classification[0][0] > 0.5
            predicted_label = labeling

            n_positive_voxels = np.count_nonzero(predicted_segmentation_vertebrae)
            # print(predicted_completeness[0].any())
            # print(predicted_completeness[0][0])

            # If the network thinks that this is a complete vertebra, we should keep only a single component
            # For incomplete vertebrae, the vertebra can be split into multiple components
            # if self['keep_all_components'] and not predicted_completeness:
            #     predicted_segmentation_vertebrae = remove_small_components(image_patch, predicted_segmentation_vertebrae, self['min_fragment_size'])
            # else:
            if self['retain_posterior_components']:
                predicted_segmentation_vertebrae = retain_posterior_components(image_patch, predicted_segmentation_vertebrae, self['min_fragment_size'])
            elif self['keep_only_largest_component']:
                predicted_segmentation_vertebrae = retain_largest_components(predicted_segmentation_vertebrae, labels=(True,), background=False, n=1, connectivity=int(3))
            else:
                predicted_segmentation_vertebrae = remove_small_components(image_patch, predicted_segmentation_vertebrae, self['min_fragment_size'])
            predicted_segmentation_vertebrae_float = np.multiply(predicted_segmentation_vertebrae_float, predicted_segmentation_vertebrae)

            # calculate the volume of detected vertebra
            detected_volume = np.count_nonzero(predicted_segmentation_vertebrae & ~memory_state)
            voxel_volume = working_resolution[0]*working_resolution[1]*working_resolution[2]
            detected_volume *= voxel_volume

            # Print out some stats
            print('Step {:<3}: {} -> {} voxels ({:.3f}), label: {}'.format(n_steps, patch_center, detected_volume, classification[0][0], predicted_label))
            if self['debug']:
                print('-> number of positive voxels: {}'.format(n_positive_voxels))

            # Export step-by-step results?
            if self['export_trail']:
                # Create a mask with the original image shape and copy-paste patch-based result into it
                trail_mask = np.zeros(scan_shape, dtype='float32')
                trail_mask_patch, probabilities = extract_valid_patch_pairs(trail_mask, prediction[0], patch_center)
                trail_mask_patch.fill(1)
                trail_mask_patch[:] = 1 + probabilities[:]

                # Write snapshot to disk
                traildir = self.trails_folder(dataset, scan, n_steps == 1)
                if n_steps == 1:
                    write_image(path.join(traildir, 'image.mha'), scan.image, scan.header)
                write_image(path.join(traildir, 'step{:05}_patch.mha'.format(n_steps)), trail_mask, scan.header)
                write_image(path.join(traildir, 'step{:05}_segmentation.mha'.format(n_steps)), segmentation,
                            scan.original_header)
                write_image(path.join(traildir, 'step{:05}_segmentation_discs.mha'.format(n_steps)), segmentation_discs,
                            scan.original_header)

                if self['export_activation_maps']:
                    write_image(path.join(traildir, 'step{:05}_image.mha'.format(n_steps)), image_patch.astype('int16'), scan.header)
                    write_image(path.join(traildir, 'step{:05}_memory.mha'.format(n_steps)), memory_state.astype('int16'), scan.header)

                    # Obtain first level activation map
                    for level in range(1, 9):
                        am = network.nth_activation_map(image_patch, memory_state, level)
                        write_image(path.join(traildir, 'step{:05}_am{}.mha'.format(n_steps, level)), am.astype('float32'), scan.header)

            # Check if a large enough fragment of vertebral bone was detected
            ts = 'cont' if n_detected_vertebrae > 0 else 'init'
            if detected_volume >= self[f'min_size_{ts}'] and predicted_label >= self[f'min_label_{ts}']:
                # Determine center of bounding box of the fragment -> center of next patch
                bb = BoundingBox(predicted_segmentation_vertebrae)
                fragment_center = bb.center
                new_patch_center = np.clip(patch_center + np.round(fragment_center).astype(int) - np.asarray(patch_shape) // 2, 0, max_coords)

                # Check if the patch moved or if the position has converged
                if n_steps_vertebra > 0 and (
                        n_steps_vertebra > self['max_steps'] or
                        max(abs(new_patch_center - patch_center)) <= self['patch_center_convergence_threshold']
                ):
                    if self['labeling_ensemble']:
                        predicted_labels, predicted_completenesses = [], []
                        # for ensemble_epoch in range(self['epoch'] - 5 * 5 * self['snapshotevery'], self['epoch'], 5 * self['snapshotevery']):
                        for ensemble_epoch in range(175000 - 5 * 10 * self['snapshotevery'], 175000, 10 * self['snapshotevery']):
                            network.restore(self.network_parameter_file(ensemble_epoch))
                            ensemble_prediction = network.segment_and_classify(image_patch, memory_state=memory_state, memory_state_discs=memory_state_discs, threshold_segmentation=False, channel=self['channel'])
                            predicted_completenesses.append(ensemble_prediction[1])
                            predicted_labels.append(ensemble_prediction[2])

                        predicted_completenesses.append(classification[0][0])
                        predicted_labels.append(labeling)

                        predicted_completeness = np.mean(predicted_completenesses) > 0.5
                        predicted_label = np.mean(predicted_labels)

                        network.restore(self.network_parameter_file(self['epoch']))
                        print(' ensembled label / completeness: {} / {}'.format(predicted_label, predicted_completeness))

                    if predicted_label < self['min_label_accept']:
                        print('False positive')
                        break

                    # Patch position converged, so we need to add the segmentation to the output mask
                    label = n_detected_vertebrae + 1
                    print(' >> vertebra {} detected (completely visible: {})'.format(label, 'yes' if predicted_completeness else 'no'))

                    segmentation_new_res_patch, prediction_patch = extract_valid_patch_pairs(segmentation_new_res, predicted_segmentation_vertebrae, patch_center)
                    segmentation_new_res_patch[segmentation_new_res_patch == 0] = prediction_patch[segmentation_new_res_patch == 0] * label

                    segmentation = put_patch_in_full_mask(segmentation, predicted_segmentation_vertebrae_float,
                                                                  patch_center, scan, label)

                    predicted_segmentation_discs = retain_largest_components(predicted_segmentation_discs,
                                                                             labels=(True,), background=False, n=1)
                    segmentation_discs_new_res = put_patch_in_full_mask(segmentation_discs_new_res, predicted_segmentation_discs,
                                                                        patch_center, scan, label, resample=False)

                    predicted_segmentation_discs_float = retain_largest_components(predicted_segmentation_discs_float,
                                                                                   labels=(True,), background=False, n=1)
                    segmentation_discs = put_patch_in_full_mask(segmentation_discs, predicted_segmentation_discs_float,
                                                                patch_center, scan, label)

                    segmentation_sc = put_patch_in_full_mask(segmentation_sc, predicted_segmentation_sc_float,
                                                             patch_center, scan, 1)

                    # Store classification of the detected vertebra
                    if predicted_completeness:
                        completely_visible_vertebrae.add(label)

                    # Store predicted label
                    labels[label] = np.clip(predicted_label, 1, self['max_vertebrae'])

                    # Increase/reset counters
                    n_detected_vertebrae += 1
                    n_steps_vertebra = 0
                else:
                    # Did not converge, keep moving toward the center of the vertebra
                    n_steps_vertebra += 1

                    if n_steps_vertebra > self['max_steps']:
                        # Move to center between two previous positions when the number of maximum steps is reached
                        patch_center = np.round((patch_center + new_patch_center) / 2).astype(int)
                    else:
                        patch_center = new_patch_center
            elif n_detected_vertebrae > 0:
                # No fragment of vertebral bone found, but we had found at least one vertebra before
                print('Lost trail')
                break
            else:
                # No fragment of vertebral bone found, keep moving in a sliding window fashion
                entire_image_searched = False

                if self['traversal_direction'] == 'up':
                    for i in range(3):
                        patch_center[i] += patch_shape[i] // 2
                        if patch_center[i] <= scan_shape[i] - 1:
                            break
                        elif i == 2:
                            entire_image_searched = True
                        else:
                            patch_center[i] = starting_coordinates[i]
                else:
                    for i in range(3):
                        patch_center[i] -= patch_shape[i] // 2
                        if patch_center[i] >= 0:
                            break
                        elif i == 2:
                            entire_image_searched = True
                        else:
                            patch_center[i] = starting_coordinates[i]

                if entire_image_searched:
                    print('Reached end of scan')
                    break

        runtime_iteration = time() - start_time_iteration

        first_complete_internal_label = None
        for internal_label in range(1, n_detected_vertebrae + 1):
            if internal_label in completely_visible_vertebrae:
                first_complete_internal_label = internal_label
                break

        # Determine most plausible labeling
        if n_detected_vertebrae > 0:
            for internal_label in labels:
                # Replace raw prediction with a probability vector
                probabilities = [0] * self['max_vertebrae']

                # Incomplete vertebrae do not get to vote since their predictions might be incorrect
                if internal_label in completely_visible_vertebrae or len(completely_visible_vertebrae) == 0:
                    raw_prediction = labels[internal_label]
                    predicted_label = int(np.round(raw_prediction))

                    # if predicted_label in (8, 19):
                    #     importance = 10
                    # elif predicted_label in (1, 24, 25):
                    #     importance = 5
                    # else:
                    importance = 1

                    if predicted_label >= raw_prediction:
                        probabilities[predicted_label - 1] = (1 - (predicted_label - raw_prediction)) * importance
                        probabilities[predicted_label - 2] = predicted_label - raw_prediction
                    else:
                        probabilities[predicted_label - 1] = (1 - (raw_prediction - predicted_label)) * importance
                        probabilities[predicted_label] = raw_prediction - predicted_label

                labels[internal_label] = probabilities

            configurations = dict()

            internal_labels = list(sorted(np.unique(segmentation[segmentation > 0])))

            if self['traversal_direction'] == 'up':
                internal_labels = internal_labels[::-1]

            for first_vertebra_label in range(1, self['max_vertebrae'] - n_detected_vertebrae + 2):
                sum_probs = 0
                for label, internal_label in zip(range(first_vertebra_label, first_vertebra_label + n_detected_vertebrae), internal_labels):
                    sum_probs += labels[internal_label][label - 1]
                configurations[first_vertebra_label] = sum_probs / n_detected_vertebrae

                # More than 5 lumbar vertebrae are not very commonly seen, so introduce a penalty for such configurations
                if self['labeling_penalties'] and first_vertebra_label + n_detected_vertebrae - 1 > 24:
                    configurations[first_vertebra_label] /= 2

            best_configuration = max(configurations.items(), key=lambda item: item[1])
            first_label = best_configuration[0]
            print('Best label configuration: {}-{} ({})'.format(first_label, first_label + n_detected_vertebrae - 1, best_configuration[1]))
            for i, configuration in enumerate(sorted(configurations.items(), key=lambda item: item[1], reverse=True)):
                if i > 0 and configuration[1] > 0.0001:
                    print(' > {}-{}: {}'.format(configuration[0], configuration[0] + n_detected_vertebrae - 1, configuration[1]))

            if self['labeling_penalties']:
                if first_label == 2:
                    first_label = 1
                    print(f'Rule-based correction (#1) to {first_label}-{first_label + n_detected_vertebrae - 1}')
                if first_label + n_detected_vertebrae - 1 == 23:
                    first_label += 1
                    print(f'Rule-based correction (#2) to {first_label}-{first_label + n_detected_vertebrae - 1}')

            # replace labels vertebra
            labels = list(range(first_label, first_label + n_detected_vertebrae))
            lut = {old: new for old, new in zip(internal_labels, labels)}
            lut[0] = 0
            replace_labels = np.vectorize(lambda v: lut[v])
            segmentation = replace_labels(segmentation).astype('int16')
            segmentation_discs = replace_labels(segmentation_discs).astype('int16')

            # Remove or relabel incompletely visible vertebrae
            completely_visible_vertebrae = set(lut[v] for v in completely_visible_vertebrae)
            for label in np.unique(segmentation[segmentation > 0]):
                if label not in completely_visible_vertebrae:
                    new_label = 0 if self['remove_incomplete'] else 100 + label
                    segmentation[segmentation == label] = new_label

        # Merge masks together
        total_segmentation = np.copy(segmentation)
        for label in labels:
            total_segmentation[np.logical_and(segmentation_discs == label, total_segmentation == 0)] = label + 200

        segmentation_sc = retain_largest_components(segmentation_sc, labels=(True,), background=False, n=1)
        total_segmentation[np.logical_and(segmentation_sc > 0, total_segmentation == 0)] = 100

        # Done, now export results...
        outdir = self.results_folder(dataset)

        # Generate screenshot
        screenshot = self.screenshot_generator(scan.original_image, total_segmentation)
        screenshot_file = path.join(outdir, '{}_screenshot.jpg'.format(scan_identifier))
        screenshot.save(screenshot_file)

        # Prepare original mask for export if requested
        mask = None
        if self['export_mask']:
            try:
                mask = scan.original_mask if self['export_orgres'] else scan.mask
                for vertebra in scan.vertebrae:
                    if vertebra.completely_visible is False:
                        mask[mask == vertebra.label] += 100
            except VolumeNotAvailableError:
                pass

        if scan.metadata['modality'] == 'MR':
            surface_erosion_threshold = -2000
            print("Surface erosion = -2000")
        else:
            surface_erosion_threshold = self['surface_erosion_threshold']
            print("Surface erosion = self['surface_erosion_threshold']")

        # Resample to original resolution (if requested) and export results
        if self['export_orgres']:
            remove_low_intensity_surface_voxels(scan.original_image, segmentation, surface_erosion_threshold)
            segmentation_path = path.join(outdir, '{}_segmentation_orgres.mha'.format(scan_identifier))
            write_image(segmentation_path, segmentation, scan.original_header)

            remove_low_intensity_surface_voxels(scan.original_image, segmentation_discs, surface_erosion_threshold)
            segmentation_discs_path = path.join(outdir, '{}_segmentation_discs_orgres.mha'.format(scan_identifier))
            write_image(segmentation_discs_path, segmentation_discs, scan.original_header)

            remove_low_intensity_surface_voxels(scan.original_image, total_segmentation, surface_erosion_threshold)
            total_segmentation_path = path.join(outdir, '{}_total_segmentation_orgres.mha'.format(scan_identifier))
            write_image(total_segmentation_path, total_segmentation, scan.original_header)

            remove_low_intensity_surface_voxels(scan.original_image, segmentation_sc, surface_erosion_threshold)
            segmentation_sc_path = path.join(outdir, '{}_segmentation_sc_orgres.mha'.format(scan_identifier))
            write_image(segmentation_sc_path, segmentation_sc, scan.original_header)

            if self['export_image']:
                write_image(path.join(outdir, '{}_image.mha'.format(scan_identifier)), scan.image, scan.header)
                write_image(path.join(outdir, '{}_image_orgres.mha'.format(scan_identifier)), scan.original_image, scan.original_header)
            if self['export_mask'] and mask is not None:
                write_image(path.join(outdir, '{}_mask_orgres.mha'.format(scan_identifier)), mask, scan.original_header)

            self.dump_arguments(
                path.join(outdir, '{}_arguments_orgres.json'.format(scan_identifier)),
                include_internal_args=True
            )
        elif self['export_original']:
            # total_segmentation = pad_or_crop_image(
            #     resample_mask(total_segmentation, scan.header['spacing'], scan.original_header['spacing']),
            #     target_shape=scan.original_shape
            # )
            total_segmentation, _ = align_mask_with_image(total_segmentation, scan.original_header, scan.initial_shape, scan.initial_header)
            # total_segmentation, _ = reorient_image(total_segmentation, scan.header, new_direction=scan.initial_header['direction'])
            remove_low_intensity_surface_voxels(scan.original_image, total_segmentation, surface_erosion_threshold)
            print(scan.initial_header['direction'])
            print(scan.header['direction'])

            segmentation_path = path.join(outdir, '{}_total_segmentation_original.mha'.format(scan_identifier))
            write_image(segmentation_path, total_segmentation, scan.initial_header)
            self.dump_arguments(
                path.join(outdir, '{}_arguments_original.json'.format(scan_identifier)),
                include_internal_args=True
            )
        else:
            remove_low_intensity_surface_voxels(scan.image, segmentation, surface_erosion_threshold)
            write_image(path.join(outdir, '{}_segmentation.mha'.format(scan_identifier)), segmentation, scan.header)

            if self['export_image']:
                write_image(path.join(outdir, '{}_image.mha'.format(scan_identifier)), scan.image, scan.header)
            if self['export_mask'] and mask is not None:
                write_image(path.join(outdir, '{}_mask.mha'.format(scan_identifier)), mask, scan.header)

            self.dump_arguments(
                path.join(outdir, '{}_arguments.json'.format(scan_identifier)),
                include_internal_args=True
            )

        # Report runtime
        runtime_overall = timedelta(seconds=round(time() - start_time_overall))
        seconds_per_step = runtime_iteration / n_steps
        runtime_iteration = timedelta(seconds=round(runtime_iteration))
        print('Runtime: {} of which {} were spent iterating ({:.3f} s/it)'.format(runtime_overall, runtime_iteration, seconds_per_step))
        print('{} vertebrae detected, {} completely visible'.format(n_detected_vertebrae, len(completely_visible_vertebrae)))

        scan.unload()
