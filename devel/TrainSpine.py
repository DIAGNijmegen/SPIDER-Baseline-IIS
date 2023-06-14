import numpy as np
from scipy import ndimage

from tiger.augmentation import random_elastic_transform, random_gaussian_noise
# from rsgt.augmentation import random_smooth_grayvalue_transform

from config import datadir, patch_shape
from datasets import Dataset, DatasetCollection, CrossValidation
from networks import IterativeSegmentationNetworkDoubleMemoryState
from experiments import Experiment
from utils import compute_distance_weight_matrix, LearningCurve, Screenshot, Timer


class TrainingPhase:
    def __init__(self, name, channel=0):
        self.name = name
        self.channel = channel


class Module(Experiment):
    def __init__(self, argv):
        super().__init__()
        self.add_param('dataset', type=str, default='nlst_csi_xvertseg')
        self.add_param('fold', type=int, default=1)
        self.add_param('filters', type=int, default=84)
        self.add_param('traversal_direction', type=str, default='up')
        self.add_param('epochs', type=int, default=100000)
        self.add_param('nthpatchempty', type=int, default=3)
        self.add_param('complfrac', type=float, default=0.98)
        self.add_param('grayvalue_augmentation', type=bool)
        self.add_param('grayvalue_inversion', type=bool)  # not used anymore
        self.add_param('multitask', type=bool)
        self.add_param('snapshotevery', type=int, default=10)
        self.add_param('demoevery', type=int, default=500)
        self.add_param('unet', type=bool, help='Trains a multiclass U-net instead of the iterative network')
        self.add_param('only_vertebra', type=bool)
        self.add_param('vertebra_and_discs', type=bool)
        self.parse_and_preserve(argv, preserve_code=True)

    def execute(self):
        # Prepare the datasets
        only_ct_data = True
        if self['dataset'] == 'RadboudUMC_MRI_complete_crossfold':
            data = CrossValidation(Dataset('RadboudUMC_MRI_complete_crossfold', datadir), self['fold'])
            for scan in data.scans:
                scan.autocrop(crop_top=True, crop_bottom=False, normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'RadboudUMC_MRI_complete':
            data = CrossValidation(Dataset('RadboudUMC_MRI_complete', datadir), self['fold'])
            for scan in data.scans:
                scan.autocrop(crop_top=True, crop_bottom=False, normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'RadboudUMC_MRI_LWKScreening_VD':
            data = Dataset('RadboudUMC_MRI_LWKScreening_VD', datadir)
            for scan in data.scans:
                scan.autocrop(normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'RadboudUMC_MRI_LWKScreening_VD_1_48':
            data = Dataset('RadboudUMC_MRI_LWKScreening_VD_1_48', datadir)
            for scan in data.scans:
                scan.autocrop(normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'RadboudUMC_MRI_LWKScreening_COMPLETE':
            data = Dataset('RadboudUMC_MRI_LWKScreening_COMPLETE', datadir)
            for scan in data.scans:
                scan.autocrop(normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'DevelopmentData':
            data = Dataset('DevelopmentData', datadir)
            for scan in data.scans:
                scan.autocrop(normalize_direction=True)
            only_ct_data = False
        elif self['dataset'] == 'RadboudUMC_MRI_LWKScreening_combined':
            data = Dataset('RadboudUMC_MRI_LWKScreening_combined', datadir)
            for scan in data.scans:
                scan.autocrop()
            #
            # low_res = Dataset('RadboudUMC_MRI_LWKScreening_low_res', datadir)
            # for scan in low_res.scans:
            #     scan.autocrop()
            #
            # data = DatasetCollection(
            #     COMPLETE,
            #     low_res
            # )

            only_ct_data = False
        elif self['dataset'] == 'Radboud_JBZ_Rijnstate_SMK':
            data = Dataset('Radboud_JBZ_Rijnstate_SMK', datadir)
            for scan in data.scans:
                scan.autocrop()
        elif self['dataset'] == 'Final_dataset':
            data = Dataset('Final_dataset', datadir)
            for scan in data.scans:
                scan.autocrop()
        else:
            raise ValueError('Unknown dataset "{}"'.format(self['dataset']))

        # Define training phases
        phases = (TrainingPhase('training'), TrainingPhase('validation'))

        # Remove unused data from memory
        data.remove_subsets([phase.name for phase in phases], remove_others=True)
        self.store_dataset_details(data)

        print('Number of images per data subset:')
        for subset in data.subsets:
            print(' > "{}": {}'.format(subset, len(data.subsets[subset])))

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

        network = IterativeSegmentationNetworkDoubleMemoryState(self['filters'], n_input_channels, n_output_channels, self['traversal_direction'], mixed_precision=False)
        print('Network parameters: {} ({})'.format(len(network), network.__class__.__name__))

        # Restore network and optimizer to resume training?
        n_completed_epochs = 0
        if self['resume']:
            completed_epochs = self.epochs()
            if len(completed_epochs) > 0:
                n_completed_epochs = completed_epochs[-1]
                network.restore(self.network_parameter_file(n_completed_epochs))
                print('Resuming training from epoch {}'.format(n_completed_epochs))

        # Prepare training
        screenshots = dict()
        for phase in phases:
            screenshots[phase.name] = Screenshot(
                dynamic_window_level=phase.channel > 0 or self['grayvalue_augmentation'] or not only_ct_data, n=10
            )

        learning_curves = dict()
        for phase in phases:
            if phase.name.startswith('validation'):
                suffix = phase.name[len('validation'):]
                learning_curves[phase.name] = [
                    LearningCurve(self.learning_curve_file('loss' + suffix)),
                    LearningCurve(self.learning_curve_file('segmentation_error' + suffix)),
                    LearningCurve(self.learning_curve_file('classification_error' + suffix)),
                    LearningCurve(self.learning_curve_file('labeling_error' + suffix)),
                    LearningCurve(self.learning_curve_file('segmentation_error_vert' + suffix)),
                    LearningCurve(self.learning_curve_file('segmentation_error_disc' + suffix)),
                    LearningCurve(self.learning_curve_file('segmentation_error_SC' + suffix))
                ]

        # Train the network
        samples_per_epoch = network.n_devices

        for epoch in range(n_completed_epochs + 1, self['epochs'] + 1):
            print('{}/{}'.format(epoch, self['epochs']))
            clock = Timer()

            for phase in phases:
                image_patches, mask_patches, labels, completenesses, weights = [], [], [], [], []
                for sample_index in range(samples_per_epoch):
                    # Select a random scan from the appropriate subset
                    scan = data.random_scan(subset=phase.name)


                    # Extract a random patch from the scan
                    if self['nthpatchempty'] <= 0:
                        empty_patch = False
                    else:
                        empty_patch = (epoch * samples_per_epoch) % self['nthpatchempty'] == 0

                    if empty_patch:
                        # Every n-th patch is sampled from a completely random location in the image
                        patch_center = [np.random.randint(0, s) for s in scan.shape]
                        mask_patch = scan.mask_patches.extract_cuboid(patch_center, patch_shape)
                        label = 0
                    else:
                        # Search for a patch with a minimum amount of positive voxels
                        for attempt in range(100):
                            vertebra = scan.random_vertebra()
                            # if vertebra.label < 26:
                            bb = vertebra.bounding_box
                            patch_center = [
                                np.random.randint(lower, upper + 1)
                                for lower, upper in zip(bb.lower_corner, bb.upper_corner)
                            ]
                            mask_patch = scan.mask_patches.extract_cuboid(patch_center, patch_shape)

                            fragment_size = np.count_nonzero(mask_patch == vertebra.label)
                            if fragment_size > 1000:
                                break

                        label = vertebra.label

                    image_patch = scan.image_patches.extract_cuboid(patch_center, patch_shape)
                    # mask_patch[mask_patch >= 100] = 0  # make sure discs and spinal canal are removed

                    # Apply augmentations
                    if phase.name.startswith('training'):
                        # if self['grayvalue_augmentation']:
                            # image_patch = random_smooth_grayvalue_transform(image_patch,
                            #                                                 min_max_val=(-1000, 3096)) * 4096 - 1000

                        random_number = np.random.rand()
                        if random_number >= 0.75:
                            # Add Gaussian noise with random sigma
                            outside_area = image_patch < -1000
                            image_patch = random_gaussian_noise(image_patch, sigma=140, random_sigma=True)
                            image_patch[outside_area] = -2000
                        elif random_number >= 0.5:
                            # Apply Gaussian smoothing with random sigma
                            random_sigma = abs(np.random.standard_normal()) / 2
                            if random_sigma > 0.1:
                                image_patch = ndimage.gaussian_filter(image_patch, sigma=random_sigma)
                        else:
                            image_patch, mask_patch = random_elastic_transform(image_patch, mask_patch)

                        # Random cropping
                        random_number = np.random.rand()
                        if random_number < 0.05:
                            crop_top = random_number < 0.025  # 50% chance to crop at the top rather than the bottom
                            crop_n = int(round(random_number / 0.05 * patch_shape[2] / 2))

                            if crop_top:
                                image_patch[:, :, -crop_n:] = -2000
                                mask_patch[:, :, -crop_n:] = 0
                            else:
                                image_patch[:, :, :crop_n] = -2000
                                mask_patch[:, :, :crop_n] = 0

                    # Compute weight map
                    if empty_patch:
                        weight_matrix_vertebra = np.ones_like(mask_patch)
                        weight_matrix_disc = np.ones_like(mask_patch)
                    else:
                        weight_matrix_vertebra = compute_distance_weight_matrix(mask_patch == vertebra.label)
                        weight_matrix_disc = compute_distance_weight_matrix(mask_patch == vertebra.disc_label)

                    weight_matrix = np.stack([weight_matrix_vertebra, weight_matrix_disc, compute_distance_weight_matrix(mask_patch == 100)])

                    # Determine completeness of the vertebra
                    if empty_patch:
                        completely_visible = False
                    else:
                        completely_visible = vertebra.completely_visible
                        if completely_visible and fragment_size < self['complfrac'] * vertebra.size:
                            completely_visible = False

                    image_patches.append(image_patch)
                    mask_patches.append(mask_patch)
                    labels.append(label)
                    completenesses.append(completely_visible)
                    weights.append(weight_matrix)

                # Pass the patch through the network
                if phase.name.startswith('training'):
                    training_loss = network.train(image_patches, mask_patches, labels, completenesses, weights,
                                                  phase.channel)
                else:
                    validation_loss = network.predict_loss(image_patches, mask_patches, labels, completenesses, weights,
                                                           phase.channel)

                    # Store loss values
                    for i, curve in enumerate(learning_curves[phase.name]):
                        curve.post([training_loss[i], validation_loss[i]], epoch)

                # Send example segmentations once in a while to the visdom client
                if epoch < min(500, self['demoevery']) or epoch % self['demoevery'] == 0:
                    segmentation, classification, labeling = network.segment_and_classify(
                        image_patch, mask_patch, label, channel=phase.channel
                    )
                    memory_state, memory_state_disc, ground_truth = network.infer_memory_state_from_mask(mask_patch, label)
                    labeled_segmentation = np.copy(segmentation[0])
                    labeled_segmentation[segmentation[0] > 0] = int(labeling) #TODO epoch > 50 debug

                    try:
                        screenshots[phase.name].save(
                            self.screenshot_file(phase.name, epoch),
                            image_patch,
                            (segmentation[0], segmentation[1], segmentation[2], mask_patch, ground_truth[0], ground_truth[1], ground_truth[2], memory_state, memory_state_disc, labeled_segmentation)
                        )
                    except (IOError, FileNotFoundError, RuntimeError):
                        print(' > saving screenshot failed!')

            print(' > took {:.1f} s'.format(clock.elapsed()))

            # Store snapshot of the network and other things once in a while
            if epoch % self['snapshotevery'] == 0 or epoch == self['epochs']:
                attempts_left = 5
                while attempts_left > 0:
                    try:
                        network.snapshot(self.network_parameter_file(epoch))
                        for curves in learning_curves.values(): #TODO before the part where it crashes
                            for curve in curves:
                                curve.flush()
                    except (IOError, FileNotFoundError, RuntimeError):
                        attempts_left -= 1
                        print(' > saving network weights / learning curve failed!'
                              '(trying {} more times)'.format(attempts_left))
                    else:
                        break
