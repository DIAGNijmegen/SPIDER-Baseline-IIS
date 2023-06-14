import numpy as np
import pandas as pd
from os import path
from collections import OrderedDict, defaultdict

from tiger.io import read_image
from tiger.resampling import align_mask_with_image, reorient_image
from tiger.metrics import dice_score, average_surface_distance
from sklearn.metrics import cohen_kappa_score as cohens_kappa

from experiments import Experiment
from datasets import Dataset, DatasetCollection, VolumeNotAvailableError
from config import datadir
from utils import flatten_dict


def safe_division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return np.nan


def evaluate_classification(manual, automatic):
    # Compares two binary masks and calculates metrics such as  TP, FP, etc
    tp = np.count_nonzero(manual & automatic)
    tn = np.count_nonzero(~manual & ~automatic)
    fp = np.count_nonzero(~manual & automatic)
    fn = np.count_nonzero(manual & ~automatic)

    if len(automatic) == 0:
        accuracy = 0
        f1 = 0
    else:
        accuracy = np.mean(manual == automatic)
        recall = safe_division(tp, tp + fn)
        precision = safe_division(tp, tp + fp)
        if np.isnan(recall) or np.isnan(precision) or recall + precision == 0:
            f1 = np.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return tp, tn, fp, fn, accuracy, f1


class Module(Experiment):
    def __init__(self, argv):
        super().__init__()
        self.add_param('dataset', type=str, list=True)
        self.add_param('subset', type=str, default='all',
                       help='Name of the subset, or mr_all_folds to evaluate all folds of MR dataset')
        self.add_param('scans', type=str, default='*')
        self.add_param('channel', type=str, default='')
        self.add_param('assume_all_completely_visible', type=bool)
        self.add_param('disable_cropping', type=bool)
        self.add_param('eval_orgres', type=bool)
        self.add_param('eval_incomplete', type=bool)
        self.add_param('ignore_missing', type=bool)
        self.add_param('skip_dice_score', type=bool)
        self.add_param('skip_surface_distance', type=bool)
        self.add_param('results', type=str, default='')
        self.add_param('binary', type=bool)
        self.add_param('SC_label', type=int, default=100)
        self.parse_and_restore(argv, execute_preserved_code=False)

    def execute(self):
        # Construct dataset collection
        data = DatasetCollection()
        for dataset_name in self['dataset']:
            data.add_dataset(Dataset(dataset_name, datadir))

        # Construct containers for the per-scan results
        all_dice_scores = defaultdict(list)
        all_surface_distances = defaultdict(list)
        all_completenesses = {'n_scans': 0, 'n_scans_with_mistakes': 0, 'n_missing': 0, 'manual': [], 'automatic': []}
        all_identifications = {'n_scans': 0, 'n_scans_with_mistakes': 0, 'n_missing': 0, 'n_correct': 0,
                               'n_incorrect': 0, 'x': [], 'y': []}

        n_scans_evaluated = 0
        n_scans_incomplete = 0
        n_scans_failure = 0

        # Iterate over all scans in the dataset
        if self['subset'] in ('all', 'mr_all_folds'):
            subset = None
        else:
            subset = self['subset']

        # Create empty dataframe to save all results
        df = pd.DataFrame(columns=['patient', 'label', 'dice', 'surface_distances', 'completeness_manual', 'completeness_automatic', 'label_automatic', 'correct_label', 'label_offset'])
        for dataset, scan in data.items(subset):
            # Create empty dataframe to collect all results of one scan
            df_scan = pd.DataFrame(columns=['patient', 'label', 'dice', 'surface_distances', 'completeness_manual', 'completeness_automatic', 'label_automatic', 'correct_label', 'label_offset'])
            # Print out the dataset and scan identifiers
            print('{}/{}'.format(dataset.identifier, scan.identifier))
            if scan.identifier == "46_t2":
                print("pause")

            # Check if this scan matches the scan selector
            if not scan.matches(self['scans']):
                print('  >> skipped')
                continue

            if self['disable_cropping']:
                scan.cropping = False

            # Try to find the automatic segmentation result
            if self['results'] != '':
                outdir = self.results_folder(dataset, ensure_exists=False, name=self['results'])
            else:
                outdir = self.results_folder(dataset, ensure_exists=False)
            if self['subset'] == 'mr_all_folds' and scan.fold > 1:
                outdir = outdir.replace('_Up1', '_Up{}'.format(scan.fold))

            mask_tpl = '{}_total_segmentation_orgres.mha' if self['eval_orgres'] else '{}_total_segmentation.mha'
            mask_identifier = '{}-{}'.format(scan.identifier, self['channel']) if self[
                                                                                      'channel'] != '' else scan.identifier
            mask_file = path.join(outdir, mask_tpl.format(mask_identifier))
            if not path.exists(mask_file):
                n_scans_incomplete += 1
                print(' > Automatic segmentation result missing!')
                continue

            mask_automatic, header_automatic = read_image(mask_file)

            # Try to load manual reference mask
            try:
                if self['eval_orgres']:
                    mask_manual = scan.original_mask
                    header = scan.original_header
                else:
                    mask_manual = scan.mask
                    header = scan.header
            except VolumeNotAvailableError:
                n_scans_incomplete += 1
                print(' > Manual segmentation mask missing!')
                continue

            # Check if manual and automatic mask have the same dimensions
            if mask_manual.shape != mask_automatic.shape:
                # print('align mask with image')
                # print(' > Manual and automatic masks shapes: {} vs {}'.format(mask_manual.shape, mask_automatic.shape))
                # print(header.direction)
                # print(header_automatic.direction)
                # # mask_automatic, _ = align_mask_with_image(mask_automatic, header_automatic, mask_manual.shape, header)
                # # header_automatic.direction = np.transpose(header_automatic.direction)
                # mask_automatic, header_automatic = reorient_image(mask_automatic, header_automatic)
                # print(' > Manual and automatic masks shapes: {} vs {}'.format(mask_manual.shape, mask_automatic.shape))
                # print(header.direction)
                # print(header_automatic.direction)
                # if mask_manual.shape != mask_automatic.shape:
                n_scans_incomplete += 1
                print(' > Manual and automatic masks have different shapes: {} vs {}'.format(mask_manual.shape,
                                                                                                 mask_automatic.shape))
                continue


            # Evaluate as binary masks?
            if self['binary']:
                mask_manual = mask_manual > 0
                mask_automatic = mask_automatic > 0

                print(' > Dice volume overlap score')
                score = dice_score(mask_manual, mask_automatic)
                all_dice_scores[1].append(score)
                print('    {}'.format(score))

                print(' > Absolute surface distance')
                distance = average_surface_distance(mask_manual, mask_automatic, voxel_spacing=scan.header['spacing'])
                all_surface_distances[1].append(distance)
                print('    {}'.format(distance))

                # Free up memory (scan remains in dataset, so has to be done by hand)
                n_scans_evaluated += 1
                scan.unload()
                print('-' * 50)
                continue

            # Build a lookup table for the labels in manual and automatic masks
            label_lut_vertebrae = OrderedDict()
            for vertebra in scan.vertebrae:
                # Determine label in automatic mask with which this label overlaps the most
                label_manual = vertebra.label
                overlap_automatic = mask_automatic[mask_manual == label_manual]
                overlap_automatic_foreground = overlap_automatic > 0
                if np.any(overlap_automatic_foreground):
                    label_automatic = np.bincount(overlap_automatic[overlap_automatic_foreground]).argmax()
                    label_lut_vertebrae[label_manual] = label_automatic

            # do the same thing for discs
            label_lut_discs = OrderedDict()
            for vertebra in scan.vertebrae:
                # Determine label in automatic mask with which this label overlaps the most
                label_manual = vertebra.disc_label
                overlap_automatic = mask_automatic[mask_manual == label_manual]
                overlap_automatic_foreground = overlap_automatic > 0
                if np.any(overlap_automatic_foreground):
                    label_automatic = np.bincount(overlap_automatic[overlap_automatic_foreground]).argmax()
                    label_lut_discs[label_manual] = label_automatic

            # If not a single match was found, the segmentation failed
            if len(label_lut_vertebrae) == 0:
                n_scans_failure += 1
                print(' > Segmentation failed')
                continue

            # check if there are discs in the ground truth without a vertebrae above it
            all_labels_list = sorted(list(np.unique(mask_manual[mask_manual > 0])))
            discs_label_list = sorted(list(np.unique(mask_manual[np.logical_and(mask_manual < (min(all_labels_list) + 200), mask_manual > 200)])))
            missing_discs = []
            for label in discs_label_list:
                if label not in label_lut_discs:
                    missing_discs.append(label)

            # add the missing discs to the look up table
            for label_manual in missing_discs:
                overlap_automatic = mask_automatic[mask_manual == label_manual]
                overlap_automatic_foreground = overlap_automatic > 0
                if np.any(overlap_automatic_foreground):
                    label_automatic = np.bincount(overlap_automatic[overlap_automatic_foreground]).argmax()
                    label_lut_discs[label_manual] = label_automatic

            # Calculate DICE volume overlap scores
            print(' > Dice volume overlap score')

            if self['skip_dice_score']:
                print('    - Skipping: Disabled by user! -')
            else:
                dice_scores_vert = dict()
                dice_scores_discs = dict()

                for vertebra in scan.vertebrae:
                    if not self['eval_incomplete'] and vertebra.completely_visible is False:
                        continue

                    label_manual = vertebra.label
                    if label_manual not in label_lut_vertebrae:
                        if self['ignore_missing']:
                            print('    - {}: missing (ignored)'.format(label_manual))
                            continue
                        score_vert = 0
                    else:
                        label_automatic = label_lut_vertebrae[label_manual]
                        score_vert = dice_score(mask_manual == label_manual, mask_automatic == label_automatic)
                        if vertebra.disc_label not in label_lut_discs:
                            score_disc = 0
                        else:
                            score_disc = dice_score(mask_manual == vertebra.disc_label, mask_automatic == label_lut_discs[vertebra.disc_label])

                    dice_scores_vert[label_manual] = score_vert
                    dice_scores_discs[vertebra.disc_label] = score_disc
                    all_dice_scores[label_manual].append(score_vert)
                    all_dice_scores[vertebra.disc_label].append(score_disc)

                    # put patient number and dice score in dataframe
                    if pd.isnull(df.index.max()):
                        df_scan.loc[0 if pd.isnull(df_scan.index.max()) else df_scan.index.max() + 1] \
                            = [scan.identifier, label_manual, score_vert, None, None, None, None, None, None]
                    else:
                        df_scan.loc[df.index.max() + 1 if pd.isnull(df_scan.index.max()) else df_scan.index.max() + 1] \
                            = [scan.identifier, label_manual, score_vert, None, None, None, None, None, None]

                    if pd.isnull(df.index.max()):
                        df_scan.loc[0 if pd.isnull(df_scan.index.max()) else df_scan.index.max() + 1] \
                            = [scan.identifier, vertebra.disc_label, score_disc, None, None, None, None, None, None]
                    else:
                        df_scan.loc[df.index.max() + 1 if pd.isnull(df_scan.index.max()) else df_scan.index.max() + 1] \
                            = [scan.identifier, vertebra.disc_label, score_disc, None, None, None, None, None, None]


                    # df_scan.loc[0 if pd.isnull(df_scan.index.max()) else df_scan.index.max() + 1] \
                    #     = [scan.identifier, label_manual, score_vert, None, None, None, None, None, None]
                    # df_scan.loc[df_scan.index.max() + 1] = [scan.identifier, vertebra.disc_label, score_disc, None,
                    #                                         None, None, None, None, None]

                    print('    - {}: {}'.format(label_manual, score_vert))
                    print('    - {}: {}'.format(vertebra.disc_label, score_disc))

                # calculate the dice for the missing discs
                for disc_label in missing_discs:
                    score_disc = dice_score(mask_manual == disc_label,
                                            mask_automatic == label_lut_discs[disc_label])
                    dice_scores_discs[disc_label] = score_disc
                    all_dice_scores[disc_label].append(score_disc)
                    df_scan.loc[df_scan.index.max() + 1] = [scan.identifier, disc_label, score_disc, None, None, None, None, None, None]
                    print('    - {}: {}'.format(disc_label, score_disc))

                score_SC = dice_score(mask_manual == 100, mask_automatic == self['SC_label'])
                df_scan.loc[df_scan.index.max() + 1] = [scan.identifier, 100, score_SC, None, None, None, None, None, None]

                dice_scores_discs[100] = score_SC
                all_dice_scores[100].append(score_SC)

                df = df.append(df_scan)
                print('    Mean: {}'.format(np.mean(list(dice_scores_vert.values()))))
                print('    Mean: {}'.format(np.mean(list(dice_scores_discs.values()))))

            # Calculate mean absolute surface distances
            print(' > Absolute surface distance')

            if self['skip_surface_distance']:
                print('    - Skipping: Disabled by user! -')
            else:
                surface_distances = OrderedDict()
                for vertebra in scan.vertebrae:
                    if not self['eval_incomplete'] and vertebra.completely_visible is False:
                        continue

                    label_manual = vertebra.label
                    if label_manual not in label_lut_vertebrae:
                        if self['ignore_missing']:
                            print('    - {}: missing (ignored)'.format(label_manual))
                            continue
                        distance_vert = np.nan
                        distance_disc = np.nan
                    else:
                        label_automatic = label_lut_vertebrae[label_manual]
                        distance_vert = average_surface_distance(mask_manual == label_manual,
                                                                 mask_automatic == label_automatic,
                                                                 voxel_spacing=header['spacing'])
                        if vertebra.disc_label not in label_lut_discs:
                            distance_disc = np.nan
                        else:
                            distance_disc = average_surface_distance(mask_manual == vertebra.disc_label,
                                                                     mask_automatic == label_lut_discs[vertebra.disc_label],
                                                                     voxel_spacing=header['spacing'])
                        # add surface distance vert to the dataframe
                        index = set(df[df['patient'] == scan.identifier].index.values) & \
                                set(df[df['label'] == label_manual].index.values)

                        if len(index) > 1 or len(index) == 0:
                            print('failed to fill dataframe')
                        else:
                            df.at[list(index)[0], 'surface_distances'] = distance_vert

                        # add surface distance disc to the dataframe
                        index = set(df[df['patient'] == scan.identifier].index.values) & \
                                set(df[df['label'] == vertebra.disc_label].index.values)
                        if len(index) > 1 or len(index) == 0:
                            print('failed to fill dataframe')
                        else:
                            df.at[list(index)[0], 'surface_distances'] = distance_disc

                    surface_distances[label_manual] = distance_vert
                    surface_distances[vertebra.disc_label] = distance_disc
                    all_surface_distances[label_manual].append(distance_vert)
                    all_surface_distances[vertebra.disc_label].append(distance_disc)
                    print('    - {}: {}'.format(label_manual, distance_vert))

                # calculate ASD for missing discs
                for disc_label in missing_discs:
                    distance_disc = average_surface_distance(mask_manual == disc_label,
                                                             mask_automatic == label_lut_discs[disc_label],
                                                             voxel_spacing=header['spacing'])

                    index = set(df[df['patient'] == scan.identifier].index.values) & \
                            set(df[df['label'] == disc_label].index.values)

                    if len(index) > 1 or len(index) == 0:
                        print('failed to fill dataframe')
                    else:
                        df.at[list(index)[0], 'surface_distances'] = distance_disc
                    surface_distances[vertebra.disc_label] = distance_disc
                    all_surface_distances[vertebra.disc_label].append(distance_disc)
                    print('    - {}: {}'.format(disc_label, distance_disc))

                # ADS for SC
                distance_SC = average_surface_distance(mask_manual == 100, mask_automatic == self['SC_label'],
                                                       voxel_spacing=header['spacing'])

                index = set(df[df['patient'] == scan.identifier].index.values) & \
                        set(df[df['label'] == 100].index.values)
                if len(index) > 1 or len(index) == 0:
                    print('failed to fill dataframe')
                else:
                    df.at[list(index)[0], 'surface_distances'] = distance_SC

                surface_distances[100] = distance_SC
                all_surface_distances[100].append(distance_SC)

                values = list(surface_distances.values())
                print('    Mean: {}'.format(np.nanmean(values) if np.count_nonzero(~np.isnan(values)) > 0 else np.nan))


            # Calculate completeness classification metrics (accuracy etc)
            print(' > Completeness classification')
            completeness_manual = []
            completeness_automatic = []

            for vertebra in scan.vertebrae:
                if self['assume_all_completely_visible']:
                    completely_visible = True
                else:
                    completely_visible = vertebra.completely_visible

                if vertebra.label in label_lut_vertebrae and completely_visible is not None:
                    completeness_manual.append(completely_visible)
                    completeness_automatic.append(label_lut_vertebrae[vertebra.label] < 100)

                    # add completeness to the dataframe
                    index = set(df[df['patient'] == scan.identifier].index.values) & \
                            set(df[df['label'] == vertebra.label].index.values)
                    if len(index) > 1 or len(index) == 0:
                        print('failed to fill dataframe')
                    else:
                        try:
                            df.at[list(index)[0], 'completeness_manual'] = completely_visible
                            df.at[list(index)[0], 'completeness_automatic'] = label_lut_vertebrae[vertebra.label] < 100
                        except IndexError:
                            print('hij stopt hier')

            completeness_manual = np.array(completeness_manual, dtype=bool)
            completeness_automatic = np.array(completeness_automatic, dtype=bool)

            if len(completeness_manual) == 0:
                print('    - Skipping: No manual annotations of vertebra completeness! -')
            else:
                n_missing = len(scan.vertebrae) - len(completeness_automatic)
                tp, tn, fp, fn, accuracy, f1 = evaluate_classification(completeness_manual, completeness_automatic)

                print('    - True positives: {}'.format(tp))
                print('    - True negatives: {}'.format(tn))
                print('    - False positives: {}'.format(fp))
                print('    - False negatives: {}'.format(fn))
                print('    - Missing: {}'.format(n_missing))
                print('    - Accuracy: {:.2f}%'.format(accuracy * 100))
                print('    - F1 score: {:.2f}%'.format(f1 * 100))

                all_completenesses['n_scans'] += 1
                all_completenesses['n_missing'] += n_missing
                all_completenesses['manual'].extend(completeness_manual)
                all_completenesses['automatic'].extend(completeness_automatic)
                if fp > 0 or fn > 0 or n_missing > 0:
                    all_completenesses['n_scans_with_mistakes'] += 1

            # Calculate vertebra labeling metrics (accuracy etc)
            print(' > Vertebra identification')
            n_correct = 0
            n_incorrect = 0
            n_missing = 0
            label_offsets = []

            for vertebra in scan.vertebrae:
                if not self['eval_incomplete'] and vertebra.completely_visible is False:
                    continue

                if vertebra.label not in label_lut_vertebrae:
                    n_missing += 1
                else:
                    label_automatic = label_lut_vertebrae[vertebra.label]
                    if label_automatic > 100:
                        label_automatic -= 100

                    all_identifications['x'].append(label_automatic)
                    all_identifications['y'].append(vertebra.label)

                    if label_automatic == vertebra.label:
                        n_correct += 1
                    else:
                        n_incorrect += 1
                        label_offsets.append(label_automatic - vertebra.label)

                    # add label evaluation to the dataframe
                    index = set(df[df['patient'] == scan.identifier].index.values) & \
                            set(df[df['label'] == vertebra.label].index.values)
                    if len(index) > 1:
                        print('failed to fill dataframe')
                    else:
                        df.at[list(index)[0], 'label_automatic'] = label_automatic
                        df.at[list(index)[0], 'correct_label'] = True if label_automatic == vertebra.label else False
                        df.at[list(index)[0], 'label_offset'] = label_automatic - vertebra.label

            print('    - Correctly labeled: {}'.format(n_correct))
            print('    - Incorrectly labeled: {}'.format(n_incorrect))
            print('    - Missing in automatic segmentation: {}'.format(n_missing))
            if len(label_offsets) > 0:
                print('    - Offsets: {}'.format(', '.join(str(lo) for lo in label_offsets)))

            all_identifications['n_scans'] += 1
            all_identifications['n_missing'] += n_missing
            all_identifications['n_correct'] += n_correct
            all_identifications['n_incorrect'] += n_incorrect
            if n_incorrect > 0 or n_missing > 0:
                all_identifications['n_scans_with_mistakes'] += 1

            # Free up memory (scan remains in dataset, so has to be done by hand)
            n_scans_evaluated += 1
            scan.unload()
            print('-' * 50)

        # Calculate overall performance (average over all scans, etc)
        print(
            'Overall ({}/{} scans)'.format(n_scans_evaluated, n_scans_evaluated + n_scans_incomplete + n_scans_failure))
        if n_scans_evaluated == 0:
            exit()

        print(' > Dice volume overlap score')
        if self['skip_dice_score']:
            print('    - Skipping: Disabled by user! -')
        else:
            if not self['binary']:
                for label_manual, scores in sorted(all_dice_scores.items()):
                    print('    - {}: {} +/- {}'.format(label_manual, np.mean(scores), np.std(scores)))
            all_values = flatten_dict(all_dice_scores)
            print('    Mean: {0} +/- {1}  [{0:.1%} +/- {1:.1%}]'.format(np.mean(all_values), np.std(all_values)))

        print(' > Absolute surface distance')
        if self['skip_surface_distance']:
            print('    - Skipping: Disabled by user! -')
        else:
            if not self['binary']:
                for label_manual, distances in sorted(all_surface_distances.items()):
                    print('    - {}: {} +/- {}'.format(label_manual, np.nanmean(distances), np.nanstd(distances)))
            all_values = flatten_dict(all_surface_distances)
            print('    Mean: {0} +/- {1} [{0:.1f} +/- {1:.1f}]'.format(np.nanmean(all_values), np.nanstd(all_values)))

        print(' > Completeness classification')
        n_scans = all_completenesses['n_scans']
        if self['binary']:
            print('    - Skipping: Binary mode! -')
        elif n_scans == 0:
            print('    - Skipping: No manual annotations of vertebra completeness! -')
        else:
            n_missing = all_completenesses['n_missing']
            completeness_manual = np.array(all_completenesses['manual'], dtype=bool)
            completeness_automatic = np.array(all_completenesses['automatic'], dtype=bool)
            tp, tn, fp, fn, accuracy, f1 = evaluate_classification(completeness_manual, completeness_automatic)

            print('    - True positives: {} ({:.2f} / scan)'.format(tp, tp / n_scans))
            print('    - True negatives: {} ({:.2f} / scan)'.format(tn, tn / n_scans))
            print('    - False positives: {} ({:.2f} / scan)'.format(fp, fp / n_scans))
            print('    - False negatives: {} ({:.2f} / scan)'.format(fn, fn / n_scans))
            print('    - Missing: {} ({:.2f} per scan)'.format(n_missing, n_missing / n_scans))
            print('    - Accuracy: {:.2f}%'.format(accuracy * 100))
            print('    - F1 score: {:.2f}%'.format(f1 * 100))
            print('    - Scans with mistakes: {}'.format(all_completenesses['n_scans_with_mistakes']))

        print(' > Vertebra identification')
        n_scans = all_identifications['n_scans']
        n_missing = all_identifications['n_missing']
        n_correct = all_identifications['n_correct']
        n_incorrect = all_identifications['n_incorrect']

        if self['binary']:
            print('    - Skipping: Binary mode! -')
        elif n_scans == 0:
            print('    - Skipping: Segmentation failed! -')
        else:
            accuracy = n_correct / (n_correct + n_incorrect)
            kappa = cohens_kappa(all_identifications['x'], all_identifications['y'], weights='linear')

            print('    - Correctly labeled: {} ({:.2f} / scan)'.format(n_correct, n_correct / n_scans))
            print('    - Incorrectly labeled: {} ({:.2f} / scan)'.format(n_incorrect, n_incorrect / n_scans))
            print('    - Missing: {} ({:.2f} / scan)'.format(n_missing, n_missing / n_scans))
            print('    - Accuracy: {:.2f}%'.format(accuracy * 100))
            print('    - Linearly weighted kappa: {:.4f}'.format(kappa))
            print('    - Scans with mistakes: {}'.format(all_identifications['n_scans_with_mistakes']))

        df.to_csv(path.join(outdir, 'evaluation.csv'))
