import numpy as np

from os import path
from glob import glob
from argparse import ArgumentParser

from tiger.io import read_image, read_json

from config import datadir


def strseq(values):
    return ', '.join([str(v) for v in sorted(values)])


class Module:
    def __init__(self, argv):
        parser = ArgumentParser()
        parser.add_argument('dataset')
        parser.add_argument('--expect-discs', action='store_true')
        parser.add_argument('--expect-spinal-canal', action='store_true')
        self.args = parser.parse_args(argv)

    def execute(self):
        # Does the dataset exist?
        if not path.exists(path.join(datadir, self.args.dataset)):
            print('Dataset "{}" does not exist in {}'.format(self.args.dataset, datadir))
            return

        # Does this dataset have a metadata file?
        try:
            metadata = read_json(path.join(datadir, self.args.dataset, 'metadata.json'))
            metadata = {m['identifier']: m for m in metadata}
            print('Metadata file contains {} entries'.format(len(metadata)))
        except IOError:
            metadata = None
            print('No metadata file')

        # Loop over all images in the dataset
        images = glob(path.join(datadir, self.args.dataset, 'images', '*.mha'))
        for imagefile in sorted(images):
            uid = path.basename(imagefile)[:-4]
            print(uid)

            # Check if mask file exists
            maskfile = path.join(datadir, self.args.dataset, 'masks', path.basename(imagefile))
            has_mask = path.exists(maskfile)

            if not has_mask:
                if metadata:
                    if uid in metadata:
                        print('> no mask, but listed in metadata')
                    else:
                        print('> no mask, not listed in metadata')
                else:
                    print('> no mask')
                continue

            # Read image and mask
            image, header = read_image(imagefile)
            mask, mask_header = read_image(maskfile)

            # Check that mask and image have the same size, etc
            if image.shape != mask.shape:
                print('> image and mask have different shapes')
            if header != mask_header:
                print('> image and mask have different headers')

            # Check that labels make sense
            labels = [v for v in np.unique(mask) if v != 0]
            label_issues = False
            if min(labels) < 1 or max(labels) > 225:
                print('> weird labels: {}'.format(strseq(labels)))

            # Check whether there are incomplete vertebrae (should not be the case!)
            incomplete_labels = set(v for v in labels if 100 < v < 200)
            if len(incomplete_labels) > 0:
                label_issues = True
                print('> incomplete labels (should never be in training data!): {}'.format(strseq(incomplete_labels)))

            # Check that there are no gaps in the labels
            vertebra_labels = list(sorted(v for v in labels if v < 100))
            expected_labels = np.arange(np.min(vertebra_labels), np.max(vertebra_labels) + 1)
            if not np.array_equal(vertebra_labels, expected_labels):
                label_issues = True
                print('> inconsistent vertebra labels: {}'.format(', '.format(vertebra_labels)))

            disc_labels = list(sorted(v for v in labels if 200 < v < 300))
            if len(disc_labels) > 0:
                expected_labels = np.arange(np.min(disc_labels), np.max(disc_labels) + 1)
                if not np.array_equal(disc_labels, expected_labels):
                    label_issues = True
                    print('> inconsistent disc labels: {}'.format(', '.format(disc_labels)))

            if self.args.expect_discs:
                missing_discs = [v + 200 for v in vertebra_labels if v + 200 not in disc_labels]
                if len(missing_discs) > 0:
                    label_issues = True
                    print('> discs expected, but missing: {}'.format(strseq(missing_discs)))

                # Check whether there is a disc above the top-most vertebra
                highest_disc = False
                extra_discs = [d - 200 for d in disc_labels if d - 200 not in vertebra_labels]
                if len(extra_discs) == 1 and max(vertebra_labels) + 1 in extra_discs:
                    print('> disc above top most vertebra')
                    highest_disc = True
                elif len(extra_discs) > 0:
                    label_issues = True
                    print('> discs without matching vertebra: {}'.format(strseq(extra_discs)))
            elif len(disc_labels) > 0:
                label_issues = True
                print('> no expecting any discs, but found some: {}'.format(strseq(disc_labels)))

            # Check that spinal canal is there
            spinal_canal = 100 in labels
            if self.args.expect_spinal_canal:
                if not spinal_canal:
                    label_issues = True
                    print('> spinal canal expected, but missing')
            else:
                if spinal_canal:
                    label_issues = True
                    print('> not expecting spinal canal, but label is present')

            # Check that first/last visible vertebrae are actually in the mask
            if metadata and uid in metadata:
                m = metadata[uid]
                for p in ('first', 'last'):
                    try:
                        k = '{}_completely_visible_vertebra'.format(p)
                        if m[k] not in labels:
                            label_issues = True
                            print(f'> {p} completely visible vertebra label not in mask')
                    except KeyError:
                        pass

            if metadata and uid in metadata:
                m = metadata[uid]
                autocrop = m['autocrop']
                if autocrop[0] is True and m['first_completely_visible_vertebra'] < max(vertebra_labels):
                    print('Autocrop should be false')
                elif highest_disc and autocrop[0] is False:
                    print('Autocrop should be true')


            # Print all labels if there were any issues
            if label_issues:
                print('> all labels: {}'.format(strseq(labels)))
        #         mistakes_list.append(1)
        #     else:
        #         mistakes_list.append(0)
        #
        # df = pd.Dataframe(list(zip(images, mistakes_list)))
        # df.to_csv(path.join(datadir, self.args.dataset, 'mistakes.csv'))
