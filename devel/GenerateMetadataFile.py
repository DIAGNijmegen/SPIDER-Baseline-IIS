import numpy as np
import json

from os import path
from glob import glob
from argparse import ArgumentParser
from collections import OrderedDict

from tiger.io import read_image

from config import datadir


class Module:
    def __init__(self, argv):
        parser = ArgumentParser()
        parser.add_argument('dataset')
        parser.add_argument('--overwrite', action='store_true')
        parser.add_argument('--filetype', type=str, default='mha')
        parser.add_argument('--subset', type=str, default='training')
        parser.add_argument('--slice-order-reversed', action='store_true')
        parser.add_argument('--rescale-intercept', type=int, default=0)
        parser.add_argument('--all-completely-visible', action='store_true')
        parser.add_argument('--autocrop', type=bool, nargs='*')
        parser.add_argument('--modality', type=str, default='CT')
        self.args = parser.parse_args(argv)

    def execute(self):
        # Does the dataset exist?
        if not path.exists(path.join(datadir, self.args.dataset)):
            print('Dataset "{}" does not exist in {}'.format(self.args.dataset, datadir))
            return

        # Does this metadata file already exist?
        outfile = path.join(datadir, self.args.dataset, 'metadata.json')
        if path.exists(outfile) and not self.args.overwrite:
            print('Metadata file already exists, specify --overwrite to replace it')
            return

        # Search for image files with corresponding mask and build metadata
        metadata = []

        images = glob(path.join(datadir, self.args.dataset, 'images', '*.{}'.format(self.args.filetype)))
        for imagefile in sorted(images):
            uid = path.basename(imagefile)[:-(1 + len(self.args.filetype))]
            print(uid)

            # Check if mask file exists
            maskfile = path.join(datadir, self.args.dataset, 'masks', path.basename(imagefile))
            has_mask = path.exists(maskfile)

            # Build metadata dictionary
            m = OrderedDict()
            m['identifier'] = uid
            m['subset'] = self.args.subset
            m['filetype'] = self.args.filetype
            m['slice-order-reversed'] = self.args.slice_order_reversed
            m['rescale-intercept'] = self.args.rescale_intercept
            m['modality'] = self.args.modality

            # Read image to determine range of visible vertebrae?
            if self.args.all_completely_visible and has_mask:
                mask, _ = read_image(maskfile)
                labels = np.unique(mask[np.logical_and(mask > 0, mask < 100)])
                m['first_completely_visible_vertebra'] = int(min(labels))
                m['last_completely_visible_vertebra'] = int(max(labels))
                print(' > range of labels: {} - {}'.format(
                    m['first_completely_visible_vertebra'], m['last_completely_visible_vertebra']
                ))

            if self.args.autocrop is not None:
                m['autocrop'] = self.args.autocrop

            m['ignore_slices_top'] = 0
            m['ignore_slices_bottom'] = 0

            metadata.append(m)
            print(' > success')

        # Write metadata to json file
        with open(outfile, 'w') as fp:
            json.dump(metadata, fp, indent=2, separators=(',', ': '))
