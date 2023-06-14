import numpy as np
import json
import re

from os import path
from collections import defaultdict
from glob import glob
from fnmatch import fnmatch

from tiger.io import read_image
from tiger.resampling import resample_image, resample_mask, reorient_image, normalize_direction_simple
from tiger.patches import PatchExtractor3D
from tiger.masks import BoundingBox
from tiger.random import random_item

from config import working_resolution


class LazyLoadingStatus:
    def __init__(self, filename):
        self.filename = filename
        self.available = path.exists(filename)
        self.loaded = False


class VolumeNotAvailableError(Exception):
    pass


class AbstractDataset:
    def __init__(self):
        self.scans = []
        self.subsets = defaultdict(list)

    def __len__(self):
        return len(self.scans)

    def random_scan(self, subset=None):
        scans = self.scans if subset is None else self.subsets[subset]
        return random_item(scans)

    def remove_subset(self, subset):
        if subset in self.subsets:
            for scan in self.subsets[subset]:
                self.scans.remove(scan)
            del self.subsets[subset]

    def remove_subsets(self, subsets=None, remove_others=False):
        if remove_others:
            subsets = self.subsets.keys() - set(subsets)

        for subset in subsets:
            self.remove_subset(subset)

    def rename_subset(self, subset, new_name):
        if subset not in self.subsets:
            raise ValueError('Subset "{}" does not exist, cannot rename non-existing subset'.format(subset))

        self.subsets[new_name].extend(self.subsets[subset])
        del self.subsets[subset]

    def items(self, subset=None):
        scans = self.scans if subset is None else self.subsets[subset]
        for scan in scans:
            yield self, scan


class DatasetCollection(AbstractDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = []
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset):
        self.datasets.append(dataset)

        added_scans = set()

        # Add scans that are in a subset
        for subset, scans in dataset.subsets.items():
            for scan in scans:
                self.scans.append(scan)
                self.subsets[subset].append(scan)
                added_scans.add(scan)

        # Add remaining scans that are not in a subset
        for scan in dataset.scans:
            if scan not in added_scans:
                self.scans.append(scan)

    def remove_subset(self, subset):
        super().remove_subset(subset)
        for dataset in self.datasets:
            dataset.remove_subset(subset)

    def rename_subset(self, subset, new_name):
        super().rename_subset(subset, new_name)
        for dataset in self.datasets:
            try:
                dataset.rename_subset(subset, new_name)
            except ValueError:
                pass  # this dataset does not have any scans in this subset, which is okay

    def items(self, subset=None):
        for dataset in self.datasets:
            yield from dataset.items(subset)


class Dataset(AbstractDataset):
    def __init__(self, identifier, datadir, metadata=None, subset=None, originals=False):
        super().__init__()
        self.identifier = identifier

        basedir = path.join(datadir, identifier)
        imagedir = path.join(basedir, 'images')
        maskdir = path.join(basedir, 'masks')

        # Build list with metadata of all the available scans
        metadata_file = path.join(basedir, 'metadata.json')
        if path.exists(metadata_file):
            with open(metadata_file) as fp:
                metadata_entries = json.load(fp)
        else:
            metadata_entries = []
            for filetype in ('mha', 'mhd'):
                # Search for image files
                image_files = glob(path.join(imagedir, '*.' + filetype))
                if len(image_files) == 0:
                    continue

                # Generate fake metadata for each image
                for filename in sorted(image_files):
                    scan_identifier = path.basename(filename)[:-(len(filetype) + 1)]
                    entry = {
                        'identifier': scan_identifier,
                        'filetype': filetype,
                    }

                    if metadata is not None:
                        for key in metadata:
                            entry[key] = metadata[key]

                    metadata_entries.append(entry)

        # Create scan instances for all scans in the metadata list
        for metadata in metadata_entries:
            if subset is not None:
                metadata['subset'] = subset

            scan = Scan(metadata, imagedir, maskdir, originals=originals)
            self.scans.append(scan)
            if scan.subset is not None:
                self.subsets[scan.subset].append(scan)


class CrossValidation(AbstractDataset):
    def __init__(self, dataset, fold):
        super().__init__()

        try:
            self.identifier = dataset.identifier
        except AttributeError:
            pass

        for scan in dataset.scans:
            self.scans.append(scan)
            if scan.fold is None:
                if scan.subset is not None:
                    self.subsets[scan.subset].append(scan)
            elif fold <= 0:
                self.subsets['training'].append(scan)
                self.subsets['validation'].append(scan)
            elif scan.fold == fold:
                self.subsets['validation'].append(scan)
            else:
                self.subsets['training'].append(scan)


class Scan:
    def __init__(self, metadata, imagedir, maskdir, originals=False):
        self.identifier = metadata['identifier']

        if 'filetype' not in metadata:
            metadata['filetype'] = 'mha'

        if 'modality' not in metadata:
            metadata['modality'] = 'CT'

        self.metadata = metadata
        self.cropping = True
        self.originals = originals

        # Determine subset and potentially also the fold for n-fold cross-validation
        try:
            self.subset = metadata['subset']
        except KeyError:
            self.subset = None

        try:
            if self.subset.startswith('fold'):
                self.fold = int(self.subset[4:])
            else:
                self.fold = None
        except (ValueError, AttributeError):
            self.fold = None

        # Initialize storage variables for image and mask
        self._image_file = LazyLoadingStatus(path.join(imagedir, '{}.{}'.format(self.identifier, metadata['filetype'])))
        self._mask_file = LazyLoadingStatus(path.join(maskdir, '{}.{}'.format(self.identifier, metadata['filetype'])))

        self._image = None
        self._image_patch_extractor = None
        self._original_image = None
        self._initial_shape = None
        self._header = None
        self._initial_header = None
        self._original_header = None
        self._mask = None
        self._mask_patch_extractor = None
        self._original_mask = None
        self._vertebrae = None

        try:
            self._complete_vertebrae = set(range(
                metadata['first_completely_visible_vertebra'],
                metadata['last_completely_visible_vertebra'] + 1
            ))
        except (KeyError, TypeError):
            self._complete_vertebrae = None

    def _flip_volume(self, volume):
        try:
            if self.metadata['slice_order_reversed']:
                return volume[:, :, ::-1]
        except KeyError:
            pass

        return volume

    def _crop_volume(self, volume):
        if not self.cropping:
            return volume

        try:
            offset_top = self.metadata['ignore_slices_top']
            offset_bottom = self.metadata['ignore_slices_bottom']
        except KeyError:
            return volume

        if offset_top == 0 and offset_bottom == 0:
            return volume

        cropped = np.copy(volume)
        if offset_top > 0 and offset_bottom > 0:
            cropped = cropped[:, :, offset_bottom:-offset_top]
        elif offset_top > 0:
            cropped = cropped[:, :, :-offset_top]
        elif offset_bottom > 0:
            cropped = cropped[:, :, offset_bottom:]
        return cropped

    def _load_image_and_header(self):
        if not self._image_file.available:
            raise VolumeNotAvailableError('The image for scan "{}" is not available'.format(self.identifier))

        # Load image into memory
        image, header = read_image(self._image_file.filename)
        image = image.astype('int16', copy=False)

        self._initial_shape = image.shape
        self._initial_header = header.copy()

        # change the direction to default [1 0 0 0 1 0 0 0 1]
        if not header.has_default_direction():
            image, header = reorient_image(image, header)
            # image, header = normalize_direction_simple(image, header)

        # Normalize image
        try:
            intercept = self.metadata['rescale_intercept']
        except KeyError:
            intercept = 0

        image = np.clip(image + intercept, -1000, 3096).astype('int16', copy=False)

        if self.metadata['modality'] == 'MR':
            # Normalize image values to the range [5%, 95%] percentile, mapped to the range [-1000, 3096]
            p5, p95 = np.percentile(image, (5, 95))
            image = (image.astype('float32') - p5.astype('float32')) / (p95 - p5).astype('float32')
            image = np.clip(image * 4096 - 1000, -1000, 3096).astype('int16')

        image = self._flip_volume(image)
        image = self._crop_volume(image)

        # Make a copy of the image and header at the original resolution
        if not self.originals:
            self._original_image = None
        else:
            self._original_image = np.copy(image)
        self._original_header = header.copy()

        # Rescale image to working resolution
        image = resample_image(image, header['spacing'], working_resolution, order=0, prefilter=True)
        header['spacing'] = working_resolution

        # Store image and header
        self._image = image
        self._image_patch_extractor = PatchExtractor3D(image, pad_value=-2000, dtype='float32')
        self._header = header
        self._image_file.loaded = True

    def _load_mask(self):
        if not self._mask_file.available:
            raise VolumeNotAvailableError('The mask for scan "{}" is not available'.format(self.identifier))

        # We need the original spacing to resample the image correctly to working resolution
        # Access it already now as it may raise an exception if the image is not available
        spacing = self.original_header['spacing']

        # Load mask into memory
        mask, mask_header = read_image(self._mask_file.filename)
        mask = mask.astype('uint8', copy=False)

        # Check whether mask and image have the same original coordinate system
        if mask_header != self._initial_header:
            print(f'Image and mask of scan {self.identifier} have different coordinate systems')

        # change the direction to default [1 0 0 0 1 0 0 0 1]
        if not mask_header.has_default_direction():
            mask, _ = reorient_image(mask, self._initial_header, interpolation='nearest')
            # mask, _ = normalize_direction_simple(mask, self._initial_header)

        # Normalize mask
        mask = self._flip_volume(mask)
        mask = self._crop_volume(mask)

        # Resample to working resolution
        if not self.originals:
            self._original_mask = None
        else:
            self._original_mask = np.copy(mask)
        mask = resample_mask(mask, spacing, working_resolution)

        # Store mask and list of labels
        self._vertebrae = []
        for label in sorted(np.unique(mask)):
            if label <= 0 or label >= 100:
                continue
            try:
                complete = label in self._complete_vertebrae
            except TypeError:
                complete = None
            self._vertebrae.append(Vertebra(label, mask, complete))

        self._mask = mask
        self._mask_patch_extractor = PatchExtractor3D(mask, pad_value=0, dtype='float32')
        self._mask_file.loaded = True
        try:
            self.highest_disc = min(np.unique(self._mask[self._mask > 100]))
        except ValueError:
            self.highest_disc = None

    @property
    def image(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._image

    @property
    def shape(self):
        return self.image.shape

    @property
    def image_patches(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._image_patch_extractor

    @property
    def original_image(self):
        if not self.originals:
            raise RuntimeError("Original volumes are not available")

        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._original_image

    @property
    def image_file(self):
        if self._image_file.available:
            return self._image_file.filename
        else:
            return None

    @property
    def original_shape(self):
        return self.original_image.shape

    @property
    def initial_shape(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._initial_shape

    @property
    def header(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._header

    @property
    def original_header(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._original_header

    @property
    def initial_header(self):
        if not self._image_file.loaded:
            self._load_image_and_header()
        return self._initial_header

    @property
    def mask(self):
        if not self._mask_file.loaded:
            self._load_mask()
        return self._mask

    @property
    def mask_patches(self):
        if not self._mask_file.loaded:
            self._load_mask()
        return self._mask_patch_extractor

    @property
    def original_mask(self):
        if not self.originals:
            raise RuntimeError("Original volumes are not available")

        if not self._mask_file.loaded:
            self._load_mask()
        return self._original_mask

    @property
    def mask_file(self):
        if self._mask_file.available:
            return self._mask_file.filename
        else:
            return None

    @property
    def vertebrae(self):
        if not self._mask_file.loaded:
            self._load_mask()
        return self._vertebrae

    def random_vertebra(self):
        return random_item(self.vertebrae)

    def __len__(self):
        return len(self.vertebrae)

    @property
    def loaded(self):
        return self._image_file.loaded or self._mask_file.loaded

    def unload(self):
        self._image_file.loaded = False
        self._mask_file.loaded = False

        self._image = None
        self._image_patch_extractor = None
        self._original_image = None
        self._header = None
        self._original_header = None
        self._mask = None
        self._mask_patch_extractor = None
        self._original_mask = None
        self._vertebrae = None

    def matches(self, expr):
        if '/' in expr:
            # Multiple expressions, separated by /, only one of them has to be a match
            for e in expr.split('/'):
                if self.matches(e):
                    return True
            return False
        elif '+' in expr:
            # Multiple expressions, separated by +, all of them have to be a match
            for e in expr.split('+'):
                if not self.matches(e):
                    return False
            return True
        elif expr[0] in ('>', '<', '=', '!'):
            # Expressions on number ranges require that the identifier contains an integer
            # Supported expressions are: >, >=, <, <=, ==, !=
            match = re.search('([0-9]+)', self.identifier)
            if match is None:
                return False

            n = int(match.group(1))
            if expr.startswith('=='):
                return n == int(expr[2:])
            elif expr.startswith('!='):
                return n != int(expr[2:])
            elif expr.startswith('='):
                return n == int(expr[2:])
            elif expr.startswith('>='):
                return n >= int(expr[2:])
            elif expr.startswith('>'):
                return n > int(expr[1:])
            elif expr.startswith('<='):
                return n <= int(expr[2:])
            elif expr.startswith('<'):
                return n < int(expr[1:])
        else:
            # Default expressions, supported are wildcards like * and ?
            return fnmatch(self.identifier, expr)

    def autocrop(self, crop_top=None, crop_bottom=None, force=False, normalize_direction=False):
        # Check if there is anything to do at all
        try:
            autocrop = self.metadata['autocrop']
        except KeyError:
            pass
        else:
            if autocrop is True or autocrop is False:
                autocrop = [autocrop, autocrop]

            if crop_top is None:
                crop_top = autocrop[0]
            if crop_bottom is None:
                crop_bottom = autocrop[1]

        if not (crop_top or crop_bottom) or not self._mask_file.available:
            return

        # Load mask manually
        if self._mask_file.loaded and self._original_mask is not None:
            mask = self._original_mask
        else:
            mask, header = read_image(self._mask_file.filename)
            mask = mask.astype('uint8', copy=False)
            if not header.has_default_direction():
                if normalize_direction:
                    # mask, _ = normalize_direction_simple(mask, header)
                    mask, _ = reorient_image(mask, header)
                else:
                    mask, _ = reorient_image(mask, header)# todo THIS IS A QUICK FIX
                    # mask, _ = normalize_direction_simple(mask, header)
            mask = self._flip_volume(mask)

        # Metadata available on complete vertebrae?
        if self._complete_vertebrae is None:
            self._complete_vertebrae = set(l for l in sorted(np.unique(mask)) if l > 0)

        # Determine offsets
        if crop_top:
            min_label = min(self._complete_vertebrae)
            if min_label > 1 or force:
                z = np.nonzero(mask == min_label)[2]
                mean_z = int(np.round((np.max(z) + np.min(z)) / 2))
                self.metadata['ignore_slices_top'] = mask.shape[2] - mean_z
                self._complete_vertebrae.discard(min_label)

        if crop_bottom:
            max_label = max(self._complete_vertebrae)
            if max_label < 24 or force:
                z = np.nonzero(mask == max_label)[2]
                mean_z = int(np.round((np.max(z) + np.min(z)) / 2))
                self.metadata['ignore_slices_bottom'] = mean_z
                self._complete_vertebrae.discard(max_label)

        # Unload data to make sure that the new offsets are applied
        self.unload()


class Vertebra:
    def __init__(self, label, mask, completely_visible):
        self.label = label
        vertebra_mask = mask == label
        self.size = np.count_nonzero(vertebra_mask)
        self.bounding_box = BoundingBox(vertebra_mask)
        self.completely_visible = completely_visible

        self.disc_label = label + 200
        self.disc_size = np.count_nonzero(mask == self.disc_label)

    def __len__(self):
        return self.size

    @property
    def has_disc(self) -> bool:
        return self.disc_size > 0
