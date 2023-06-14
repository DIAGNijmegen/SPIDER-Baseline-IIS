import csv
import numpy as np

from os import path, makedirs
from time import time
from scipy import ndimage
from skimage import measure, morphology

from tiger.patches import compute_crop
from tiger.screenshots import ScreenshotGenerator, ScreenshotCollage, VertebraColors


class Timer:
    def __init__(self):
        self.start_time = time()

    def elapsed(self):
        current_time = time()
        elapsed = current_time - self.start_time
        self.start_time = current_time
        return elapsed


def flatten_dict(d):
    return [v for vs in d.values() for v in vs]


def compute_distance_weight_matrix(mask, alpha=1, beta=8, omega=6):
    mask = np.asarray(mask)
    if np.any(mask):
        distance_to_border = ndimage.distance_transform_edt(mask > 0) + ndimage.distance_transform_edt(mask == 0)
        weights = alpha + beta * np.exp(-(distance_to_border**2 / omega**2))
        return np.asarray(weights, dtype='float32')
    else:
        return np.ones_like(mask, dtype='float32')


def remove_small_components(image, mask, min_size=1001):
    m = mask > 0

    outside_fov = image <= -2000
    m[outside_fov] = True

    cmap = measure.label(m, background=0)
    cmap[outside_fov] = 0
    labels, sizes = np.unique(cmap[cmap > 0], return_counts=True)

    for label, size in zip(labels, sizes):
        if size < min_size:
            mask[cmap == label] = 0

    mask[outside_fov] = 0
    return mask


def remove_low_intensity_surface_voxels(image, segmentation, intensity_threshold):
    if intensity_threshold > -1000:
        low_intensity_voxels = image < intensity_threshold
        low_intensity_voxels[:, :, 0] = False
        low_intensity_voxels[:, :, -1] = False

        for label in np.unique(segmentation):
            if label == 0:
                continue

            voxels = segmentation == label
            surface_voxels_removed = morphology.binary_erosion(voxels)
            surface_voxels = np.logical_and(voxels, surface_voxels_removed == 0)
            removable_surface_voxels = np.logical_and(surface_voxels, low_intensity_voxels)
            segmentation[removable_surface_voxels] = 0

    return segmentation


def extract_valid_patch_pairs(mask, mask_patch, patch_center):
    """ Extracts two patches of equal size corresponding to the valid area of the supplied patch """
    if mask.ndim == 4:
        crops = [compute_crop(c, s, a) for c, s, a in zip(patch_center, mask_patch.shape[1:], mask.shape[1:])]
        patch1 = mask[:, crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
        patch2 = mask_patch[
            (slice(None),) + tuple(slice(c[2] if c[2] > 0 else None, -c[3] if c[3] > 0 else None) for c in crops)
        ]
    else:
        crops = [compute_crop(c, s, a) for c, s, a in zip(patch_center, mask_patch.shape, mask.shape)]
        patch1 = mask[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
        patch2 = mask_patch[
            tuple(slice(c[2] if c[2] > 0 else None, -c[3] if c[3] > 0 else None) for c in crops)
        ]
    return patch1, patch2


class LearningCurve:
    def __init__(self, filename, legend=('Training', 'Validation')):
        self.filename = filename
        self.cache = []

        # Make sure the output directory exists
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)

        # Writer header for CSV file
        if not path.exists(filename):
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Epoch'] + list(legend))

    def post(self, values, step):
        self.cache.append([step] + values)

    def flush(self):
        if len(self.cache) > 0:
            with open(self.filename, 'a') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                for row in self.cache:
                    writer.writerow(row)
                self.cache.clear()


class Screenshot(ScreenshotCollage):
    def __init__(self, captions, dynamic_window_level=False):
        lut = VertebraColors()
        wl = None if dynamic_window_level else (400, 50)
        generators = [
            ScreenshotGenerator(axis=0, window_level=wl, overlay_lut=lut, caption=caption)
            for caption in captions
        ]
        super().__init__(generators)

    def save(self, filename, image, mask=None):
        # Generator screenshot
        screenshot = self(image, mask)

        # Make sure target directory exists
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)

        # Save screenshot
        screenshot.save(filename)
