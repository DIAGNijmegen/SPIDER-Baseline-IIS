from pathlib import Path
from tiger.io import read_image
from tiger.screenshots import ScreenshotGenerator, ScreenshotCollage, VertebraColors
import numpy as np


dataset_dir = Path('Z:/experiments/spine_segmentation_t9185/datasets/Final_dataset')
images_path = dataset_dir / 'images'
masks_path = dataset_dir / 'masks'

generated_screenshots = (dataset_dir / 'screenshots').glob('*.jpg')
files = [file.stem for file in generated_screenshots if file.is_file()]

coordinates = (0.4, 0.45, 0.5, 0.55, 0.6)
lut = VertebraColors()
screenshot_generator1 = ScreenshotCollage(
    [ScreenshotGenerator(c, axis=2, window_level=None, overlay_lut=lut) for c in coordinates],
    # window/level is overwritten later
    spacing=3,
    margin=3
)

screenshot_generator2 = ScreenshotCollage(
    [ScreenshotGenerator(c, axis=0, window_level=None, overlay_lut=lut) for c in coordinates],
    # window/level is overwritten later
    spacing=3,
    margin=3
)

rotated_direction_vector = np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 1.]])

for file in masks_path.iterdir():
    if file.stem in files:
        print(str(file.stem + ' is already processed'))
    else:
        mask, headerM = read_image(file)
        image, header = read_image(images_path / str(file.parts[-1]))

        if (header.direction.round() == rotated_direction_vector).all():
            screenshot = screenshot_generator2(image, mask)
        else:
            screenshot = screenshot_generator1(image, mask)
        screenshot_file = dataset_dir / 'screenshots' / str(file.stem + '.jpg')
        screenshot.save(screenshot_file)
        print(file.stem)
