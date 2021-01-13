import pathlib
import numpy as np

from cr.vision.plots import plot_images_and_masks
from cr.vision.dl.aug import sgmt as aug_sgmt

from whales import get_dataset

ds_dir = pathlib.Path(r'E:\CarnotResearchDrive\datasets\segmentation\whales')
print(ds_dir)

image_ds, mask_ds = get_dataset(ds_dir, max_samples=10)


# Make everything 4D (images, height, width, channels)
print(image_ds.max(), mask_ds.max())
print(image_ds.shape, mask_ds.shape)

plot_images_and_masks(image_ds, mask_ds, num_images_to_plot=4, figsize=6)

train_gen = aug_sgmt.augment_train_2d(image_ds, mask_ds,
    args = dict(
        rotation_range=5.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
        ))


sample_batch = next(train_gen)

images, masks = sample_batch


plot_images_and_masks(images, masks, num_images_to_plot=4, figsize=6)
