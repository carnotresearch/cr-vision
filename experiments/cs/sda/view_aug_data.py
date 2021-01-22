import pathlib

import sda

from cr.vision.plots import plot_images_and_masks


rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'


images = sda.get_dataset(rootdir)

# prepare the training set generator
train_gen = sda.augment_training_set(images)

sample_batch = next(train_gen)

src_images, target_images = sample_batch

print(src_images.shape)

print(target_images.shape)
