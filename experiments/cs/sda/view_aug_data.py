from pathlib import Path
from cr.vision.io import images_from_dir
import sda

from cr.vision.plots import plot_images_with_reconstructions


rootdir  = Path(r'D:\datasets\vision\birds\CUB_200_2011\birds_subset_5000')
training = images_from_dir(rootdir / 'training', size=60)
print(training.shape)

# prepare the training set generator
train_gen = sda.augment_training_set(training)

sample_batch = next(train_gen)

src_images, target_images = sample_batch

print(src_images.shape)

print(target_images.shape)

plot_images_with_reconstructions(src_images, target_images)