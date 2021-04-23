
from pathlib import Path
from cr.vision.io import images_from_dir

rootdir  = Path(r'D:\datasets\vision\birds\CUB_200_2011\birds_subset_5000')
training = images_from_dir(rootdir / 'training', size=6)
print(training.shape)
validation = images_from_dir(rootdir / 'validation', size=2)
print(validation.shape)
test = images_from_dir(rootdir / 'test', size=2)
print(test.shape)