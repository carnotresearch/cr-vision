import pathlib

import sda

from cr.vision.plots import plot_images_with_reconstructions


rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'

images = sda.get_dataset(rootdir)

model = sda.load_saved_model()
print(model.summary())

reconstructions = model.predict(images)

plot_images_with_reconstructions(images, reconstructions)
