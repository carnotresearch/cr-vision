"""
Use the trained model to predict
"""

import pathlib

import whales

from cr.vision.plots import plot_images_and_masks


ds_dir = pathlib.Path(r'E:\datasets\vision\segmentation\whales')
images, masks = whales.get_dataset(ds_dir, max_samples=4)


model = whales.load_saved_model()
print(model.summary())

predictions = model.predict(images)

plot_images_and_masks(images, masks, predictions, num_images_to_plot=4, figsize=6)
