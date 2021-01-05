"""
Helper functions for complicated plots
"""
import numpy as np
import matplotlib.pyplot as plt


from .image import binary_mask_to_rgba

def _get_color_map(images):
    if images.ndim == 3:
        return "gray"
    elif images.ndim == 4:
        # last dim
        last = images.shape[3]
        if last == 3:
            # Normal images with 3 channels
            return "jet"
        elif last == 1:
            # Normal images with 1 channel
            return "gray"
        elif last == 4:
            return "gray"

def plot_images_and_masks(originals, masks, 
    predictions=None,
    num_images_to_plot=10,
    figsize=4,
    alpha=0.5,
    color="red"):
    """
    Plots images in columns 
    """
    num_images = originals.shape[0]
    if num_images_to_plot > num_images:
        num_images_to_plot = num_images
    if  predictions is None:
        # original, mask, overlay
        cols  = 3
    else:
        # original, mask, prediction, overlay
        cols = 4
    # create figure and axes
    figure, axes = plt.subplots(num_images_to_plot, cols,
        figsize=(cols*figsize, num_images_to_plot*figsize),
        squeeze=False)
    # Set the titles of the columns
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("mask", fontsize=15)
    if predictions is None:
        axes[0, 2].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    or_cmap = _get_color_map(originals)
    m_cmap = _get_color_map(masks)
    # now plot the images one by one
    for i in range(num_images_to_plot):
        # original
        original = originals[i]
        axis = axes[i, 0]
        axis.imshow(original, cmap=or_cmap)
        axis.set_axis_off()
        # mask
        mask = masks[i]
        axis = axes[i, 1]
        axis.imshow(mask, cmap=m_cmap)
        axis.set_axis_off()
        if predictions is None:
            # overlay original with mask
            axis = axes[i, 2]
            axis.imshow(original, cmap=or_cmap)
            mask_rgba = binary_mask_to_rgba(mask)
            axis.imshow(mask_rgba, cmap=m_cmap, alpha=alpha)
            axis.set_axis_off()


        # prediction
        # overlay
    plt.show()
