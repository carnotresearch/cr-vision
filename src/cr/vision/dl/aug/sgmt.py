"""
Augmentation functions for image segmentation tasks
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_train_2d(images, masks, 
    batch_size=32,
    seed=0, 
    args=dict(
        rotation_range=10.0,
        height_shift_range=0.02,
        shear_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant"
        )):
    image_datagen = ImageDataGenerator(**args)
    mask_datagen = ImageDataGenerator(**args)
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    augmented_images = image_datagen.flow(
        images, batch_size=batch_size, shuffle=True, seed=seed)
    augmented_masks = mask_datagen.flow(
        masks, batch_size=batch_size, shuffle=True, seed=seed)
    generator = zip(augmented_images, augmented_masks)
    return generator


def augment_test_2d(images, masks, 
    batch_size=32, 
    seed=0,
    args=dict(
        rotation_range=10.0,
        height_shift_range=0.02,
        shear_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant"
        )):
    image_datagen = ImageDataGenerator(**args)
    mask_datagen = ImageDataGenerator(**args)
    image_datagen.fit(images, augment=False, seed=seed)
    mask_datagen.fit(masks, augment=False, seed=seed)
    augmented_images = image_datagen.flow(
        images, batch_size=batch_size, shuffle=True, seed=seed)
    augmented_masks = mask_datagen.flow(
        masks, batch_size=batch_size, shuffle=True, seed=seed)
    generator = zip(augmented_images, augmented_masks)
    return generator
