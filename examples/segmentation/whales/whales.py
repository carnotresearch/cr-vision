import pathlib
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import cv2
import imageio

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD


from cr import vision
from cr.vision.dl.aug import sgmt as sgmt_aug
from cr.vision.dl.metrics import sgmt as sgmt_metrics

from cr.vision.dl.nets.sgmt.unet import model_custom_unet

def get_original_and_mask_files(dataset_dir):
    mask_paths = list(dataset_dir.glob("*.png"))
    original_paths = [file.with_suffix('.jpg') for file in mask_paths]
    return original_paths, mask_paths


def get_dataset(dataset_dir, max_samples=None):
    original_paths, mask_paths = get_original_and_mask_files(dataset_dir)
    image_list = []
    mask_list = []
    i = 1
    for image_path, mask_path in zip(original_paths, mask_paths):
        print(f'[{i}] Processing {image_path.name}')

        # read image as RGB
        image = imageio.imread(image_path)
        image = vision.resize(image, 384, 384)
        image_list.append(image)
        
        # read mask in gray scale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = vision.resize(mask, 384, 384)
        mask_list.append(mask)

        if max_samples is not None and i >= max_samples: 
            break
        i += 1

    image_ds = np.asarray(image_list, dtype=np.float32)
    mask_ds = np.asarray(mask_list, dtype=np.float32)
    print(image_ds.shape)
    print(mask_ds.shape)
    print(image_ds.max(), mask_ds.max())

    # normalize to (0,1) range
    image_ds = image_ds / image_ds.max()
    mask_ds = mask_ds / mask_ds.max()
    mask_ds = np.expand_dims(mask_ds, axis=-1)

    return image_ds, mask_ds

def augment_training_set(x_train, y_train):
    gen = sgmt_aug.augment_train_2d(x_train, y_train,
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
    return gen

def get_model(input_shape):
    model = model_custom_unet(input_shape,
        filters=32,
        use_batch_norm=True,
        use_bias=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        num_layers=4)
    return model


def checkpoints():
    model_filename = 'segm_model_v3.h5'
    callback_checkpoint = ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )
    return callback_checkpoint


def compile_model(model):
    model.compile(
        optimizer=Adam(), 
        #optimizer=SGD(lr=0.01, momentum=0.99),
        loss='binary_crossentropy',
        #loss=jaccard_distance,
        metrics=[sgmt_metrics.iou, sgmt_metrics.iou_thresholded]
        )
    return model