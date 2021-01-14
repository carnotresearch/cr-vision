import sys
import os
import glob
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.model_selection import train_test_split

IN_COLAB = 'google.colab' in sys.modules
GD_EXP_DIR = '/content/drive/MyDrive/work/cr-vision/experiments/segmentation/whales'

CHECKPOINT_FILENAME = 'sgmt_checkpoint.hdf5'
HISTORY_FILENAME = 'history.json'
SAVED_MODEL_DIR = 'sgmt_model'
if IN_COLAB:
    CHECKPOINT_FILENAME = f'{GD_EXP_DIR}/{CHECKPOINT_FILENAME}'
    HISTORY_FILENAME = F'{GD_EXP_DIR}/{HISTORY_FILENAME}'
    SAVED_MODEL_DIR = F'{GD_EXP_DIR}/{SAVED_MODEL_DIR}'

from cr import vision
from cr.vision.dl.aug import sgmt as sgmt_aug
from cr.vision.dl.metrics import sgmt as sgmt_metrics

from cr.vision.dl.nets.sgmt.unet import model_custom_unet

def get_original_and_mask_files(dataset_dir):
    mask_paths = list(dataset_dir.glob("*.png"))
    original_paths = [file.with_suffix('.jpg') for file in mask_paths]
    return original_paths, mask_paths


def get_dataset(dataset_dir, max_samples=None):
    print('Loading the dataset.')
    original_paths, mask_paths = get_original_and_mask_files(dataset_dir)
    image_list = []
    mask_list = []
    i = 1
    for image_path, mask_path in zip(original_paths, mask_paths):
        # print(f'[{i}] Processing {image_path.name}')
        print(".", end="")
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
    print("")

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

def split_data_set(images, masks,
    test_size=0.9,
    random_state=0):
    print("Splitting training and testing datasets.")
    x_train, x_val, y_train, y_val = train_test_split(
        images, masks, 
        test_size=test_size, 
        random_state=random_state)
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    return x_train, x_val, y_train, y_val

def augment_training_set(x_train, y_train):
    print("Preparing augmented training set generator.")
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
    print("Building model.")
    model = model_custom_unet(input_shape,
        filters=32,
        use_batch_norm=True,
        use_bias=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        num_layers=4)
    return model


def get_callbacks():
    print("Preparing callbacks for model training.")
    callback_checkpoint = ModelCheckpoint(
        CHECKPOINT_FILENAME, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )
    return [callback_checkpoint]

def init_with_checkpoint_weights(model):
    print("Checking if there is any previously saved checkpoint.")
    path = pathlib.Path(CHECKPOINT_FILENAME)
    if path.exists():
        print(f'Loading weights from {CHECKPOINT_FILENAME}')
        model.load_weights(CHECKPOINT_FILENAME)
    else:
        print('No previously saved checkpoint to load.')


def compile_model(model):
    print("Compiling model for training.")
    model.compile(
        optimizer=Adam(), 
        loss='binary_crossentropy',
        metrics=[sgmt_metrics.iou, sgmt_metrics.iou_thresholded]
        )
    return model


def fit_model(model, train_gen, 
    val_images, val_masks, 
    callbacks,
    steps_per_epoch=200,
    epochs=20):
    print(f'Initiating model training for {epochs} epochs.')
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,    
        validation_data=(val_images, val_masks),
        callbacks=callbacks
    )
    return history


def save_history(history):
    print('Saving training history.')
    # Make a pandas data frame
    hist_df = pd.DataFrame(history.history) 
    with open(HISTORY_FILENAME, mode='w') as f:
        hist_df.to_json(f)


def save_model(model):
    print('Saving trained model in tensorflow SavedModel format.')
    model.save(SAVED_MODEL_DIR)
