import sys
import os
import glob
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from cr.vision.dl.nets.cs.csresnet import build_models

IN_COLAB = 'google.colab' in sys.modules
GD_EXP_DIR = '/content/drive/MyDrive/work/cr-vision/experiments/cs/csresnet'


CHECKPOINT_FILENAME = 'checkpoint.hdf5'
HISTORY_FILENAME = 'history.json'
SAVED_MODEL_DIR = 'saved_model'
CACHE_DIR = "./data_cache"
if IN_COLAB:
    CHECKPOINT_FILENAME = f'{GD_EXP_DIR}/{CHECKPOINT_FILENAME}'
    HISTORY_FILENAME = f'{GD_EXP_DIR}/{HISTORY_FILENAME}'
    SAVED_MODEL_DIR = f'{GD_EXP_DIR}/{SAVED_MODEL_DIR}'
    CACHE_DIR = f'{GD_EXP_DIR}/data_cache'



def augment_training_set(src_images, target_images=None,
    batch_size=32,
    seed=0):
    if target_images is None:
        target_images = src_images
    print("Preparing augmented training set generator.")
    args=dict(
        rotation_range=10.0,
        height_shift_range=0.02,
        shear_range=5,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant"
        )
    src_datagen = ImageDataGenerator(**args)
    target_datagen = ImageDataGenerator(**args)

    src_datagen.fit(src_images, augment=True, seed=seed)
    target_datagen.fit(target_images, augment=True, seed=seed)

    augmented_src = src_datagen.flow(
        src_images, batch_size=batch_size, shuffle=True, seed=seed)
    augmented_targets = target_datagen.flow(
        target_images, batch_size=batch_size, shuffle=True, seed=seed)

    generator = zip(augmented_src, augmented_targets)
    return generator


def build_callbacks():
    print("Preparing callbacks for model training.")
    callback_checkpoint = ModelCheckpoint(
        CHECKPOINT_FILENAME, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True,
    )
    return [callback_checkpoint]

def init_with_checkpoint(model):
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
        loss='mse')
    return model


def fit_model(model, train_gen, val_images, callbacks,
    steps_per_epoch=200,
    epochs=20):
    print(f'Initiating model training for {epochs} epochs.')
    print(train_gen)
    print(steps_per_epoch, epochs)
    print(val_images.shape)
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,    
        validation_data=(val_images, val_images),
        callbacks=callbacks
    )
    return history

def save_history(history, compression_ratio):
    print('Saving training history.')
    # Make a pandas data frame
    hist_df = pd.DataFrame(history.history)
    filename = f'cr_{compression_ratio}_{HISTORY_FILENAME}'
    with open(filename, mode='w') as f:
        hist_df.to_json(f)

def save_model(model, name, compression_ratio):
    print(f'Saving {name} at compression ration {compression_ratio} in tensorflow SavedModel format.')
    model_dir = f'{SAVED_MODEL_DIR}_{name}_{compression_ratio}'
    model.save(model_dir)


def load_saved_model(name, compression_ratio):
    model_dir = f'{SAVED_MODEL_DIR}_{name}_{compression_ratio}'
    print(f'Loading the saved model from {model_dir}')
    model = keras.models.load_model(model_dir,
        #custom_objects={'iou':sgmt_metrics.iou,
        #'iou_thresholded': sgmt_metrics.iou_thresholded},
        compile=False)
    return model
