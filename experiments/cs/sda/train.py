"""
Setup model training
"""

from pathlib import Path
from cr.vision.io import images_from_dir

import sda

rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'
from cr.vision.io import images_from_dir

rootdir  = Path(r'D:\datasets\vision\birds\CUB_200_2011\birds_subset_5000')
training_set = images_from_dir(rootdir / 'training', size=120)
validation_set = images_from_dir(rootdir / 'validation', size=40)

print(f"training set: {training_set.shape}")
print(f"validation set: {validation_set.shape}")
n_train = training_set.shape[0]
print(f'training set size: {n_train}')

# prepare the training set generator
batch_size = 32
train_gen = sda.augment_training_set(training_set,
    batch_size=batch_size)

# Setup the network to be trained
input_shape = training_set[0].shape

patch_size = 16
compression_ratio = 4
models = sda.build_models(input_shape,
    patch_size=patch_size,
    compression_ratio=compression_ratio)
autoencoder = models.autoencoder


sda.compile_model(autoencoder)

# sda.init_with_checkpoint(model)

callbacks = sda.build_callbacks()

steps_per_epoch = int(n_train / batch_size)
print(f'steps_per_epoch: {steps_per_epoch}')

history = sda.fit_model(autoencoder, train_gen,
    validation_set, callbacks,
    steps_per_epoch=20,
    epochs=4)

sda.save_model(models.autoencoder, 'autoencoder', compression_ratio)
sda.save_model(models.encoder, 'encoder', compression_ratio)
sda.save_model(models.decoder, 'decoder', compression_ratio)

sda.save_history(history, compression_ratio)


