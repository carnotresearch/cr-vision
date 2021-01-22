"""
Setup model training
"""

import pathlib

import sda

rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'


images = sda.get_dataset(rootdir)


x_train, x_val, y_train, y_val = sda.split_dataset(images)

# prepare the training set generator
train_gen = sda.augment_training_set(x_train, y_train)

# Setup the network to be trained
input_shape = x_train[0].shape


model = sda.build_model(input_shape)


sda.compile_model(model)

# sda.init_with_checkpoint(model)

callbacks = sda.build_callbacks()


history = sda.fit_model(model, train_gen,
    x_val, callbacks,
    steps_per_epoch=20,
    epochs=4)

sda.save_model(model)

sda.save_history(history)


