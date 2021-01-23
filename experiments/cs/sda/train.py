"""
Setup model training
"""

import pathlib

import sda

rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'


dataset = sda.get_dataset(rootdir)
training_set = dataset.training_set
validation_set = dataset.validation_set

# prepare the training set generator
train_gen = sda.augment_training_set(training_set, training_set)

# Setup the network to be trained
input_shape = training_set[0].shape


model = sda.build_model(input_shape)


sda.compile_model(model)

# sda.init_with_checkpoint(model)

callbacks = sda.build_callbacks()


history = sda.fit_model(model, train_gen,
    validation_set, callbacks,
    steps_per_epoch=20,
    epochs=4)

sda.save_model(model)

sda.save_history(history)


