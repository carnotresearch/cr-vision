"""
Setup model training
"""
import pathlib

import whales

ds_dir = pathlib.Path(r'E:\datasets\vision\segmentation\whales')
image_ds, mask_ds = whales.get_dataset(ds_dir)


x_train, x_val, y_train, y_val = whales.split_data_set(image_ds, mask_ds)


# prepare the training set generator
train_gen = whales.augment_training_set(x_train, y_train)

# Setup the network to be trained
input_shape = x_train[0].shape

model = whales.get_model(input_shape)

whales.compile_model(model)

callbacks = whales.get_callbacks()

# make sure that 

history = whales.fit_model(model, train_gen,
    x_val, y_val, callbacks,
    steps_per_epoch=4,
    epochs=2)

whales.save_model(model)

whales.save_history(history)

