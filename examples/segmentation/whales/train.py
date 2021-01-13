"""
Setup model training
"""
import pathlib
from sklearn.model_selection import train_test_split

import whales

ds_dir = pathlib.Path(r'E:\CarnotResearchDrive\datasets\segmentation\whales')
image_ds, mask_ds = whales.get_dataset(ds_dir)

x_train, x_val, y_train, y_val = train_test_split(image_ds, mask_ds, test_size=0.9, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# prepare the training set generator
train_gen = whales.augment_training_set(x_train, y_train)

# Setup the network to be trained
input_shape = x_train[0].shape

model = whales.get_model(input_shape)

whales.compile_model(model)

checkpoints = whales.checkpoints()

history = model.fit_generator(
    train_gen,
    steps_per_epoch=200,
    epochs=50,    
    validation_data=(x_val, y_val),
    callbacks=[checkpoints]
)
