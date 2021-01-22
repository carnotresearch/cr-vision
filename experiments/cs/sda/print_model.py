
from cr.vision.dl.nets.cs.sda import model_sda

input_shape  = (256, 256, 3)

model = model_sda(input_shape)
print(model.summary())

for layer in model.layers:
    print(f'{layer.name} trainable: {layer.trainable}')