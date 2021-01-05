
from cr.vision.dl.nets.sgmt.unet import model_custom_unet

input_shape = (256, 256, 1)

model = model_custom_unet(input_shape)
print (model.summary())
