
from cr.vision.dl.nets.sgmt.unet import model_unet_ronneberger_2015

input_shape = (572, 572, 1)

model = model_unet_ronneberger_2015(input_shape)
print (model.summary())
