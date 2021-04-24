
from cr.vision.dl.nets.cs.csresnet import build_models
import csresnet

patch_size = 32
stride_size = 16
compression_ratio = 64
num_res_channels=32
num_res_blocks=2
input_shape  = (256, 256, 3)
print("Building Models.")
model = build_models(input_shape,
    patch_size=patch_size,
    stride_size=stride_size,
    num_res_channels=num_res_channels,
    num_res_blocks=num_res_blocks,
    compression_ratio=compression_ratio)
print("\n\n\nENCODER MODEL")
print(model.encoder.summary())
print("\n\n\nDECODER MODEL")
print(model.decoder.summary())
print("\n\n\nAUTOENCODER MODEL")
print(model.autoencoder.summary())

name = csresnet.form_model_name('autoencoder', patch_size, stride_size, num_res_blocks, num_res_channels, compression_ratio)
csresnet.save_model(model.autoencoder, name)
