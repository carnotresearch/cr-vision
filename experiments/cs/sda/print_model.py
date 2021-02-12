
from cr.vision.dl.nets.cs.sda import build_models

patch_size = 8
compression_ratio = 16
input_shape  = (256, 256, 3)
print("Building Models.")
model = build_models(input_shape,
    patch_size=patch_size,
    compression_ratio=compression_ratio)
print("\n\n\nENCODER MODEL")
print(model.encoder.summary())
print("\n\n\nDECODER MODEL")
print(model.decoder.summary())
print("\n\n\nAUTOENCODER MODEL")
print(model.autoencoder.summary())
