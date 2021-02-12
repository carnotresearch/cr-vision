import pathlib

import sda

from cr.vision.plots import plot_images_with_reconstructions
from skimage.metrics import peak_signal_noise_ratio

rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'

dataset = sda.get_dataset(rootdir, 
    size=200,
    validation=0.2,
    test=0)

images = dataset.validation_set

compression_ratio=4
model = sda.load_saved_model('autoencoder', compression_ratio)
print(model.summary())

print(f"images: {images.shape}")

reconstructions = model.predict(images)

n = images.shape[0]


for i in range(n):
    src = images[i]
    dst = reconstructions[i]
    psnr = peak_signal_noise_ratio(src, dst)
    print(f"[{i+1}] PSNR : {psnr} dB")


plot_images_with_reconstructions(images[:10], reconstructions[:10])

