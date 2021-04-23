from pathlib import Path
from cr.vision.io import images_from_dir

import sda

from cr.vision.plots import plot_images_with_reconstructions
from skimage.metrics import peak_signal_noise_ratio

rootdir  = Path(r'D:\datasets\vision\birds\CUB_200_2011\birds_subset_5000')
images = images_from_dir(rootdir / 'test', size=40)
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

