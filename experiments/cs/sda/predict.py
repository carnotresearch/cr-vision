import pathlib

import sda

from cr.vision.plots import plot_images_with_reconstructions
from skimage.metrics import peak_signal_noise_ratio

rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'

dataset = sda.get_dataset(rootdir)

images = dataset.test_set

model = sda.load_saved_model()
print(model.summary())

reconstructions = model.predict(images)

n = images.shape[0]

for i in range(n):
    src = images[i]
    dst = reconstructions[i]
    psnr = peak_signal_noise_ratio(src, dst)
    print(f"[{i+1}] PSNR : {psnr} dB")


plot_images_with_reconstructions(images[:10], reconstructions[:10])

