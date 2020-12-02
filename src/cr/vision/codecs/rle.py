"""
Run length encoding of a binary image (in numpy format)

References

* https://www.kaggle.com/paulorzp/run-length-encode-and-decode
* https://www.kaggle.com/stainsby/fast-tested-rle

"""
import numpy as np


def encode(image):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode(mask, shape):
    runs = mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (runs[0:][::2], runs[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
