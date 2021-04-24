import pathlib
import numpy as np
import imageio
import itertools
try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property


from cr.vision.core.scaling import resize_crop
from cr.vision.core.cvt_color import gray_to_rgb

def is_gray(image):
    return (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)

def read_images(paths, target_width, target_height):
    images = []
    for path in paths:
        image = imageio.imread(path)
        if is_gray(image):
            print(f'{path}, {image.shape}')
            # make sure that image is rgb (and not gray scale)
            image = gray_to_rgb(image)
            print(f'shape after conversion {image.shape}')
        image = resize_crop(image, target_width, target_height)
        images.append(image)
    return np.array(images)


# Set of extensions
EXTENSIONS = {'.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG'}

class ImagesFromDir:
    def __init__(self, 
        rootdir, 
        size=10,
        width=256,
        height=256, 
        cache=None, 
        preprocess=None,
        force=False,
        validation=0.2,
        test=0.2,
        rng=np.random.default_rng()):
        self.rootdir = rootdir
        self.cache = cache
        self.size = size
        self.width = width
        self.height = height
        self.force = force
        extra = validation + test
        self.n_train_split = int( (1 - extra) * size)
        self.n_val_split = int((1 - test) * size)
        if preprocess is None:
            preprocess = lambda x : x / 255
        self.preprocess = preprocess
        self.rng = rng

    @cached_property
    def all_paths(self):
        rootdir = pathlib.Path(self.rootdir)
        paths = rootdir.glob('**/*')
        images = [path for path in paths if path.is_file() and path.suffix in EXTENSIONS]
        # to do verify extension
        return images
    
    @cached_property
    def sampled_paths(self):
        if not self.force and self.cache is not None and 'sampled_paths' in self.cache:
            print("Reading from cache")
            return self.cache['sampled_paths']
        all_paths = self.all_paths
        sample = self.rng.choice(all_paths, size=self.size, replace=False)
        if self.cache is not None:
            print('Saving to cache')
            self.cache['sampled_paths'] = sample
        return sample
    
    @cached_property
    def training_paths(self):
        paths = self.sampled_paths
        return paths[:self.n_train_split]
    
    @cached_property
    def validation_paths(self):
        paths = self.sampled_paths
        return paths[self.n_train_split:self.n_val_split]

    @cached_property
    def test_paths(self):
        paths = self.sampled_paths
        return paths[self.n_val_split:]
    
    @cached_property
    def training_set(self):
        images = read_images(self.training_paths, self.width, self.height)
        images = self.preprocess(images)
        return images

    @cached_property
    def validation_set(self):
        images = read_images(self.validation_paths, self.width, self.height)
        images = self.preprocess(images)
        return images

    @cached_property
    def test_set(self):
        images = read_images(self.test_paths, self.width, self.height)
        images = self.preprocess(images)
        return images


def images_from_dir(dir_path,
        width=256,
        height=256,
        size=None,
        preprocess=None,
        include_paths = False
    ):
    rootdir = pathlib.Path(dir_path)
    paths = rootdir.glob('**/*')
    if size is not None:
        paths = itertools.islice(paths, size)
    image_paths = [path for path in paths if path.is_file() and path.suffix in EXTENSIONS]
    images = read_images(image_paths, width, height)
    if preprocess is None:
        preprocess = lambda x : x / 255
    images = preprocess(images)
    if include_paths:
        return images, image_paths
    return images
