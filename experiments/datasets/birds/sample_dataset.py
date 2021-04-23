"""

"""
import numpy as np
from pathlib import Path
import shutil
from cr.vision.io import ImagesFromDir

rootdir  = r'D:\datasets\vision\birds\CUB_200_2011\images'
root = Path(rootdir)

rng = np.random.default_rng(2021)
ds = ImagesFromDir(rootdir, size=5000, rng=rng)

paths = ds.sampled_paths

n = len(paths)

n_training = int (n * .6)
n_validation = int (n * .2)
n_test = n - n_training - n_validation
training = paths[:n_training]
n_2 = n_training + n_validation
validation = paths[n_training:n_2]
test = paths[n_2:]
print(f'training: {len(training)}, validation: {len(validation)}, test: {len(test)}')


def copy_to_dir(src_paths, dst_dir):
    print(f'Creating directory: {dst_dir}')
    dst_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = [path.relative_to(root) for path in src_paths]
    names = ['_'.join(path.parts) for path in rel_paths]
    for (name, src_path) in zip(names, src_paths):
        dst_path = dst_dir / name
        shutil.copy(src_path, dst_path)
        print('.', end="", flush=True)
    print()

cub_dir = root.parent
dst_dir = cub_dir / f'birds_subset_{n}'
copy_to_dir(training, dst_dir /  'training')
copy_to_dir(validation, dst_dir /  'validation')
copy_to_dir(test, dst_dir /  'test')