'''
Common settings for examples in this directory.
'''
from pathlib import Path

FILE_PATH = Path(__file__).resolve()

MODULE_DIR = FILE_PATH.parent
EXAMPLES_DIR = MODULE_DIR.parent
PACKAGE_DIR = EXAMPLES_DIR.parent
DATA_DIR = PACKAGE_DIR / 'data'
IMAGES_DIR = DATA_DIR / 'images'
VIDEO_DIR = DATA_DIR / 'videos'
