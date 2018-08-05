'''
Common settings for examples in this directory.
'''
import os

MODULE_DIR = os.path.dirname(__file__)
EXAMPLES_DIR = os.path.dirname(MODULE_DIR)
PACKAGE_DIR = os.path.dirname(EXAMPLES_DIR)
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
