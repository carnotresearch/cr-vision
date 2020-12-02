# verifies that all the required libraries are installed and can 
# be easily imported

# standard libraries
import pathlib
from pathlib import Path
import os
import sys
import threading
import asyncio
import datetime
import traceback

# language support
import pylint
from dataclasses import dataclass

# scientific computing
import numpy
import scipy
import sympy
import scipy.stats

# machine learning
import pandas
import sklearn
from sklearn import datasets
from sklearn import svm
import statsmodels

# visualization
import matplotlib
from matplotlib import pyplot as plt
from bokeh.plotting import figure, output_file, show

# databases
import sqlite3

import rtree

# documentation
import alabaster

# data read/write
import h5py
import xlrd
import xlwt

# networking
import pycurl

# unit testing
import nose
import py.test

# cryptography 
import bcrypt
from Crypto.Hash import SHA256 # pycrypto

# stream oriented programming
import rx
import rx.operators as ops

# utilities
import zope.interface
import zope.event
import click


# computer vision
from PIL import Image # pillow
import cv2
import skimage
import skvideo
import imageio

# deep learning
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, backend, models, utils

# CR-VISION LIBRARY
from cr import vision
from cr.vision.core import colors
from cr.vision import edits
from cr.vision.edits import logo
from cr.vision import geom
from cr.vision.concurrent import WebcamReadStream
from cr.vision import io
import rx.operators as ops
from cr.vision.core import bb
from cr.vision.core import effects
from cr.vision import filters
from cr.vision.crx import step
from cr.vision.crx import EventLoopPlayer
from cr.vision import object_detection as od

# Print the version numbers of various libraries
print("numpy version: {}".format(numpy.__version__))
print("scipy version: {}".format(scipy.__version__))

print("matplotlib version: {}".format(matplotlib.__version__))

print("pandas version: {}".format(pandas.__version__))

print("CV2 version: {}".format(cv2.__version__))
print("skvideo version: {}".format(skvideo.__version__))
print("skimage version: {}".format(skimage.__version__))

print("tensorflow version: {}".format(tensorflow.__version__))


print("rx version: {}".format(rx.__version__))
print("click version: {}".format(click.__version__))


print("nose version: {}".format(nose.__version__))
print("py.test version: {}".format(py.test.__version__))
