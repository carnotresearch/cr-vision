'''
indigits.vision package
'''
# pylint: disable=W0401

# Library setup
from .setup import *
# Numpy essentials
from .matrix import *
from .vector import *
from .array import *
from .errors import *

# IO
from .io import *

# Basic Image Processing
from .core import *
from .core import effects
from .geom_transform import *
from .filters import *
from .edits import *
from .concurrent import *

# Image Quality Measurements
from . import measure

# Reactive Extensions
from . import crx

# Video Processing
from .video import *
from .motion import  *

# Application Specific Stuff
from .object_tracker import *
from .saliency import *
from .retargeting import *
from .edits import animation
from .apps import *


from . import traffic
from . import faces
from . import pedestrians


