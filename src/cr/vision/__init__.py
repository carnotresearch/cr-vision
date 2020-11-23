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


# core capabilities
from .core.colors import *
from .core.contour import *
from .core.cvt_color import *
from .core.effects import *
from .core.histogram import *
from .core.misc import *
from .core.noise_gaussian import *
from .core.noise_snp import *
from .core.opencv import *
from .core.operations import *
from .core.plot import *
from .core.scaling import *
from .core.template import *
from .core.threshold import *
from .core.types import *


# Image Quality Measurements
from . import measure

# Reactive Extensions
from . import crx

# Application Specific Stuff
from .object_tracker import *
from .saliency import *
from .retargeting import *
from .edits import animation
from .apps import *


from . import traffic
from . import faces
from . import pedestrians
from . import video
from . import io

