from pathlib import Path
import urllib
import requests
from dataclasses import dataclass


_INITIALIZED = False

VISION_DIR = ''


def _initialize():
    global _INITIALIZED
    if _INITIALIZED:
        return
    # print("Initializing CR-VISION")
    home_dir = Path.home()
    vision_dir = home_dir / '.cr-vision'
    # print(home_dir, vision_dir)
    # Make sure that vision directory exists
    vision_dir.mkdir(parents=True, exist_ok=True)
    global VISION_DIR
    VISION_DIR = vision_dir
    _INITIALIZED = True

_initialize()


def ensure_resource(name, uri=None):
    path = VISION_DIR  / name
    if path.is_file():
        # It's already downloaded, nothing to do.
        return path
    if uri is None:
        uri = _get_uri(name)
    if uri is None:
        # We could not find the download URL
        return None
    r = requests.get(uri, stream=True)
    CHUNK_SIZE = 1024
    print("Downloading {}".format(name))
    with path.open('wb') as o:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            o.write(chunk)
    print("Download complete for {}".format(name))
    return path


@dataclass
class _Resource:
    name: str
    uri: str

_KNOWN_RESOURCES = [
    _Resource(name="haarcascade_frontalface_default.xml", 
        uri="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
]


def _get_uri(name):
    for res in _KNOWN_RESOURCES:
        if res.name == name:
            return res.uri
    return None