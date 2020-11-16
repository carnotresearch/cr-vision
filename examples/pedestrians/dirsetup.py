from pathlib import Path

FILE_PATH = Path(__file__).resolve()

MODULE_DIR = FILE_PATH.parent
DATA_DIR = MODULE_DIR.parent.parent / 'data'
IMAGES_DIR = DATA_DIR / 'images' / 'pedestrians'


