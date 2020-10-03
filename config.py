import ast
import os

# select the type of the model here
from models.unet import Model, hparams

# names:
DATA_PATH = './data/brain_images/'

SPLIT_TABLE_PATH = './data/split_table/'
SPLIT_TABLE_NAME = 'split_table.json'

PIC_FOLDER = './data/pictures/'
DEBUG_FOLDER = './data/CV_debug/'

for f in [PIC_FOLDER, DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)
