import ast
import os

# select the type of the model here
from models.adv_FPN import Model, hparams

# names:
DATA_PATH = '../data/HepaticVessel/processed_data/'

SPLIT_TABLE_PATH = './split_table/'
SPLIT_TABLE_NAME = 'split_table.json'

DEBUG_FOLDER = '../data/CV_debug/'

for f in [DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)
