import ast
import os

# select the type of the model here


# names:
DATA_PATH = '../data/siim-isic-melanoma-classification/train'

SPLIT_TABLE_PATH = './split_table/'
SPLIT_TABLE_NAME = 'split_table.json'

DEBUG_FOLDER = '../data/CV_debug/'

for f in [DEBUG_FOLDER]:
    os.makedirs(f, exist_ok=True)