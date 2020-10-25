import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

class Metric:

    # Compute the evaluation metric for the Challenge.
    def compute(self, labels, outputs,smooth=1):
        intersection = np.sum(labels * outputs)
        union = np.sum(labels) + np.sum(outputs)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

        #
        # intersection = np.sum(labels * outputs)
        # union = np.sum(labels) + np.sum(outputs) - intersection
        # if union == 0:
        #     return 1
        # else:
        #     return intersection/union

    def get_one_hot(self, y):

        y_binary = np.zeros((y.shape[0], 4))
        for i in range(4):
            y_binary[np.where(y == i), i] = 1

        return y_binary
