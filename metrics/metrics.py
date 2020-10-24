import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import jaccard_score


class Metric:

    # Compute the evaluation metric for the Challenge.
    def compute(self, labels, outputs):

        labels = self.get_one_hot(labels)
        outputs = self.get_one_hot(outputs)

        return jaccard_score(labels, outputs, average='micro')

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
