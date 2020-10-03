import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import f1_score

class Metric:

    # Compute the evaluation metric for the Challenge.
    def compute(self, labels, outputs):
        intersection = np.sum(labels * outputs)
        union = np.sum(labels) + np.sum(outputs) - intersection
        if union == 0:
            return 1
        else:
            return intersection/union
        #return f1_score(labels, outputs)




