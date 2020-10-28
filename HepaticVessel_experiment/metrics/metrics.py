import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class Metric:

    # Compute the evaluation metric for the Challenge.
    def compute(self, labels, outputs):
        return roc_auc_score(labels, outputs,average='macro')
