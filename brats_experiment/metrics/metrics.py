import numpy as np, os, os.path, sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score,multilabel_confusion_matrix


class Metric:

    def __init__(self):

        self.confustion_matrix=None

    def calc_cm(self,labels, outputs):

        if self.confustion_matrix is None:
            self.confustion_matrix = multilabel_confusion_matrix(labels, outputs)

        else:
            self.confustion_matrix += multilabel_confusion_matrix(labels, outputs)


    # Compute the evaluation metric for the Challenge.
    def compute(self):#, labels, outputs, smooth=1):
        f1 = 0
        for i in range(self.confustion_matrix.shape[0]):
            tn, fp, fn, tp = self.confustion_matrix[i, :, :].ravel()
            f1 += tp / (tp + 0.5 * (fp + fn)) / self.confustion_matrix.shape[0]
        #dice = f1_score(labels, outputs, average='macro')
        return f1

    def get_one_hot(self, y):

        y_binary = np.zeros((y.shape[0], 4))
        for i in range(4):
            y_binary[np.where(y == i), i] = 1

        return y_binary
