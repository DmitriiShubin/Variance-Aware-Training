from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np
import numba
from time import time

class Metric:
    def __init__(self):


        self.smoothing = 1e-5
        self.J = 0

    def calc_cm(self, labels, outputs,train=True):

        if train:
            labels = np.argmax(labels, axis=1)
            outputs = np.argmax(outputs, axis=1)

        labels = np.eye(2, dtype=np.float32)[labels.astype(np.int8)]
        outputs = np.eye(2, dtype=np.float32)[outputs.astype(np.int8)]

        intersection = np.sum(labels * outputs,axis=0)
        union = np.sum(labels,axis=0) + np.sum(outputs,axis=0)
        self.J += (intersection[1]+ self.smoothing) / (union[1] + self.smoothing)

    def compute(self):
        # J  = (self.intersection[1]+ self.smoothing) / (self.union[1] + self.smoothing)
        # self.intersection = 0
        # self.union = 0
        J = self.J
        self.J = 0
        return J


@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





