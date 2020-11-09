from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np
import numba
from time import time

class Metric:
    def __init__(self):


        self.smoothing = 1e-5
        self.intersection = 0
        self.union = 0

    def calc_cm(self, labels, outputs,train=True):

        if train:
            labels = np.argmax(labels, axis=1)
            outputs = np.argmax(outputs, axis=1)

        labels = np.eye(2, dtype=np.float32)[labels.astype(np.int8)]
        outputs = np.eye(2, dtype=np.float32)[outputs.astype(np.int8)]

        self.intersection += np.sum(labels * outputs,axis=0)
        self.union += np.sum(labels,axis=0) + np.sum(outputs,axis=0)


    def compute(self):
        J  = (self.intersection[1]+ self.smoothing) / (self.union[1] + self.smoothing)
        self.intersection = 0
        self.union = 0

        return J


@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





