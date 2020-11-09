from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np
import numba
from time import time

class Metric:
    def __init__(self):


        self.smooth = 1
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def calc_cm(self, labels, outputs,train=True):

        if train:
            labels = np.argmax(labels, axis=1)
            outputs = np.argmax(outputs, axis=1)

        labels = np.eye(2, dtype=np.float32)[labels.astype(np.int8)]
        outputs = np.eye(2, dtype=np.float32)[outputs.astype(np.int8)]

        self.tp += np.sum(labels * outputs,axis=0)
        self.fp += np.sum((1 - labels) * outputs,axis=0)
        self.fn += np.sum(labels * (1 - outputs),axis=0)


    def compute(self):
        J  = (self.tp[1]) / (self.tp[1] + 0.5*(self.fp[1] + self.fn[1] + self.smooth))
        self.tp = 0
        self.fp = 0
        self.fn = 0

        return J


@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





