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

        # self.intersection += np.sum(labels*outputs)
        # self.union += np.sum(labels+outputs) - np.sum(labels*outputs)

        self.tp += (labels * outputs)
        self.fp += ((1 - labels) * outputs)
        self.fn += (labels * (1 - outputs))


    def compute(self):
        J  = np.mean((self.tp) / (self.tp + self.fp + self.fn + self.smooth))
        self.tp = 0
        self.fp = 0
        self.fn = 0

        return J

    def one_hot(self,x):
        return np.eye(self.num_classes, dtype=np.float32)[x.astype(np.int8)]

@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





