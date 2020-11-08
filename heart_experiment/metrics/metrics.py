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

    def calc_cm(self, labels, outputs):

        # outputs = threshold(outputs)
        #
        # self.intersection += np.sum(labels*outputs)
        # self.union += np.sum(labels+outputs) - np.sum(labels*outputs)

        self.tp += (labels * outputs).sum().astype(np.float32)
        self.fp += ((1 - labels) * outputs).sum().astype(np.float32)
        self.fn += (labels * (1 - outputs)).sum().astype(np.float32)


    def compute(self):
        J  = (self.tp) / (self.tp + self.fp + self.fn)
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





