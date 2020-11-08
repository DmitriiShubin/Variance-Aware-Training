from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np
import numba
from time import time

class Metric:
    def __init__(self):

        self.confustion_matrix = None
        self.num_classes = 1
        self.smooth = 1
        self.iou = 0
        self.intersection = 0
        self.union = 0

    def calc_cm(self, labels, outputs):

        outputs = threshold(outputs)

        self.intersection += np.logical_and(labels, outputs)
        self.union += np.logical_or(labels, outputs)


    def compute(self):
        J  = np.sum(self.intersection) / np.sum(self.union)
        self.intersection = 0
        self.union = 0
        return J

    def one_hot(self,x):
        return np.eye(self.num_classes, dtype=np.float32)[x.astype(np.int8)]

@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





