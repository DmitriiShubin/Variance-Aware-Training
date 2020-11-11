from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np
import numba
from time import time

class Metric:
    def __init__(self):


        self.smoothing = 1e-5
        self.intersection = np.array([0,0])
        self.union = np.array([0,0])

        self.threshold = 0.5

        self.tp = np.array([0,0])
        self.fp = np.array([0,0])
        self.fn = np.array([0,0])

    def calc_cm(self, labels, outputs,train=True):

        if train:
            labels = np.argmax(labels, axis=1)
            outputs = np.argmax(outputs, axis=1)

        labels = np.eye(2, dtype=np.float32)[labels.astype(np.int8)]
        outputs = np.eye(2, dtype=np.float32)[outputs.astype(np.int8)]

        tp = np.sum(labels * outputs, axis=0)
        fp = np.sum(outputs, axis=0) - tp
        fn = np.sum(labels, axis=0) - tp

        self.tp = self.tp+ tp
        self.fp = self.fp + fp
        self.fn = self.fn +fn


        #
        self.intersection =  self.intersection + np.sum(labels * outputs,axis=0)
        self.union = self.union + np.sum(labels,axis=0) + np.sum(outputs,axis=0) - np.sum(labels * outputs,axis=0)

    def compute(self):
        J  = (self.intersection[1]+ self.smoothing) / (self.union[1] + self.smoothing)
        self.intersection = np.array([0,0])
        self.union = np.array([0,0])

        f1 = ((1 + 2 ** 2) * self.tp[1] + self.smoothing) \
                / ((1 + 2 ** 2) * self.tp[1] + 2 ** 2 * self.fn[1] + self.fp[1] + self.smoothing)

        self.tp = np.array([0, 0])
        self.fp = np.array([0, 0])
        self.fn = np.array([0, 0])

        return f1,J


@numba.jit(nopython=False, parallel=True,forceobj=True)
def threshold(x):
    x = np.round(x)
    return x





