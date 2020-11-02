from sklearn.metrics import confusion_matrix
import numpy as np


class Metric:
    def __init__(self):

        self.confustion_matrix = None
        self.eps = 1e-5

    def calc_cm(self, labels, outputs):

        if self.confustion_matrix is None:
            self.confustion_matrix = confusion_matrix(labels, outputs, labels=[0, 1]).astype(np.float32)

        else:
            self.confustion_matrix += confusion_matrix(labels, outputs, labels=[0, 1]).astype(np.float32)

    def compute(self):
        f1 = 0

        tn, fp, fn, tp = self.confustion_matrix.ravel()
        f1 += tp / (tp + 0.5 * (fp + fn) + self.eps)

        self.confustion_matrix = None
        return f1
