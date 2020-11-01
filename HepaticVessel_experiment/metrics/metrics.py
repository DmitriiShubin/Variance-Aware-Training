from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

class Metric:
    def __init__(self):

        self.confustion_matrix = None

    def calc_cm(self, labels, outputs):

        if self.confustion_matrix is None:
            self.confustion_matrix = multilabel_confusion_matrix(labels, outputs, labels=[0, 1, 2]).astype(np.float32)

        else:
            self.confustion_matrix += multilabel_confusion_matrix(labels, outputs, labels=[0, 1, 2]).astype(np.float32)

    def compute(self):
        f1 = 0
        for i in range(self.confustion_matrix.shape[0]):
            tn, fp, fn, tp = self.confustion_matrix[i, :, :].ravel()
            f1 += tp / (tp + 0.5 * (fp + fn)) / self.confustion_matrix.shape[0]

        self.confustion_matrix = None
        return f1
