from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
import numpy as np


class Metric:
    def __init__(self):

        self.confustion_matrix = None
        self.num_classes = 1
        self.smooth = 1
        self.iou = 0

    def calc_cm(self, labels, outputs):

        if self.confustion_matrix is None:
            self.confustion_matrix = confusion_matrix(labels, outputs, labels=[0, 1]).astype(
                np.float32
            )

        else:
            self.confustion_matrix += confusion_matrix(labels, outputs, labels=[0, 1]).astype(
                np.float32
            )

        #IoU:
        # if self.num_classes>1:
        #     outputs = self.one_hot(outputs)
        #     labels = self.one_hot(labels)
        # intersection = np.sum(np.abs(labels * outputs), axis=0)
        # union = np.sum(labels,axis=0) + np.sum(outputs,axis=0) - intersection
        # if self.num_classes>1:
        #     self.iou += np.mean((intersection) / (union + self.smooth), axis=0)
        # else:
        #     self.iou += (intersection) / (union + self.smooth)

    def compute(self):
        # f1 = 0
        # for i in range(self.confustion_matrix.shape[0]):
        #     tn, fp, fn, tp = self.confustion_matrix[i, :, :].ravel()
        #     f1 += tp / (tp + 0.5 * (fp + fn)) / self.confustion_matrix.shape[0]
        #
        # self.confustion_matrix = None
        J = 0
        tn, fp, fn, tp = self.confustion_matrix.ravel()
        J += tp / (tp + fp + fn)

        self.confustion_matrix = None
        return J

    def one_hot(self,x):
        return np.eye(self.num_classes, dtype=np.float32)[x.astype(np.int8)]



