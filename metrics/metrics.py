import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score


class Dice_score:
    def __init__(self, n_classes: int = 2, exclude_class: int = 0):

        self.tp = np.array([0] * (n_classes))
        self.fp = np.array([0] * (n_classes))
        self.fn = np.array([0] * (n_classes))

        self.n_classes = n_classes
        self.exclude_class = exclude_class

    def calc_running_score(self, labels, outputs):

        # TODO
        labels = np.eye(self.n_classes)[labels.astype(np.int32)]
        outputs = np.eye(self.n_classes)[outputs.astype(np.int32)]

        tp = np.sum(labels * outputs, axis=0)
        fp = np.sum(outputs, axis=0) - tp
        fn = np.sum(labels, axis=0) - tp

        self.tp = self.tp + tp
        self.fp = self.fp + fp
        self.fn = self.fn + fn

    def compute(self):

        # dice macro
        f1 = ((1 + 2 ** 2) * self.tp[1:] + 1e-3) / (
            (1 + 2 ** 2) * self.tp[1:] + 2 ** 2 * self.fn[1:] + self.fp[1:] + 1e-3
        )

        self.tp = np.array([0] * (self.n_classes))
        self.fp = np.array([0] * (self.n_classes))
        self.fn = np.array([0] * (self.n_classes))

        return np.mean(f1)

    def calc_running_score_samplewise(self, labels, outputs):

        mae = np.mean(np.abs(labels - outputs), axis=1)

        return mae.tolist()

    def reset(self):
        self.tp = np.array([0] * (self.n_classes))
        self.fp = np.array([0] * (self.n_classes))
        self.fn = np.array([0] * (self.n_classes))
        return True


class RocAuc:
    def __init__(self):

        self.labels = []
        self.outputs = []

    def calc_running_score(self, labels: np.array, outputs: np.array):

        self.labels += labels.tolist()
        self.outputs += outputs.tolist()

    def compute(self):

        score = roc_auc_score(self.labels, self.outputs)

        self.labels = []
        self.outputs = []

        return score

    def reset(self):
        self.labels = []
        self.outputs = []
        return True
