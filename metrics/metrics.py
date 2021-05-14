import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score
from skll.metrics import kappa
from sklearn.metrics import mean_squared_error as mse
from mean_average_precision import MetricBuilder


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

        outputs = np.round(outputs, 1)

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


class F1:
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

        return True

    def compute(self):

        # dice macro
        f1 = self.tp / (self.tp + 0.5 * (self.fp + self.fn))

        self.reset()

        return np.mean(f1)

    def reset(self):
        self.tp = np.array([0] * (self.n_classes))
        self.fp = np.array([0] * (self.n_classes))
        self.fn = np.array([0] * (self.n_classes))
        return True


class AP:
    def __init__(self, n_classes):

        self.iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

        self.AP = 0.0
        self.n_pictures = 0.0
        self.map_calc = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=n_classes - 1
        )

    def calc_running_score(self, y_batch, bboxes, scores, classes):

        # TOOO: shange index
        if y_batch[0]['labels'][0] == 0:
            gt = np.zeros((0, 6))

            if bboxes.shape[0] != 0:
                self.n_pictures += 1

        else:
            self.n_pictures += 1
            gt = np.concatenate(
                [
                    y_batch[0]['boxes'].astype(np.int32),
                    np.zeros((y_batch[0]['boxes'].shape[0], 1)),
                    np.zeros((y_batch[0]['boxes'].shape[0], 2)),
                ],
                axis=1,
            )

        preds = np.concatenate([bboxes.astype(np.int32), classes.astype(np.int32), scores], axis=1)

        # gt:   [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        # pred: [xmin, ymin, xmax, ymax, class_id, confidence]
        self.map_calc.add(preds, gt)
        self.AP += self.map_calc.value(
            iou_thresholds=self.iou_thresholds, recall_thresholds=np.arange(0.0, 1.01, 0.01), mpolicy='soft'
        )['mAP']
        self.map_calc.reset()

        return True

    def compute(self):

        mAP = self.AP / self.n_pictures

        self.reset_matric()

        return mAP

    def reset_matric(self):
        self.AP = 0.0
        self.n_pictures = 0.0
        return True


class Kappa:
    def __init__(self):

        self.outputs = []
        self.labels = []
        self.N = 5

    def calc_running_score(self,labels: np.array , outputs: np.array ):

        self.labels += labels.tolist()
        self.outputs += outputs.tolist()

    def compute(self):

        score = kappa(self.labels,self.outputs,weights='quadratic')

        self.reset()
        return score


    def reset(self):
        self.outputs = []
        self.labels = []
        return True