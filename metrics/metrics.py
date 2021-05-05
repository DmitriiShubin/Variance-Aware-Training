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


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    assert bb1['x1'] <= bb1['x2'], f"bb1: {bb1['x1']}| bb1: {bb1['x2']}"
    assert bb1['y1'] <= bb1['y2'], f"bb1: {bb1['y1']}| bb1: {bb1['y2']}"
    assert bb2['x1'] <= bb2['x2'], f"bb1: {bb2['x1']}| bb1: {bb2['x2']}"
    assert bb2['y1'] <= bb2['y2'], f"bb1: {bb2['y1']}| bb1: {bb2['y2']}"

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class AP:
    def __init__(self, uou_thresholds=[0.5, 0.6, 0.7, 0.8]):

        self.uou_thresholds = uou_thresholds

        self.AP_scores = []

    def calc_running_score(self, labels: list, outputs: list):

        for index, label_sample in enumerate(labels):

            for gt_object in label_sample.keys():
                gt_object = label_sample[gt_object]

                precision = 0

                for threshold in self.uou_thresholds:

                    tp = 0
                    fp = 0
                    fn = 0
                    # if any bb was predicted for image where where is not bb, it counts as FP
                    if gt_object['Obj_score'] == -1:
                        for pred_object in outputs[index].keys():
                            fp += 1
                        continue

                    # FP - when bb is predicted, but overlaps with target with IOU < threshold
                    # TP - gt_bb and predicted bb overlaps with IOU >= threshold
                    for pred_object in outputs[index].keys():
                        pred_object = outputs[index][pred_object]
                        if pred_object['Obj_score'] == -1:
                            fn+=1
                            continue
                        iou = get_iou(pred_object, gt_object)
                        if iou < threshold:
                            fp += 1
                        else:
                            tp += 1

                    if tp == 0 and fp == 0:
                        continue
                    precision += tp / (tp + fp + fn)

                precision /= len(self.uou_thresholds)

                self.AP_scores.append(precision)

        return True

    def compute(self):

        AP = np.mean(self.AP_scores)

        self.reset()

        return AP

    def reset(self):
        self.AP_scores = []
        return True
