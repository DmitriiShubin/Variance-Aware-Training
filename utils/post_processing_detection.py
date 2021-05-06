import numpy as np
import torch
from torchvision.ops import nms, remove_small_boxes


class Post_Processing:
    def __init__(self, objectness_threshold, nms_threshold):
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold

    def run(self, target: torch.Tensor, classification: list, bboxes: list):

        target_batch_processed = []
        pred_batch_processed = []

        for index, target_sample in enumerate(target):

            # process target

            n_objects = target_sample.shape[0]

            target_processed = {}
            if n_objects == 0:

                target_processed['0'] = {'x1': 0.0, 'y1': 0.0, 'Obj_score': -1, 'x2': 0.0, 'y2': 0.0}
                target_batch_processed.append(target_processed)

            else:
                for object in range(n_objects):
                    target_processed[object] = {
                        'x1': float(target_sample[object, 0]),  # - float(target_sample[object,2])/2,
                        'y1': float(target_sample[object, 1]),  # - float(target_sample[object,3])/2,
                        'Obj_score': float(target_sample[object, 4]),
                        'x2': float(target_sample[object, 2]),  # + float(target_sample[object,2])/2,
                        'y2': float(target_sample[object, 3]),  # - float(target_sample[object,3])/2,
                    }
                target_batch_processed.append(target_processed)

            # process predictions
            classification_sample = classification[index]
            bboxes_sample = bboxes[index]

            # #thresholding
            # scores_over_thresh = (classification_sample > self.objectness_threshold)
            # bboxes_sample = bboxes_sample[scores_over_thresh,:]
            # classification_sample = classification_sample[scores_over_thresh]

            pred_processed = {}
            if classification_sample.shape[0] == 0:
                pred_processed['0'] = {
                    'x1': 0.0,
                    'y1': 0.0,
                    'Obj_score': -1,
                    'x2': 0.0,
                    'y2': 0.0,
                }
                pred_batch_processed.append(pred_processed)
            else:
                # #NMS
                # anchors_nms_idx = nms(bboxes_sample, classification_sample, self.nms_threshold)
                # bboxes_sample = bboxes_sample[anchors_nms_idx]
                # classification_sample = classification_sample[anchors_nms_idx]
                #
                # #small boxes
                # anchors_nms_idx = remove_small_boxes(bboxes_sample, min_size=5)
                # bboxes_sample = bboxes_sample[anchors_nms_idx]
                # classification_sample = classification_sample[anchors_nms_idx]

                # if classification_sample.shape[0] == 0:
                #     pred_processed['0'] = {
                #         'x1': 0.0,
                #         'y1': 0.0,
                #         'Obj_score': -1,
                #         'x2': 0.0,
                #         'y2': 0.0,
                #     }
                #     pred_batch_processed.append(pred_processed)
                # else:

                for object in range(classification_sample.shape[0]):
                    if float(bboxes_sample[object, 2] - bboxes_sample[object, 0]) < 1:
                        continue
                    if float(bboxes_sample[object, 3] - bboxes_sample[object, 1]) < 1:
                        continue
                    pred_processed[object] = {
                        'x1': max(
                            float(bboxes_sample[object, 0]), 0
                        ),  # - float(bboxes_sample[object, 2]) / 2,
                        'y1': max(
                            float(bboxes_sample[object, 1]), 0
                        ),  # - float(bboxes_sample[object, 3]) / 2,
                        'Obj_score': float(classification_sample[object]),
                        'x2': max(
                            float(bboxes_sample[object, 2]), 0
                        ),  # + float(bboxes_sample[object, 2]) / 2,
                        'y2': max(
                            float(bboxes_sample[object, 3]), 0
                        ),  # - float(bboxes_sample[object, 3]) / 2,
                    }
                pred_batch_processed.append(pred_processed)

        return target_batch_processed, pred_batch_processed

    def __convert_bbox_coordinates(self, bbox: dict):
        """
        Converts input center coordinates to bbox coordinates
        Args:
            bbox: dict

        Returns:
            bbox_converted: dict
        """
        bbox_converted = {
            'x1': bbox['cx'] - bbox['w'] / 2,
            'x2': bbox['cx'] + bbox['w'] / 2,
            'y1': bbox['cy'] - bbox['h'] / 2,
            'y2': bbox['cy'] + bbox['h'] / 2,
        }

        return bbox_converted
