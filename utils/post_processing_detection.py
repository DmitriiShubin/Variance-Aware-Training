import numpy as np
import torch


class Post_Processing:
    def __init__(self, threshold=0.4):
        self.threshold = threshold

    def run(self, target: list, scores: torch.Tensor,):

        #TODO: add thresholding

        anchors_nms_idx = nms(transformed_anchors[0], scores[0], self.hparams['nms_threshold'])

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        a = [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

        # upback predictions
        pred_obj1_batch, pred_obj2_batch, pred_obj3_batch = pred_obj
        pred_h1_batch, pred_h2_batch, pred_h3_batch = pred_h
        pred_w1_batch, pred_w2_batch, pred_w3_batch = pred_w
        pred_cx1_batch, pred_cx2_batch, pred_cx3_batch = pred_cx
        pred_cy1_batch, pred_cy2_batch, pred_cy3_batch = pred_cy

        predictions_batch = []

        for index, sample in enumerate(target):

            # process target
            for object in sample.keys():
                temp = sample[object]
                temp = self.__convert_bbox_coordinates(temp)
                temp['Target'] = sample[object]['Target']
                sample[object] = temp
            target[index] = sample

            pred_obj1, pred_obj2, pred_obj3 = (
                pred_obj1_batch[index],
                pred_obj2_batch[index],
                pred_obj3_batch[index],
            )
            pred_h1, pred_h2, pred_h3 = pred_h1_batch[index], pred_h2_batch[index], pred_h3_batch[index]
            pred_w1, pred_w2, pred_w3 = pred_w1_batch[index], pred_w2_batch[index], pred_w3_batch[index]
            pred_cx1, pred_cx2, pred_cx3 = (
                pred_cx1_batch[index],
                pred_cx2_batch[index],
                pred_cx3_batch[index],
            )
            pred_cy1, pred_cy2, pred_cy3 = (
                pred_cy1_batch[index],
                pred_cy2_batch[index],
                pred_cy3_batch[index],
            )

            # process predictions: select only anchors where abjects are present
            pred_w1 = pred_w1[torch.where(pred_obj1 >= self.threshold)]
            pred_cx1 = pred_cx1[torch.where(pred_obj1 >= self.threshold)]
            pred_cy1 = pred_cy1[torch.where(pred_obj1 >= self.threshold)]
            pred_h1 = pred_h1[torch.where(pred_obj1 >= self.threshold)]
            pred_obj1 = pred_obj1[torch.where(pred_obj1 >= self.threshold)]

            pred_w2 = pred_w2[torch.where(pred_obj2 >= self.threshold)]
            pred_cx2 = pred_cx2[torch.where(pred_obj2 >= self.threshold)]
            pred_cy2 = pred_cy2[torch.where(pred_obj2 >= self.threshold)]
            pred_h2 = pred_h2[torch.where(pred_obj2 >= self.threshold)]
            pred_obj2 = pred_obj2[torch.where(pred_obj2 >= self.threshold)]

            pred_w3 = pred_w3[torch.where(pred_obj3 >= self.threshold)]
            pred_cx3 = pred_cx3[torch.where(pred_obj3 >= self.threshold)]
            pred_cy3 = pred_cy3[torch.where(pred_obj3 >= self.threshold)]
            pred_h3 = pred_h3[torch.where(pred_obj3 >= self.threshold)]
            pred_obj3 = pred_obj3[torch.where(pred_obj3 >= self.threshold)]

            predictions = {}

            # channel 1
            obj_index = 0
            for i in range(pred_obj1.shape[0]):
                temp = {
                    'w': float(pred_w1[i].numpy()),
                    'h': float(pred_h1[i].numpy()),
                    'cx': float(pred_cx1[i].numpy()),
                    'cy': float(pred_cy1[i].numpy()),
                }
                temp = self.__convert_bbox_coordinates(temp)
                temp['obj_score'] = float(pred_obj1[i].numpy())
                predictions[obj_index] = temp
                obj_index += 1

            # channel 2
            for i in range(pred_obj2.shape[0]):
                temp = {
                    'w': float(pred_w2[i].numpy()),
                    'h': float(pred_h2[i].numpy()),
                    'cx': float(pred_cx2[i].numpy()),
                    'cy': float(pred_cy2[i].numpy()),
                }
                temp = self.__convert_bbox_coordinates(temp)
                temp['obj_score'] = float(pred_obj2[i].numpy())
                predictions[obj_index] = temp
                obj_index += 1

            # channel 3
            for i in range(pred_obj3.shape[0]):
                temp = {
                    'w': float(pred_w3[i].numpy()),
                    'h': float(pred_h3[i].numpy()),
                    'cx': float(pred_cx3[i].numpy()),
                    'cy': float(pred_cy3[i].numpy()),
                }
                temp = self.__convert_bbox_coordinates(temp)
                temp['obj_score'] = float(pred_obj3[i].numpy())
                predictions[obj_index] = temp
                obj_index += 1

            predictions_batch.append(predictions)

        return target, predictions_batch

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
