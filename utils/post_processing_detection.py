import numpy as np
import torch
from torchvision.ops import nms, remove_small_boxes


class Post_Processing:
    def run(self,pred,objectness_threshold,nms_threshold):

        bboxes = pred[0]['boxes'].detach().numpy()
        scores = pred[0]['scores'].unsqueeze(dim=1).detach().numpy()
        classes = pred[0]['labels'].unsqueeze(dim=1).detach().numpy() - 1

        # thresholding
        bboxes = bboxes[scores[:, 0] > objectness_threshold, :]
        classes = classes[scores[:, 0] > objectness_threshold, :]
        scores = scores[scores[:, 0] > objectness_threshold, :]

        if bboxes.shape[0] == 0:
            return bboxes, scores, classes

        # removing small objects
        small_boxes_idx = remove_small_boxes(torch.Tensor(bboxes), min_size=1.0)
        bboxes = bboxes[small_boxes_idx, :]
        classes = classes[small_boxes_idx, :]
        scores = scores[small_boxes_idx, :]

        if bboxes.shape[0] == 0:
            return bboxes, scores, classes

        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
            scores = np.expand_dims(scores, axis=0)
            classes = np.expand_dims(classes, axis=0)

        # nms
        small_boxes_idx = nms(torch.Tensor(bboxes), torch.Tensor(scores[:, 0]), iou_threshold=nms_threshold)
        bboxes = bboxes[small_boxes_idx, :]
        classes = classes[small_boxes_idx, :]
        scores = scores[small_boxes_idx, :]

        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
            classes = np.expand_dims(classes, axis=0)
            scores = np.expand_dims(scores, axis=0)


        return bboxes, scores, classes
