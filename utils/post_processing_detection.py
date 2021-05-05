import numpy as np
import torch
from torchvision.ops import nms,remove_small_boxes

class Post_Processing:
    def __init__(self, objectness_threshold,nms_threshold):
        self.objectness_threshold = objectness_threshold
        self.nms_threshold = nms_threshold

    def run(self, target: torch.Tensor, classification: torch.Tensor, bboxes: torch.Tensor):


        classification = classification[:,:,0] #select only targets
        print(torch.max(classification))

        target_batch_processed = []
        pred_batch_processed = []

        for index, target_sample in enumerate(target):

            #process target

            n_objects = target_sample.shape[0]

            target_processed = {}
            if n_objects == 0:

                target_processed['0'] = {
                    'x1':0.0,
                    'y1':0.0,
                    'Obj_score':-1,
                    'x2':0.0,
                    'y2':0.0
                }
                target_batch_processed.append(target_processed)

            else:
                for object in range(n_objects):
                    target_processed[object] = {
                        'x1': float(target_sample[object,0]), # - float(target_sample[object,2])/2,
                        'y1': float(target_sample[object,1]), # - float(target_sample[object,3])/2,
                        'Obj_score': float(target_sample[object,4]),
                        'x2': float(target_sample[object,2]), # + float(target_sample[object,2])/2,
                        'y2': float(target_sample[object,3]), # - float(target_sample[object,3])/2,
                    }
                target_batch_processed.append(target_processed)


            #process predictions
            classification_sample = classification[index]
            bboxes_sample = bboxes[index]

            #thresholding
            scores_over_thresh = (classification_sample > self.objectness_threshold)
            bboxes_sample = bboxes_sample[scores_over_thresh,:]
            classification_sample = classification_sample[scores_over_thresh]

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
                #NMS
                anchors_nms_idx = nms(bboxes_sample, classification_sample, self.nms_threshold)
                bboxes_sample = bboxes_sample[anchors_nms_idx]
                classification_sample = classification_sample[anchors_nms_idx]

                #small boxes
                anchors_nms_idx = remove_small_boxes(bboxes_sample, min_size=5)
                bboxes_sample = bboxes_sample[anchors_nms_idx]
                classification_sample = classification_sample[anchors_nms_idx]


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

                    for object in range(classification_sample.shape[0]):
                        pred_processed[object] = {
                            'x1': max(float(bboxes_sample[object, 0]),0), #- float(bboxes_sample[object, 2]) / 2,
                            'y1': max(float(bboxes_sample[object, 1]),0), # - float(bboxes_sample[object, 3]) / 2,
                            'Obj_score': float(classification_sample[object]),
                            'x2': max(float(bboxes_sample[object, 2]),0), # + float(bboxes_sample[object, 2]) / 2,
                            'y2': max(float(bboxes_sample[object, 3]),0), # - float(bboxes_sample[object, 3]) / 2,
                        }
                    pred_batch_processed.append(pred_processed)





        # scores = torch.max(classification, dim=2, keepdim=True)[0]
        #
        # scores_over_thresh = (scores > self.hparams['objectness_threshold'])[0, :, 0]
        #
        # if scores_over_thresh.sum() == 0:
        #     # no boxes to NMS, just return
        #     return [torch.zeros(0).to(device), torch.zeros(0).to(device), torch.zeros(0, 4).to(device)]
        #
        # classification = classification[:, scores_over_thresh, :]
        # transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        # scores = scores[:, scores_over_thresh, :]
        #
        #
        #
        # nms_scores, nms_class = classification[0, anchors_nms_idx, :]  # .max(dim=1)




        # anchors_nms_idx = nms(transformed_anchors[0], scores[0], self.hparams['nms_threshold'])
        #
        # nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        #
        # a = [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


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
