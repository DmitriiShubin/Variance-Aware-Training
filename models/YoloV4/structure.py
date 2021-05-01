import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from models.YoloV4.FPN import FPN
import numpy as np
from metrics.metrics import get_iou


class PredictionHead(nn.Module):
    def __init__(self, in_ch_1, n_classes, image_size):
        super().__init__()

        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]  # w and h of anchors

        self.anchors = torch.Tensor(self.anchors)
        self.anchr_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.scale_factors = [8, 16, 32]
        self.grids = []
        self.n_classes = n_classes
        self.image_size = image_size

        # normalize anchors based on scaling factors
        for scale in self.scale_factors:
            self.grids.append(int(image_size / scale))

        # calculate cx, cy for each grid cell
        self.create_cell_coordinates_grid()

        output_ch = (4 + 1) * 3

        # 15×32×32+15×16×16+15×8×8 = 20160 predictions in total (4032 bounding boxes)

        self.conv_1 = nn.Conv2d(in_ch_1, output_ch, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv2d(in_ch_1, output_ch, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv2d(in_ch_1, output_ch, kernel_size=1, stride=1)

    def create_cell_coordinates_grid(self):
        self.channel_1_c = np.zeros((self.grids[0], self.grids[0], 2))  # X x Y x number of coordinates
        self.channel_2_c = np.zeros((self.grids[1], self.grids[1], 2))
        self.channel_3_c = np.zeros((self.grids[2], self.grids[2], 2))

        self.channel_1_c = self.fill_coordinates(self.channel_1_c)
        self.channel_2_c = self.fill_coordinates(self.channel_2_c)
        self.channel_3_c = self.fill_coordinates(self.channel_3_c)

        return True

    def fill_coordinates(self, array):

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i, j, 0] = (self.image_size / array.shape[0]) * i
                array[i, j, 1] = (self.image_size / array.shape[1]) * j

        return array

    def forward(self, neck1, neck2, neck3, train=False):

        x1 = self.conv_1(neck1)
        x2 = self.conv_2(neck2)
        x3 = self.conv_3(neck3)

        # extract objectness score
        objectness_1 = torch.sigmoid(x1[:, 0:3, :, :])  # Batch x n_anchors x H x W
        objectness_2 = torch.sigmoid(x2[:, 0:3, :, :])
        objectness_3 = torch.sigmoid(x3[:, 0:3, :, :])

        # extract bounding boxes
        bb_H_channel_1 = x1[:, 3:6, :, :]  # Batch x 2*n_anchors x H x W
        bb_H_channel_2 = x2[:, 3:6, :, :]  # Batch x 2*n_anchors x H x W
        bb_H_channel_3 = x3[:, 3:6, :, :]  # Batch x 2*n_anchors x H x W

        bb_W_channel_1 = x1[:, 6:9, :, :]  # Batch x 2*n_anchors x H x W
        bb_W_channel_2 = x2[:, 6:9, :, :]  # Batch x 2*n_anchors x H x W
        bb_W_channel_3 = x3[:, 6:9, :, :]  # Batch x 2*n_anchors x H x W

        bb_cx_channel_1 = torch.sigmoid(x1[:, 9:12, :, :])  # Batch x n_classes*n_anchors x H x W
        bb_cx_channel_2 = torch.sigmoid(x2[:, 9:12, :, :])
        bb_cx_channel_3 = torch.sigmoid(x3[:, 9:12, :, :])

        bb_cy_channel_1 = torch.sigmoid(x1[:, 12:, :, :])  # Batch x n_classes*n_anchors x H x W
        bb_cy_channel_2 = torch.sigmoid(x2[:, 12:, :, :])
        bb_cy_channel_3 = torch.sigmoid(x3[:, 12:, :, :])

        if train:

            return (
                [objectness_1, objectness_2, objectness_3],
                [bb_H_channel_1, bb_H_channel_2, bb_H_channel_3],
                [bb_W_channel_1, bb_W_channel_2, bb_W_channel_3],
                [bb_cx_channel_1, bb_cx_channel_2, bb_cx_channel_3],
                [bb_cy_channel_1, bb_cy_channel_2, bb_cy_channel_3],
            )
        else:
            # TODO: recalculate center coordinates
            return (
                [objectness_1, objectness_2, objectness_3],
                [
                    # TODO: refactor
                    torch.exp(bb_W_channel_1) * self.anchors[0:3, 0],
                    torch.exp(bb_W_channel_2) * self.anchors[3:6, 0],
                    torch.exp(bb_W_channel_3) * self.anchors[6:9, 0],
                ],
                [
                    # TODO: refactor
                    torch.exp(bb_H_channel_1) * self.anchors[0:3, 1],
                    torch.exp(bb_H_channel_2) * self.anchors[3:6, 1],
                    torch.exp(bb_H_channel_3) * self.anchors[6:9, 1],
                ],
                [
                    bb_cx_channel_1 * self.scale_factors[0] + self.self.channel_1_c[:, :, 0],
                    bb_cx_channel_2 * self.scale_factors[1] + self.self.channel_2_c[:, :, 0],
                    bb_cx_channel_3 * self.scale_factors[2] + self.self.channel_3_c[:, :, 0],
                ],
                [
                    bb_cy_channel_1 * self.scale_factors[0] + self.self.channel_1_c[:, :, 1],
                    bb_cy_channel_2 * self.scale_factors[1] + self.self.channel_2_c[:, :, 1],
                    bb_cy_channel_3 * self.scale_factors[2] + self.self.channel_3_c[:, :, 1],
                ],
            )

    def build_target(self, targets:dict):

        objectness_1 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[0]), self.grids[0], self.grids[0]))
        )
        objectness_2 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[1]), self.grids[1], self.grids[1]))
        )
        objectness_3 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[2]), self.grids[2], self.grids[2]))
        )

        bb_cx_channel_1 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[0]), self.grids[0], self.grids[0]))
        )
        bb_cx_channel_2 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[1]), self.grids[1], self.grids[1]))
        )
        bb_cx_channel_3 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[2]), self.grids[2], self.grids[2]))
        )

        bb_cy_channel_1 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[0]), self.grids[0], self.grids[0]))
        )
        bb_cy_channel_2 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[1]), self.grids[1], self.grids[1]))
        )
        bb_cy_channel_3 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[2]), self.grids[2], self.grids[2]))
        )

        bb_H_channel_1 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[0]), self.grids[0], self.grids[0]))
        )
        bb_H_channel_2 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[1]), self.grids[1], self.grids[1]))
        )
        bb_H_channel_3 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[2]), self.grids[2], self.grids[2]))
        )

        bb_W_channel_1 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[0]), self.grids[0], self.grids[0]))
        )
        bb_W_channel_2 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[1]), self.grids[1], self.grids[1]))
        )
        bb_W_channel_3 = torch.tensor(
            np.zeros((len(targets), len(self.anchr_masks[2]), self.grids[2], self.grids[2]))
        )

        for sample in range(len(targets)):  # iterate over the batch

            target = targets[sample]

            if target['0']['Target'] == 0:
                continue

            for key in target.keys():
                object = target[key]

                # mark grid cells when the object is present
                objectness_1[
                    sample,
                    :,
                    int(object['cx'] / self.scale_factors[0]),
                    int(object['cy'] / self.scale_factors[0]),
                ] = 1
                objectness_2[
                    sample,
                    :,
                    int(object['cx'] / self.scale_factors[1]),
                    int(object['cy'] / self.scale_factors[1]),
                ] = 1
                objectness_3[
                    sample,
                    :,
                    int(object['cx'] / self.scale_factors[2]),
                    int(object['cy'] / self.scale_factors[2]),
                ] = 1

                # select the best fitting anchor
                iou1 = []
                iou2 = []
                iou3 = []

                # channel 1
                for anchor in self.anchr_masks[0]:

                    object_temp = object.copy()
                    object_temp['cx'] = object['cx'] / self.scale_factors[0] - int(
                        object['cx'] / self.scale_factors[0]
                    )
                    object_temp['cy'] = object['cy'] / self.scale_factors[0] - int(
                        object['cy'] / self.scale_factors[0]
                    )
                    target_converted = self.convert_bbox_coordinates(object_temp)

                    anchor = self.anchors[anchor]
                    anchor = {'cx': 0.5, 'cy': 0.5, 'w': anchor.numpy()[0], 'h': anchor.numpy()[1]}
                    anchor = self.convert_bbox_coordinates(anchor)
                    iou1.append(self.get_iou(target_converted, anchor))

                # channel 2
                for anchor in self.anchr_masks[1]:
                    object_temp = object.copy()
                    object_temp['cx'] = object['cx'] / self.scale_factors[1] - int(
                        object['cx'] / self.scale_factors[1]
                    )
                    object_temp['cy'] = object['cy'] / self.scale_factors[1] - int(
                        object['cy'] / self.scale_factors[1]
                    )
                    target_converted = self.convert_bbox_coordinates(object_temp)

                    anchor = self.anchors[anchor]
                    anchor = {'cx': 0.5, 'cy': 0.5, 'w': anchor.numpy()[0], 'h': anchor.numpy()[1]}
                    anchor = self.convert_bbox_coordinates(anchor)
                    iou2.append(self.get_iou(target_converted, anchor))

                # channel 3
                for anchor in self.anchr_masks[2]:
                    object_temp = object.copy()
                    object_temp['cx'] = object['cx'] / self.scale_factors[2] - int(
                        object['cx'] / self.scale_factors[2]
                    )
                    object_temp['cy'] = object['cy'] / self.scale_factors[2] - int(
                        object['cy'] / self.scale_factors[2]
                    )
                    target_converted = self.convert_bbox_coordinates(object_temp)

                    anchor = self.anchors[anchor]
                    anchor = {'cx': 0.5, 'cy': 0.5, 'w': anchor.numpy()[0], 'h': anchor.numpy()[1]}
                    anchor = self.convert_bbox_coordinates(anchor)
                    iou3.append(self.get_iou(target_converted, anchor))

                # TODO: repeat for all channels

                iou1 = np.array(iou1)
                iou3 = np.array(iou2)
                iou3 = np.array(iou3)

                # calculate centers
                bb_cx_channel_1[
                    sample,
                    np.where(iou1 == np.max(iou1)),
                    int(object['cx'] / self.scale_factors[0]),
                    int(object['cy'] / self.scale_factors[0]),
                ] = object['cx'] / self.scale_factors[0] - int(object['cx'] / self.scale_factors[0])

                bb_cx_channel_2[
                    sample,
                    np.where(iou2 == np.max(iou2)),
                    int(object['cx'] / self.scale_factors[1]),
                    int(object['cy'] / self.scale_factors[1]),
                ] = object['cx'] / self.scale_factors[1] - int(object['cx'] / self.scale_factors[1])

                bb_cx_channel_3[
                    sample,
                    np.where(iou3 == np.max(iou3)),
                    int(object['cx'] / self.scale_factors[2]),
                    int(object['cy'] / self.scale_factors[2]),
                ] = object['cx'] / self.scale_factors[2] - int(object['cx'] / self.scale_factors[2])

                bb_cy_channel_1[
                    sample,
                    np.where(iou1 == np.max(iou1)),
                    int(object['cx'] / self.scale_factors[0]),
                    int(object['cy'] / self.scale_factors[0]),
                ] = object['cy'] / self.scale_factors[0] - int(object['cy'] / self.scale_factors[0])

                bb_cy_channel_2[
                    sample,
                    np.where(iou2 == np.max(iou2)),
                    int(object['cx'] / self.scale_factors[1]),
                    int(object['cy'] / self.scale_factors[1]),
                ] = object['cy'] / self.scale_factors[1] - int(object['cy'] / self.scale_factors[1])

                bb_cy_channel_3[
                    sample,
                    np.where(iou3 == np.max(iou3)),
                    int(object['cx'] / self.scale_factors[2]),
                    int(object['cy'] / self.scale_factors[2]),
                ] = object['cy'] / self.scale_factors[2] - int(object['cy'] / self.scale_factors[2])

                # calculate height and width
                bb_H_channel_1[
                    sample,
                    np.where(iou1 == np.max(iou1)),
                    int(object['cx'] / self.scale_factors[0]),
                    int(object['cy'] / self.scale_factors[0]),
                ] = np.log(
                    object['h']
                    / self.anchors[self.anchr_masks[0]].numpy()[np.where(iou1 == np.max(iou1))][0, 0]
                )

                bb_W_channel_1[
                    sample,
                    np.where(iou1 == np.max(iou1)),
                    int(object['cx'] / self.scale_factors[0]),
                    int(object['cy'] / self.scale_factors[0]),
                ] = np.log(
                    object['w']
                    / self.anchors[self.anchr_masks[0]].numpy()[np.where(iou1 == np.max(iou1))][0, 1]
                )

                bb_H_channel_2[
                    sample,
                    np.where(iou2 == np.max(iou2)),
                    int(object['cx'] / self.scale_factors[1]),
                    int(object['cy'] / self.scale_factors[1]),
                ] = np.log(
                    object['h']
                    / self.anchors[self.anchr_masks[1]].numpy()[np.where(iou2 == np.max(iou2))][0, 0]
                )

                bb_W_channel_2[
                    sample,
                    np.where(iou2 == np.max(iou2)),
                    int(object['cx'] / self.scale_factors[1]),
                    int(object['cy'] / self.scale_factors[1]),
                ] = np.log(
                    object['w']
                    / self.anchors[self.anchr_masks[1]].numpy()[np.where(iou2 == np.max(iou2))][0, 1]
                )

                bb_H_channel_3[
                    sample,
                    np.where(iou3 == np.max(iou3)),
                    int(object['cx'] / self.scale_factors[2]),
                    int(object['cy'] / self.scale_factors[2]),
                ] = np.log(
                    object['h']
                    / self.anchors[self.anchr_masks[2]].numpy()[np.where(iou3 == np.max(iou3))][0, 0]
                )

                bb_W_channel_3[
                    sample,
                    np.where(iou3 == np.max(iou3)),
                    int(object['cx'] / self.scale_factors[2]),
                    int(object['cy'] / self.scale_factors[2]),
                ] = np.log(
                    object['w']
                    / self.anchors[self.anchr_masks[2]].numpy()[np.where(iou3 == np.max(iou3))][0, 1]
                )

        return (
            objectness_1,
            objectness_2,
            objectness_3,
            bb_H_channel_1,
            bb_H_channel_2,
            bb_H_channel_3,
            bb_W_channel_1,
            bb_W_channel_2,
            bb_W_channel_3,
            bb_cx_channel_1,
            bb_cx_channel_2,
            bb_cx_channel_3,
            bb_cy_channel_1,
            bb_cy_channel_2,
            bb_cy_channel_3,
        )

    def convert_bbox_coordinates(self, bbox: dict):
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

    def get_iou(self, bb1, bb2):
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
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

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


class YoloV4(nn.Module):
    """
    https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/4ccef0ec8fe984e059378813e33b3740929e0c19
    """

    def __init__(self, hparams, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        # backbone
        self.backbone = EfficientNet.from_pretrained(hparams['pre_trained_model'])

        self.backbone.eval()
        dummy = torch.rand(1, 3, 512, 512)
        endpoints = self.backbone.extract_endpoints(dummy)
        self.backbone.train()

        # neck
        self.neek = FPN(
            endpoints['reduction_3'].shape[1],
            endpoints['reduction_4'].shape[1],
            endpoints['reduction_5'].shape[1],
        )

        self.head = PredictionHead(
            endpoints['reduction_3'].shape[1], n_classes=hparams['n_classes'], image_size=512
        )  # output_ch, n_classes, inference)

    def forward(self, input, train=False):

        endpoints = self.backbone.extract_endpoints(input)

        neck1, neck2, neck3 = self.neek(
            endpoints['reduction_3'], endpoints['reduction_4'], endpoints['reduction_5'],
        )

        output = self.head(neck1, neck2, neck3, train=train)
        return output
