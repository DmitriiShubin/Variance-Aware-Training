# basic libs
import numpy as np
import torch
import json

# pytorch
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision.ops import box_convert

# custom modules
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, volums_list, aug, n_classes):

        self.n_classes = n_classes
        self.volums_list = volums_list
        self.preprocessing = Preprocessing(aug)

    def __len__(self):
        return len(self.volums_list)

    def __getitem__(self, idx):

        X, y, X_s, y_s = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.float)
        X_s = torch.tensor(X_s, dtype=torch.float)
        y_s = torch.tensor(y_s, dtype=torch.float)

        return X, y, X_s, y_s

    def load_data(self, id):

        X = np.load(self.volums_list[id]).astype(np.float32)
        y = json.load(open(self.volums_list[id].replace('image.npy', 'label.json')))

        annot = self.get_annotations(y)

        annot_dummy = annot.copy()
        X, annot['boxes'], annot['labels'] = self.preprocessing.run(
            X=X, bboxes=annot['boxes'], classes=annot['labels']
        )

        # second head
        images_subset = self.volums_list.copy()
        images_subset.remove(self.volums_list[id])
        X_s = np.load(np.random.choice(np.array(images_subset))).astype(np.float32)

        y_s = [0]

        X_s, _, _ = self.preprocessing.run(X=X_s, bboxes=annot_dummy['boxes'], classes=annot_dummy['labels'])

        return X, annot, X_s, y_s

    def get_annotations(self, y):

        # some images appear to miss annotations (like image with id 257034)
        if y['0']['Target'] == 0:
            target = {
                'boxes': torch.Tensor([[0, 0, 1, 1]]),
                'labels': torch.Tensor([0.0]).type(torch.int64),
            }
            return target

        # parse annotations
        boxes = []
        labels = []
        for object in y.keys():
            object = y[object]
            # some annotations have basically no width / height, skip them
            if object['w'] < 1 or object['h'] < 1:
                continue

            boxes.append([object['x'], object['y'], object['x'] + object['w'], object['y'] + object['h']])
            labels.append([1.0])

        boxes = torch.Tensor(boxes)
        labels = torch.Tensor(labels).type(torch.int64)
        labels = labels.view(-1)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations()

    def run(self, X, bboxes, classes):

        if self.aug:
            X, bboxes, classes = self.augmentations.run(X, bboxes, classes)

        X = self.standard_scaling(X)

        return X, bboxes, classes

    def standard_scaling(self, X):
        X = X.astype(np.float32)

        for i in range(X.shape[0]):
            std = np.std(X[i, :, :])
            mean = np.mean(X[i, :, :])
            if std > 0:
                X[i, :, :] = (X[i, :, :] - mean) / std
            else:
                X[i, :, :] = X[i, :, :] - mean

        return X

    def imagenet_normalize(self, X):

        X = X / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for i in range(len(mean)):
            X[:, :, i] = (X[:, :, i] - mean[i]) / std[i]

        return X

    def minmax_scaling(self, X):
        min = np.min(X)
        max = np.max(X)
        if max > min + 1e-3:
            X = (X - min) / (max - min)
        else:
            X = X - np.mean(X)
        return X


class Augmentations:
    def __init__(self):

        prob = 0.5
        self.augs = A.Compose(
            [
                A.HorizontalFlip(p=prob),
                # A.VerticalFlip(p=prob),
                A.Rotate(limit=15, p=prob),
                # # # # A.ElasticTransform(alpha=0.05, p=prob),
                # A.RandomSizedCrop(min_max_height=(140, 220), height=256, width=256, p=prob),
                A.RandomGamma(gamma_limit=(80, 120), p=prob),
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.3),
        )

    def run(self, image, bboxes, classes):

        image = np.transpose(image.astype(np.float32), (1, 2, 0))

        bboxes = box_convert(bboxes, 'xyxy', 'xywh')

        bboxes = bboxes.tolist()
        classes = classes.tolist()

        # apply augs
        augmented = self.augs(image=image, bboxes=bboxes, category_ids=classes)
        image = augmented['image']
        bboxes = augmented['bboxes']
        classes = augmented['category_ids']
        if len(bboxes) == 0:
            bboxes = torch.Tensor(np.zeros((0, 4)))
            classes = torch.Tensor([0.0]).type(torch.int64)
        else:
            bboxes = torch.Tensor(bboxes)
            bboxes = box_convert(bboxes, 'xywh', 'xyxy')
            classes = torch.Tensor(classes).type(torch.int64)

        image = np.transpose(image.astype(np.float32), (2, 0, 1))

        return image, bboxes, classes

    def process_with_bbooxes(self, image, bboxes, classes):

        shape = image.shape[1]
        #
        bboxes = bboxes / shape
        #

        bboxes = bboxes.tolist()
        classes = classes.tolist()

        # apply augs
        augmented = self.augs(image=image, bboxes=bboxes, class_labels=classes)
        image = augmented['image']
        bboxes = augmented['bboxes']
        classes = augmented['class_labels']
        if len(bboxes) == 0:
            bboxes = torch.Tensor(np.zeros((0, 5)))
            classes = torch.Tensor([0.0]).type(torch.int64)
        else:
            # print(bboxes)
            # print(classes)
            bboxes = torch.Tensor(bboxes)
            bboxes = bboxes * shape
            classes = torch.Tensor(classes).type(torch.int64)

        return image, bboxes, classes
