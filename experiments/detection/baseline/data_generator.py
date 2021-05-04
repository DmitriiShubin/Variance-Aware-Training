# basic libs
import numpy as np
import torch
import json

# pytorch
import torch
from torch.utils.data import Dataset
import albumentations as A

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

        sample = self.load_data(idx)

        #X = torch.tensor(X, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.float)

        return sample

    def load_data(self, id):

        X = np.load(self.volums_list[id]).astype(np.float32)
        y = json.load(open(self.volums_list[id].replace('image.npy', 'label.json')))

        X = self.preprocessing.run(X=X)

        annot = self.get_annotations(y)

        sample = {'img': torch.from_numpy(X), 'annot': torch.from_numpy(annot),'scale':1}

        return sample

    def get_annotations(self,y):

        # get ground truth annotations
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if y['0']['Target'] == 0:
            return annotations

        # parse annotations
        for object in y.keys():
            object = y[object]
            # some annotations have basically no width / height, skip them
            if object['w'] < 1 or object['h'] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, 0] = object['cx']
            annotation[0, 1] = object['cy']
            annotation[0, 2] = object['w']
            annotation[0, 3] = object['h']
            annotation[0, 4] = object['Target']
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations()

    def run(self, X):

        if self.aug:
            X = self.augmentations.run(X)

        X = self.standard_scaling(X)

        return X

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
                # A.HorizontalFlip(p=prob),
                # A.VerticalFlip(p=prob),
                # A.Rotate(limit=5, p=prob),
                # # A.ElasticTransform(alpha=0.05, p=prob),
                # A.RandomSizedCrop(min_max_height=(76, 76), height=96, width=96, p=prob),
                # A.RandomGamma(gamma_limit=(80, 120), p=prob),
            ]
        )

    def run(self, image):

        image = np.transpose(image.astype(np.float32), (1, 2, 0))
        # apply augs
        augmented = self.augs(image=image)
        image = augmented['image']
        image = np.transpose(image.astype(np.float32), (2, 0, 1))

        return image
