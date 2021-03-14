# basic libs
import numpy as np
import torch


# pytorch
import torch
from torch.utils.data import Dataset
import albumentations as A

# custom modules
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, volums_list, aug, n_classes, dataset):

        self.n_classes = n_classes
        self.volums_list = volums_list
        self.preprocessing = Preprocessing(aug, dataset)

    def __len__(self):
        return len(self.volums_list)

    def __getitem__(self, idx):

        X, y = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X, y

    def load_data(self, id):

        X = np.load(self.volums_list[id]).astype(np.float32)
        y = np.load(self.volums_list[id][:-10] + 'labels.npy').astype(np.float32)

        y = self.one_hot_voxel(y)

        X, y = self.preprocessing.run(X=X, y=y)

        return X, y

    def one_hot_voxel(self, y):
        y = np.transpose(y.astype(np.int32), (1, 2, 0))
        y = np.eye(self.n_classes)[y[:, :, -1].astype(np.int32)]
        y = np.transpose(y.astype(np.float32), (2, 0, 1))
        return y


class Preprocessing:
    def __init__(self, aug, dataset):

        self.aug = aug
        self.augmentations = Augmentations(dataset)

    def run(self, X, y):

        X = self.standard_scaling(X)

        if self.aug:

            X, y = self.augmentations.run(X, y)

        return X, y

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

    def padding(self, X, y):
        max_shape = 256  # np.max(X.shape)

        X = np.concatenate([X, np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3] // 2))], axis=-1)
        X = np.concatenate(
            [np.zeros((X.shape[0], X.shape[1], X.shape[2], max_shape - X.shape[3])), X], axis=-1
        )

        y = np.concatenate([y, np.zeros((y.shape[0], y.shape[1], y.shape[2], y.shape[3] // 2))], axis=-1)
        y = np.concatenate(
            [np.zeros((y.shape[0], y.shape[1], y.shape[2], max_shape - y.shape[3])), y], axis=-1
        )

        return X, y

    def crop(self, X, y, cropsize=128):
        X_pos = np.random.choice(X.shape[1] - cropsize)
        Y_pos = np.random.choice(X.shape[2] - cropsize)
        Z_pos = np.random.choice(X.shape[3] - cropsize)

        X = X[:, X_pos : X_pos + cropsize, Y_pos : Y_pos + cropsize, Z_pos : Z_pos + cropsize]
        y = y[:, X_pos : X_pos + cropsize, Y_pos : Y_pos + cropsize, Z_pos : Z_pos + cropsize]
        return X, y


class Augmentations:
    def __init__(self, dataset):

        if dataset == 'brats':
            prob = 0.5
            self.augs = A.Compose(
                [  # A.Blur(blur_limit=3,p=prob),
                    A.HorizontalFlip(p=prob),
                    A.VerticalFlip(p=prob),
                    A.Rotate(limit=10, p=prob),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=prob),
                    A.RandomSizedCrop(min_max_height=(210, 210), height=240, width=240, p=prob),
                    # A.RandomGamma(gamma_limit=(80,120),p=prob)
                ]
            )

    def run(self, image, mask):

        image = np.transpose(image.astype(np.float32), (1, 2, 0))
        mask = np.transpose(mask.astype(np.float32), (1, 2, 0))

        # apply augs
        augmented = self.augs(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        mask = np.transpose(mask.astype(np.float32), (2, 0, 1))

        return image, mask
