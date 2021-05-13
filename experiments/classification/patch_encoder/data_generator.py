# basic libs
import numpy as np
import torch
import cv2
import imutils

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

        X1, X2, y = self.load_data(idx)

        X1 = torch.tensor(X1, dtype=torch.float)
        X2 = torch.tensor(X2, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        return X1, X2, y

    def load_data(self, id):

        X = np.load(self.volums_list[id]).astype(np.float32)
        X = self.preprocessing.run(X)

        y = np.random.choice(np.arange(8))
        y2 = np.random.choice(np.arange(8))

        X1 = self.select_patch(X, y)
        X2 = self.select_patch(X, y2)

        if y2 == 8 or y == 8:
            y = [1]
        elif abs(y2 - y) == 1:
            y = [1]
        elif (y2 == 7 and y == 0) or (y == 7 and y2 == 0):
            y = [1]
        else:
            y = [0]

        return X1, X2, y

    def select_patch(self, X, sector):
        # optional: upsample if too much

        X = np.transpose(X.astype(np.float32), (1, 2, 0))

        # X = cv2.resize(X, (X.shape[0] * 3, X.shape[1] * 3))

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=2)

        split_x = X.shape[0] // 3
        split_y = X.shape[1] // 3

        overlap = split_x // 4

        if sector == 0:
            X2 = X[0 : split_x + overlap * 2, 0 : split_y + overlap * 2, :]
        elif sector == 1:
            X2 = X[split_x - overlap : split_x * 2 + overlap, 0 : split_y + overlap * 2, :]
        elif sector == 2:
            X2 = X[split_x * 2 - overlap * 2 : split_x * 3, 0 : split_y + overlap * 2, :]
        elif sector == 3:
            X2 = X[split_x * 2 - overlap * 2 : split_x * 3, split_y - overlap : split_y * 2 + overlap, :]
        elif sector == 4:
            X2 = X[split_x * 2 - overlap * 2 : split_x * 3, split_y * 2 - overlap * 2 : split_y * 3, :]
        elif sector == 5:
            X2 = X[split_x - overlap : split_x * 2 + overlap, split_y * 2 - overlap * 2 : split_y * 3, :]
        elif sector == 6:
            X2 = X[0 : split_x + overlap * 2, split_y * 2 - overlap * 2 : split_y * 3, :]
        elif sector == 7:
            X2 = X[0 : split_x + overlap * 2, split_y - overlap : split_y * 2 + overlap, :]
        elif sector == 8:
            X2 = X[split_x - overlap : split_x * 2 + overlap, split_y - overlap : split_y * 2 + overlap, :]

        X2 = np.transpose(X2.astype(np.float32), (2, 0, 1))

        return X2


class Preprocessing:
    def __init__(self, aug, dataset):

        self.aug = aug
        self.augmentations = Augmentations(dataset)
        self.dataset = dataset

    def run(self, X):

        if self.dataset.find('RSNA') == -1:
            X = np.transpose(X.astype(np.float32), (2, 0, 1))

        if self.aug:
            X = self.augmentations.run(X)

        X = self.imagenet_normalize(X)

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

        prob = 0.5

        if dataset == 'APTOS_1':
            self.augs = A.Compose(
                [
                    # A.Blur(blur_limit=3, p=prob),
                    A.HorizontalFlip(p=prob),
                    A.VerticalFlip(p=prob),
                    A.Rotate(limit=90, p=prob),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=prob),
                    # A.RandomSizedCrop(min_max_height=(180, 220), height=256, width=256, p=prob),
                    A.RandomGamma(gamma_limit=(80, 120), p=prob),
                ]
            )
        elif dataset == 'APTOS_2':
            self.augs = A.Compose(
                [
                    # A.Blur(blur_limit=3, p=prob),
                    # A.HorizontalFlip(p=prob),
                    # A.VerticalFlip(p=prob),
                    # A.Rotate(limit=90, p=prob),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=prob),
                    # A.RandomSizedCrop(min_max_height=(180, 220), height=256, width=256, p=prob),
                    # A.RandomGamma(gamma_limit=(80, 120), p=prob),
                ]
            )
        elif dataset == 'RSNA_1':
            self.augs = A.Compose(
                [
                    # A.Blur(blur_limit=3, p=prob),
                    # A.HorizontalFlip(p=prob),
                    # A.VerticalFlip(p=prob),
                    # A.Rotate(limit=90, p=prob),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=prob),
                    # A.RandomSizedCrop(min_max_height=(180, 220), height=256, width=256, p=prob),
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
