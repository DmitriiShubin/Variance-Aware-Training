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
    def __init__(self, volumes_list, aug, dataset):

        self.volumes_list = volumes_list
        self.preprocessing = Preprocessing(aug, dataset)

        self.generate_pairs(n_pairs=int(len(self.volumes_list) *1))

    # TODO
    def generate_pairs(self, n_pairs: int):

        # create a list of labels
        labels = []
        for volume in self.volumes_list:
            labels.append(volume.split('/')[-2])

        labels = np.array(labels)

        self.volumes_list = np.array(self.volumes_list)

        # generate pairs
        self.pairs_list = []
        for i in range(n_pairs):
            pairs = {}

            # select anchor
            anchor = self.volumes_list[np.random.choice(self.volumes_list.shape[0])]
            anchor_label = labels[np.where(np.array(self.volumes_list) == anchor)]
            pairs['anchor'] = anchor

            # select positive
            records_pos_subset = self.volumes_list[np.where(labels == anchor_label)]
            # records_pos_subset = records_pos_subset[records_pos_subset != anchor]
            pairs['positive'] = records_pos_subset[np.random.choice(records_pos_subset.shape[0])]

            # select negative
            records_neg_subset = self.volumes_list[np.where(labels != anchor_label)]
            pairs['negative'] = records_neg_subset[np.random.choice(records_neg_subset.shape[0])]
            self.pairs_list.append(pairs)

        self.volumes_list = self.volumes_list.tolist()

        return True

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):

        X_anchor, X_positive, X_negative = self.load_data(idx)

        X_anchor = torch.tensor(X_anchor, dtype=torch.float)
        X_positive = torch.tensor(X_positive, dtype=torch.float)
        X_negative = torch.tensor(X_negative, dtype=torch.float)

        return X_anchor, X_positive, X_negative

    def load_data(self, id):

        X_anchor = np.load(self.pairs_list[id]['anchor'])
        X_positive = np.load(self.pairs_list[id]['positive'])
        X_negative = np.load(self.pairs_list[id]['negative'])

        X_anchor = self.preprocessing.run(X_anchor)
        X_positive = self.preprocessing.run(X_positive, aug=True)
        X_negative = self.preprocessing.run(X_negative, aug=True)

        return X_anchor, X_positive, X_negative


class Preprocessing:
    def __init__(self, aug, dataset):

        self.aug = aug
        self.augmentations = Augmentations(dataset)

    def run(self, X, aug=False):

        if aug:
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
                [   #A.Blur(blur_limit=1,p=prob),
                    A.HorizontalFlip(p=prob),
                    A.VerticalFlip(p=prob),
                    A.Rotate(limit=10, p=prob),
                    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=prob),
                    A.RandomSizedCrop(min_max_height=(210, 210), height=240, width=240, p=prob),
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
