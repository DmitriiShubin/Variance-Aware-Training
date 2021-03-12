# basic libs
import numpy as np
import torch

# pytorch
import torch
from torch.utils.data import Dataset
import albumentations

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
        y = np.transpose(y.astype(np.int32), (1, 2, 3, 0))
        y = np.eye(self.n_classes)[y[:, :, :, -1]]
        y = np.transpose(y.astype(np.int32), (3, 0, 1, 2))
        return y


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations()

    def run(self, X, y):

        X = self.std_scaling(X)

        # if self.aug:
        #     X,y = self.augmentations.run(X,y)

        return X, y

    def minmax_scaling(self, X):
        min = np.min(X)
        max = np.max(X)
        if max > min + 1e-3:
            X = (X - min) / (max - min)
        else:
            X = X - np.mean(X)
        return X

    def std_scaling(self, X):
        std = np.std(X)
        mean = np.mean(X)
        if std > 0:
            return (X - mean) / std
        else:
            return X - mean

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
    def __init__(self, prob=0.5):
        self.prob_major = prob
        # TODO: add albumentations

    def run(self, X, y):

        X = self.random_intentity(X)

        return X, y

    def rotate_batch_3d(self, x, y):

        x = np.concatenate([x, y], axis=0)

        batch_size = x.shape[0]
        y = np.zeros((batch_size, 10))
        rotated_batch = []
        for index, volume in enumerate(x):
            rot = np.random.random_integers(10) - 1

            if rot == 1:
                volume = np.transpose(np.flip(volume, 1), (1, 0, 2, 3))  # 90 deg Z
            elif rot == 2:
                volume = np.flip(volume, (0, 1))  # 180 degrees on z axis
            elif rot == 3:
                volume = np.flip(np.transpose(volume, (1, 0, 2, 3)), 1)  # 90 deg Z
            elif rot == 4:
                volume = np.transpose(np.flip(volume, 1), (0, 2, 1, 3))  # 90 deg X
            elif rot == 5:
                volume = np.flip(volume, (1, 2))  # 180 degrees on x axis
            elif rot == 6:
                volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 1)  # 90 deg X
            elif rot == 7:
                volume = np.transpose(np.flip(volume, 0), (2, 1, 0, 3))  # 90 deg Y
            elif rot == 8:
                volume = np.flip(volume, (0, 2))  # 180 degrees on y axis
            elif rot == 10:
                volume = np.flip(np.transpose(volume, (2, 1, 0, 3)), 0)  # 90 deg Y

            rotated_batch.append(volume)
            y[index, rot] = 1
        return np.stack(rotated_batch)

    def random_intentity(self, X, intensity=0.2):

        prob = np.random.uniform()

        if prob >= self.prob_major:
            X *= X * ((0.5 - np.random.uniform()) * intensity)

        return X
