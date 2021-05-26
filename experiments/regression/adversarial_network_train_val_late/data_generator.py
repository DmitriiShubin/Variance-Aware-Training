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
    def __init__(self, volums_list, aug, n_classes):

        self.n_classes = n_classes
        self.volums_list = volums_list
        self.preprocessing = Preprocessing(aug)

    def __len__(self):
        return len(self.volums_list)

    def __getitem__(self, idx):

        X, y, X_s, y_s = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        X_s = torch.tensor(X_s, dtype=torch.float)
        y_s = torch.tensor(y_s, dtype=torch.float)

        return X, y, X_s, y_s

    def load_data(self, id):

        X = np.load(self.volums_list[id]).astype(np.float32)
        y = np.load(self.volums_list[id][:-4] + '_label.npy').astype(np.float32)


        X = self.preprocessing.run(X=X)

        # second head
        images_subset = self.volums_list.copy()
        images_subset.remove(self.volums_list[id])
        X_s = np.load(np.random.choice(np.array(images_subset))).astype(np.float32)

        y_s = [0]
        X_s = self.preprocessing.run(X=X_s)

        return X, y, X_s, y_s


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations()

    def run(self, X):

        if self.aug:
            X = self.augmentations.run(X)

        X = self.imagenet_normalize(X)

        X = np.transpose(X.astype(np.float32), (2, 0, 1))

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


class Augmentations:
    def __init__(self):

        prob = 0.5
        self.augs = A.Compose(
            [
                A.HorizontalFlip(p=prob),
                A.VerticalFlip(p=prob),
                A.Rotate(limit=5, p=prob),
                # A.GlassBlur(sigma=1),
                # A.GridDistortion(distort_limit=0.3),
                # A.ElasticTransform(alpha=0.05, p=prob),
                A.RandomSizedCrop(min_max_height=(180, 220), height=256, width=256, p=prob),
                A.RandomGamma(gamma_limit=(80, 120), p=prob),
                # A.RandomBrightness(limit=0.2, p=prob)
            ]
        )

    def run(self, image):

        # apply augs
        augmented = self.augs(image=image)
        image = augmented['image']

        return image
