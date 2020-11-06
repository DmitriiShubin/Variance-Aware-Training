# basic libs
import numpy as np
import json
import os
import gc
import random
from scipy import signal
import cv2
import albumentations as A
from config import DATA_PATH
from time import time

# pytorch
import torch
from torch.utils.data import Dataset

# custom modules
np.random.seed(42)


class Dataset_train(Dataset):
    def __init__(self, patients, aug):

        self.patients = patients
        self.images_list = []

        for patient in patients:
            images = [
                i[:-4]
                for i in os.listdir(DATA_PATH + patient)
                if i.find('.npy') != -1 and i.find('seg') == -1
            ]
            for image in images:
                self.images_list.append(DATA_PATH + patient + '/' + image)

        self.preprocessing = Preprocessing(aug)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        X, y, X_s, y_s = self.load_data(idx)

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        X_s = torch.tensor(X_s, dtype=torch.float)
        y_s = torch.tensor(y_s, dtype=torch.float)

        return X, y, X_s, y_s

    def load_data(self, id):

        X = np.load(self.images_list[id] + '.npy').astype(np.float32)

        y = np.load(self.images_list[id] + '_seg.npy').astype(
            np.float32
        )  # cv2.imread(self.images_list[id] + '_mask.tif', cv2.IMREAD_COLOR)
        y_ = y.copy()

        X, y = self.preprocessing.run(X=X, y=y)

        # second head
        sampled_patient = np.round(np.random.uniform(size=1)[0],1)
        if sampled_patient >= 0.5:
            # NOT the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id) == -1]

            X_s = np.load(np.random.choice(np.array(images_subset)) + '.npy')

            y_s = [0]
        else:
            # the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id) != -1]
            images_subset.remove(self.images_list[id])

            X_s = np.load(np.random.choice(np.array(images_subset)) + '.npy')
            y_s = [1]

        X_s, y_ = self.preprocessing.run(X=X_s, y=y_)

        return X, y, X_s, y_s


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations(0.0)

    def run(self, X, y, label_process=True):

        # X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # if self.aug:
        #     X, y = self.augmentations.apply_augs(X, y)

        # apply scaling
        for i in range(X.shape[2]):
            if np.std(X[:, :, i]) > 0:
                X[:, :, i] = (X[:, :, i] - np.mean(X[:, :, i])) / np.std(X[:, :, i])
            else:
                X[:, :, i] = X[:, :, i] - np.mean(X[:, :, i])

        y = np.eye(2, dtype=np.float32)[y.astype(np.int8)]
        y = y.reshape(y.shape[0], y.shape[1], y.shape[-1])

        # reshape to match pytorch
        X = X.transpose(2, 0, 1)
        y = y.transpose(2, 0, 1)

        if label_process:
            return X, y
        else:
            return X


class Augmentations:
    def __init__(self, prob):

        self.augs = A.Compose(
            [
                A.HorizontalFlip(p=prob),
                A.Rotate(limit=10, p=prob),
                A.RandomSizedCrop(min_max_height=(240, 240), height=240, width=240, p=prob),
            ]
        )

    def apply_augs(self, image, mask):

        # apply horizontal flip
        augmented = self.augs(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        return image, mask
