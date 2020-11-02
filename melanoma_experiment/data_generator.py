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
import itertools
import pandas as pd

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
            images = [i for i in os.listdir(DATA_PATH + patient) if i.find('.jpg') != -1]
            for image in images:
                self.images_list.append(DATA_PATH + patient + '/' + image)

        # read labels table
        self.df = pd.read_csv('../data/melanoma/train.csv')
        self.df = self.df.drop(
            [
                'patient_id',
                'sex',
                'age_approx',
                'anatom_site_general_challenge',
                'diagnosis',
                'benign_malignant',
            ],
            axis=1,
        )
        self.df.index = self.df['image_name']
        self.df = self.df.drop(['image_name'], axis=1).to_dict('index')
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

        X = cv2.imread(self.images_list[id])

        y = [self.df[self.images_list[id].split('/')[-1][:-4]]['target']]

        X, y = self.preprocessing.run(X=X, y=y)

        sampled_patient = np.random.uniform(size=1)[0]
        if sampled_patient >= 0.5:
            # NOT the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id) == -1]

            X_s = cv2.imread(np.random.choice(np.array(images_subset)))
            y_s = [0]
        else:
            # the same patient
            images_subset = self.images_list.copy()
            patient_id = self.images_list[id].split('/')[-2]
            images_subset = [i for i in images_subset if i.find(patient_id) != -1]
            images_subset.remove(self.images_list[id])

            X_s = cv2.imread(np.random.choice(np.array(images_subset)))
            y_s = [1]

        X_s, y_ = self.preprocessing.run(X=X_s, y=y_s)

        return X, y, X_s, y_s


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations(0.0)

    def run(self, X, y, label_process=True):

        # apply scaling
        X = X.astype(np.float32) / 255

        # reshape to match pytorch
        X = X.transpose(2, 0, 1)

        if label_process:
            return X, y
        else:
            return X


class Augmentations:
    def __init__(self, prob):

        self.augs = A.Compose(
            [
                # A.HorizontalFlip(p=prob),
                A.Rotate(limit=10, p=prob),
                A.RandomSizedCrop(min_max_height=(200, 200), height=240, width=240, p=prob),
            ]
        )

    def apply_augs(self, image, mask):

        # apply horizontal flip
        augmented = self.augs(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        return image, mask
