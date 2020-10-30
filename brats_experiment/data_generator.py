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
                i[:-10]
                for i in os.listdir(DATA_PATH + patient)
                if i.find('.npy') != -1 and i.find('flair') != -1
            ]
            for image in images:
                self.images_list.append(DATA_PATH + patient + '/' + image)

        #read dataset info
        self.datasets = json.load(open('./split_table/adversarial_datasets.json'))

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

        X = np.load(self.images_list[id] + '_flair.npy')
        X = np.append(X, np.load(self.images_list[id] + '_t1.npy'), axis=2)
        X = np.append(X, np.load(self.images_list[id] + '_t1ce.npy'), axis=2)
        X = np.append(X, np.load(self.images_list[id] + '_t2.npy'), axis=2)

        y = np.load(self.images_list[id] + '_seg.npy').astype(np.float32)
        y_ = y.copy()

        X, y = self.preprocessing.run(X=X, y=y)


        # second head
        for i in self.datasets.keys():
            if self.images_list[id].split('/')[-1][:20] in self.datasets[i]:
                dataset_number = i


        sampled_patient = np.random.uniform(size=1)[0]
        if sampled_patient >= 0.5:
            # NOT the same patient
            dataset_adv = list(self.datasets.keys())
            dataset_adv.remove(dataset_number)
            dataset_adv = np.random.choice(dataset_adv)
            images_subset = self.images_list.copy()
            images_subset = [i for i in images_subset if (i.split('/')[-1][:20] in self.datasets[dataset_adv])]
            X_s = np.load(np.random.choice(np.array(images_subset)) + '_flair.npy')
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1.npy'), axis=2)
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1ce.npy'), axis=2)
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t2.npy'), axis=2)
            y_s = [0]

        else:
            # the same patient
            images_subset = self.images_list.copy()
            images_subset = [i for i in images_subset if (i.split('/')[-1][:20] in self.datasets[dataset_number]) and (self.images_list[id] != i)]
            X_s = np.load(np.random.choice(np.array(images_subset)) + '_flair.npy')
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1.npy'), axis=2)
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1ce.npy'), axis=2)
            X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t2.npy'), axis=2)
            y_s = [1]


        # sampled_patient = np.random.uniform(size=1)[0]
        # if sampled_patient >= 0.5:
        #     # NOT the same patient
        #     images_subset = self.images_list.copy()
        #     patient_id = self.images_list[id].split('/')[-2]
        #     images_subset = [i for i in images_subset if i.find(patient_id) == -1]
        #
        #     X_s = np.load(np.random.choice(np.array(images_subset)) + '_flair.npy')
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1.npy'), axis=2)
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1ce.npy'), axis=2)
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t2.npy'), axis=2)
        #
        #     y_s = [0]
        # else:
        #     # the same patient
        #     images_subset = self.images_list.copy()
        #     patient_id = self.images_list[id].split('/')[-2]
        #     images_subset = [i for i in images_subset if i.find(patient_id) != -1]
        #     images_subset.remove(self.images_list[id])
        #
        #     X_s = np.load(np.random.choice(np.array(images_subset)) + '_flair.npy')
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1.npy'), axis=2)
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t1ce.npy'), axis=2)
        #     X_s = np.append(X_s, np.load(np.random.choice(np.array(images_subset)) + '_t2.npy'), axis=2)
        #     y_s = [1]


        X_s, y_ = self.preprocessing.run(X=X_s, y=y_)

        return X, y, X_s, y_s


class Preprocessing:
    def __init__(self, aug):

        self.aug = aug
        self.augmentations = Augmentations(0.0)

    def run(self, X, y, label_process=True):

        # apply scaling
        for i in range(4):
            if np.std(X[:, :, i]) > 0:
                X[:, :, i] = (X[:, :, i] - np.mean(X[:, :, i])) / np.std(X[:, :, i])
            else:
                X[:, :, i] = X[:, :, i] - np.mean(X[:, :, i])

        y[np.where(y == 4)] = 3

        a = np.where(y == 1)

        y = np.eye(4, dtype=np.float32)[y.astype(np.int8)]
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
                #A.HorizontalFlip(p=prob),
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
