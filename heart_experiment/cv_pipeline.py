# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import os
from tqdm import tqdm
from config import SPLIT_TABLE_PATH, SPLIT_TABLE_NAME, DEBUG_FOLDER

from data_generator import Dataset_train
from metrics import Metric
from postprocessing import PostProcessing


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(self, hparams, gpu, model):

        # load the model

        self.hparams = hparams
        self.gpu = gpu

        print('\n')
        print('Selected Learning rate:', self.hparams['lr'])
        print('\n')

        self.exclusions = []

        self.splits = self.load_split_table()
        self.metric = Metric()

        self.model = model

    def load_split_table(self):

        splits = []

        split_files = [i for i in os.listdir(SPLIT_TABLE_PATH) if i.find('table.json') != -1]

        for i in range(len(split_files)):
            data = json.load(open(SPLIT_TABLE_PATH + str(i) + '_' + SPLIT_TABLE_NAME))

            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold != self.hparams['start_fold']:
                    continue
            # TODO
            train = Dataset_train(self.splits['train'].values[fold][:2], aug=False)
            valid = Dataset_train(self.splits['val'].values[fold][:2], aug=False)

            X, y, _, _ = train.__getitem__(0)
            self.model = self.model(n_channels=X.shape[0], hparams=self.hparams, gpu=self.gpu)

            # train model
            #start_training = self.model.fit(train=train, valid=valid)

            # get model predictions
            y_val, pred_val = self.model.predict(valid)

            pred_val_processed = np.argmax(pred_val, axis=1)
            y_val = np.argmax(y_val, axis=1)

            pred_val_processed = pred_val_processed.reshape(-1)
            y_val = y_val.reshape(-1)

            self.metric.calc_cm(labels=y_val, outputs=pred_val_processed)
            fold_score = self.metric.compute()  # y_val, pred_val_processed)
            print("Model's final scrore: ", fold_score)
            # save the model
            self.model.model_save(
                self.hparams['model_path']
                + self.hparams['model_name']
                + f"_{self.hparams['start_fold']}"
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )

        return fold_score, start_training

    def save_debug_data(self, pred_val, validation_list):

        for index, data in enumerate(validation_list):

            patient_fold = data.split('/')[-2]

            prediction = {}
            prediction['predicted_label'] = pred_val[index].tolist()
            os.makedirs(DEBUG_FOLDER + patient_fold, exist_ok=True)
            # save debug data
            with open(DEBUG_FOLDER + data + '.json', 'w') as outfile:
                json.dump(prediction, outfile)

        return True
