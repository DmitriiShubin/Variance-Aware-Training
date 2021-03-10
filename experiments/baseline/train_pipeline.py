# import
import numpy as np
import json
import pandas as pd
import torch
import os


from experiments.baseline.data_generator import Dataset_train
from metrics import Metric


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class TrainPipeline:
    def __init__(self, hparams, gpu, model):

        # load the model

        self.hparams = hparams
        self.gpu = gpu

        print('\n')
        print('Selected Learning rate:', self.hparams['optimizer_hparams']['lr'])
        print('\n')

        self.exclusions = []

        self.splits, self.splits_test = self.load_split_table()
        self.metric = Metric(self.hparams['model']['n_classes'])

        self.model = model

    def load_split_table(self):

        splits_cv = pd.DataFrame([json.load(open(self.hparams['split_table_path']))])

        splits_test = pd.DataFrame([json.load(open(self.hparams['test_split_table_path']))])

        return splits_cv, splits_test

    def train(self):

        start_training = []
        fold_scores_val = []
        fold_scores_test = []

        self.model = self.model(hparams=self.hparams, gpu=self.gpu)

        train = Dataset_train(
            self.splits['train'].values[0], aug=False, n_classes=self.hparams['model']['n_classes']
        )
        valid = Dataset_train(
            self.splits['val'].values[0], aug=False, n_classes=self.hparams['model']['n_classes']
        )
        test = Dataset_train(
            self.splits_test['test'].values[0], aug=False, n_classes=self.hparams['model']['n_classes']
        )

        # train model
        start_training.append(self.model.fit(train=train, valid=valid))

        # get model predictions
        error_val, fold_score, pred_val = self.model.predict(valid)
        error_test, fold_score_test, pred_test = self.model.predict(test)

        print("Model's final scrore, cv: ", fold_score)
        print("Model's final scrore, test: ", fold_score_test)

        fold_scores_val.append(fold_score)
        fold_scores_test.append(fold_score_test)

        # save the model
        self.model.save(
            self.hparams['model_path']
            + '/'
            + self.hparams['model_name']
            + f"_{self.hparams}"
            + '_fold_'
            + str(np.round(fold_score, 2))
            + '_'
            + str(np.round(fold_score_test, 2))
            + '_'
            + str(start_training)
        )

        # save data for debug
        # self.save_debug_data(error_val, pred_val, self.splits['val'].values[fold])
        self.save_debug_data(error_test, pred_test, self.splits_test['test'].values[0])

        return fold_scores_val, fold_scores_test, start_training

    def save_debug_data(self, error, pred, validation_list):

        for index, data in enumerate(validation_list):

            patient_fold = data.split('/')[-2]
            data = data.split('/')[-1]

            out_json = {}
            out_json['prediction'] = pred[index].tolist()
            out_json['error'] = error[index].tolist()

            os.makedirs(DEBUG_FOLDER + patient_fold, exist_ok=True)
            # save debug data
            with open(DEBUG_FOLDER + patient_fold + '/' + f'{data[:-4]}.json', 'w') as outfile:
                json.dump(out_json, outfile)

        return True
