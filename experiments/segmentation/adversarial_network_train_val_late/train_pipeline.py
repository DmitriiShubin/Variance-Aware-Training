# import
import numpy as np
import json
import pandas as pd
import torch
import os


from metrics import Dice_score


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class TrainPipeline:
    def __init__(self, hparams, gpu, model, Dataset_train):

        # load the model

        self.hparams = hparams
        self.gpu = gpu
        self.Dataset_train = Dataset_train

        print('\n')
        print('Selected Learning rate:', self.hparams['optimizer_hparams']['lr'])
        print('\n')

        self.exclusions = []

        self.splits, self.splits_test = self.load_split_table()
        self.metric = Dice_score(self.hparams['model']['n_classes'])

        self.model = model

    def load_split_table(self):

        splits_cv = pd.DataFrame([json.load(open(self.hparams['split_table_path']))])

        splits_test = pd.DataFrame([json.load(open(self.hparams['test_split_table_path']))])

        return splits_cv, splits_test

    def train(self):

        self.model = self.model(hparams=self.hparams, gpu=self.gpu)

        train = self.Dataset_train(
            self.splits['train'].values[0],
            aug=True,
            n_classes=self.hparams['model']['n_classes'],
            dataset=self.hparams['dataset'],
        )
        valid = self.Dataset_train(
            self.splits['val'].values[0],
            aug=False,
            n_classes=self.hparams['model']['n_classes'],
            dataset=self.hparams['dataset'],
        )
        pretrain = self.Dataset_train(
            self.splits['pretrain'].values[0],
            aug=True,
            n_classes=self.hparams['model']['n_classes'],
            dataset=self.hparams['dataset'],
        )
        test = self.Dataset_train(
            self.splits_test['test'].values[0],
            aug=False,
            n_classes=self.hparams['model']['n_classes'],
            dataset=self.hparams['dataset'],
        )

        # train model
        start_training = self.model.fit(train=train, valid=valid, pretrain=pretrain)

        # get model predictions
        error_val, fold_score = self.model.predict(valid)
        error_test, fold_score_test = self.model.predict(test)

        print("Model's final scrore, cv: ", fold_score)
        print("Model's final scrore, test: ", fold_score_test)

        # save the model
        self.model.save(
            self.hparams['model_path']
            + self.hparams['model_name']
            + f"_{self.hparams['split_table_path'].split('/')[-1][:-5]}"
            + '_fold_'
            + str(np.round(fold_score, 2))
            + '_'
            + str(np.round(fold_score_test, 2))
            + '_'
            + str(start_training)
        )

        # save data for debug
        self.save_debug_data(error_val, self.splits['val'].values[0])
        self.save_debug_data(error_test, self.splits_test['test'].values[0])

        return fold_score, fold_score_test, start_training

    def save_debug_data(self, error, validation_list):

        for index, data in enumerate(validation_list):

            patient_fold = data.split('/')[-2]
            data = data.split('/')[-1]

            out_json = {}
            out_json['error'] = error[index].tolist()

            os.makedirs(self.hparams['debug_path'] + patient_fold, exist_ok=True)
            # save debug data
            with open(self.hparams['debug_path'] + patient_fold + '/' + f'{data[:-4]}.json', 'w') as outfile:
                json.dump(out_json, outfile)

        return True
