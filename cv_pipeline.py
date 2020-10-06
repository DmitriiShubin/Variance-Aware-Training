# import
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
import os
from tqdm import tqdm

from data_generator import Dataset_train, Dataset_test
from metrics import Metric
from postprocessing import PostProcessing

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


seed_everything(42)


class CVPipeline:
    def __init__(self, hparams, split_table_path, split_table_name, debug_folder, model, gpu,downsample):

        # load the model

        self.hparams = hparams
        self.model = model
        self.gpu = gpu
        self.downsample = downsample

        print('\n')
        print('Selected Learning rate:', self.hparams['lr'])
        print('\n')

        self.debug_folder = debug_folder
        self.split_table_path = split_table_path
        self.split_table_name = split_table_name
        self.exclusions = []


        self.splits = self.load_split_table()
        self.metric = Metric()



    def load_split_table(self):

        splits = []

        split_files = [i for i in os.listdir(self.split_table_path) if i.find('.json')!=-1]

        for i in range(len(split_files)):
            data = json.load(open(self.split_table_path + str(i) + '_' + self.split_table_name))

            splits.append(data)

        splits = pd.DataFrame(splits)

        return splits

    def train(self):

        score = 0
        for fold in range(self.splits.shape[0]):

            if fold is not None:
                if fold != self.hparams['start_fold']:
                    continue
            #TODO
            train = Dataset_train(self.splits['train'].values[fold][:2], aug=True,downsample=self.downsample)
            valid = Dataset_train(self.splits['val'].values[fold][:2], aug=False,downsample=self.downsample)

            X, y,_,_ = train.__getitem__(0)
            self.model = self.model(n_channels=X.shape[0], hparams=self.hparams, gpu=self.gpu
            )

            # train model
            self.model.fit(train=train, valid=valid)

            # get model predictions
            y_val,pred_val = self.model.predict(valid)
            self.postprocessing = PostProcessing(fold=self.hparams['start_fold'])

            pred_val_processed = self.postprocessing.run(pred_val)

            # TODO: add activations
            # heatmap = self.model.get_heatmap(valid)


            fold_score = self.metric.compute(y_val, pred_val_processed)
            print("Model's final scrore: ",fold_score)
            # save the model
            self.model.model_save(
                self.hparams['model_path']
                + self.hparams['model_name']+f"_{self.hparams['start_fold']}"
                + '_fold_'
                + str(fold_score)
                + '.pt'
            )


            images_list = valid.images_list.copy()

            for index,record in enumerate(images_list):
                a = record.split('/')
                images_list[index] = f'{a[-2]}/{a[-1]}'

            # create a dictionary for debugging
            self.save_debug_data(pred_val, images_list)



        return fold_score

    def save_debug_data(self, pred_val, validation_list):

        for index, data in enumerate(validation_list):

            data_folder = f'./data/CV_debug/'
            patient_fold = data.split('/')[-2]

            prediction = {}
            prediction['predicted_label'] = pred_val[index].tolist()
            os.makedirs(data_folder + patient_fold, exist_ok=True)
            # save debug data
            with open(data_folder +data+ '.json', 'w') as outfile:
                json.dump(prediction, outfile)

        return True
