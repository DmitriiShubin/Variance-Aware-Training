# basic libs
import numpy as np
from tqdm import tqdm
import os
import yaml

# pytorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom modules
from metrics import Metric
from utils.pytorchtools import EarlyStopping
from torch.nn.parallel import DataParallel as DP
from utils.post_processing import Post_Processing
from time import time
from utils.loss_functions import TripletLoss

# model
from models.encoder_triplet.structure import Encoder_triplet


class Model:
    """
    This class handles basic methods for handling the model:
    1. Fit the model
    2. Make predictions
    3. Make inference predictions
    3. Save
    4. Load weights
    5. Restore the model
    6. Restore the model with averaged weights
    """

    def __init__(self, hparams, gpu=None, inference=False):

        self.hparams = hparams
        self.gpu = gpu
        self.inference = inference

        self.start_training = time()

        # ininialize model architecture
        self.__setup_model(inference=inference, gpu=gpu)

        # define model parameters
        self.__setup_model_hparams()

        # declare preprocessing object
        self.postprocessing = Post_Processing()
        self.__seed_everything(42)

    def fit(self, train, valid):

        # setup train and val dataloaders
        train_loader = DataLoader(
            train,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=self.hparams['num_workers'],
        )
        valid_loader = DataLoader(
            valid,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=self.hparams['num_workers'],
        )

        # tensorboard
        writer = SummaryWriter(f"runs/{self.hparams['model_name']}_{self.start_training}")

        print('Start training the model')
        for epoch in range(self.hparams['n_epochs']):

            # training mode
            self.model.train()
            avg_loss = 0.0

            for X_anchor, X_positive, X_negative in tqdm(train_loader):

                # push the data into the GPU
                X_anchor = X_anchor.float().to(self.device)
                X_positive = X_positive.float().to(self.device)
                X_negative = X_negative.float().to(self.device)

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                pred_anchor = self.model(X_anchor)
                pred_positive = self.model(X_positive)
                pred_negative = self.model(X_negative)

                train_loss = self.loss(pred_anchor, pred_positive, pred_negative)

                # remove data from GPU
                X_anchor = X_anchor.float().cpu().detach()
                X_positive = X_positive.float().cpu().detach()
                X_negative = X_negative.float().cpu().detach()
                pred_anchor = pred_anchor.float().cpu().detach()
                pred_positive = pred_positive.float().cpu().detach()
                pred_negative = pred_negative.float().cpu().detach()

                # calc loss
                avg_loss += train_loss.item() / len(train_loader)

                # gradient clipping
                if self.apply_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                # backprop
                train_loss.backward()

                # iptimizer step
                self.optimizer.step()

                # evaluate the model
            print('Model evaluation')

            # val mode
            self.model.eval()
            avg_val_loss = 0.0

            with torch.no_grad():

                for X_anchor, X_positive, X_negative in tqdm(valid_loader):
                    # push the data into the GPU
                    X_anchor = X_anchor.float().to(self.device)
                    X_positive = X_positive.float().to(self.device)
                    X_negative = X_negative.float().to(self.device)

                    # get model predictions
                    pred_anchor = self.model(X_anchor)
                    pred_positive = self.model(X_positive)
                    pred_negative = self.model(X_negative)

                    avg_val_loss += self.loss(pred_anchor, pred_positive, pred_negative).item() / len(
                        valid_loader
                    )

                    # remove data from GPU
                    X_anchor = X_anchor.float().cpu().detach()
                    X_positive = X_positive.float().cpu().detach()
                    X_negative = X_negative.float().cpu().detach()
                    pred_anchor = pred_anchor.float().cpu().detach()
                    pred_positive = pred_positive.float().cpu().detach()
                    pred_negative = pred_negative.float().cpu().detach()

            # early stopping for scheduler
            if self.hparams['scheduler_name'] == 'ReduceLROnPlateau':
                self.scheduler.step(avg_val_loss)
            else:
                self.scheduler.step()

            es_result = self.early_stopping(score=avg_val_loss, model=self.model, threshold=None)

            # print statistics
            if self.hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| Current LR: ',
                    self.__get_lr(self.optimizer),
                )

            # add data to tensorboard
            writer.add_scalars(
                'Loss',
                {'Train_loss': avg_loss, 'Val_loss': avg_val_loss},
                epoch,
            )

            # early stopping procesudre
            if es_result == 2:
                print("Early Stopping")
                print(f'global best val_loss model score {self.early_stopping.best_score}')
                break
            elif es_result == 1:
                print(f'save global val_loss model score {avg_val_loss}')

        writer.close()

        # load the best model trained so fat
        self.model = self.early_stopping.load_best_weights()

        return self.start_training

    def predict(self, X_test):
        """
        This function makes:
        1. batch-wise predictions
        2. calculation of the metric for each sample
        3. calculation of the metric for the entire dataset

        Parameters
        ----------
        X_test

        Returns
        -------

        """

        # evaluate the model
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            X_test,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=0,
        )

        avg_test_loss = 0.0

        print('Getting predictions')
        with torch.no_grad():
            for X_anchor, X_positive, X_negative in tqdm(test_loader):
                # push the data into the GPU
                X_anchor = X_anchor.float().to(self.device)
                X_positive = X_positive.float().to(self.device)
                X_negative = X_negative.float().to(self.device)

                # get model predictions
                pred_anchor = self.model(X_anchor)
                pred_positive = self.model(X_positive)
                pred_negative = self.model(X_negative)

                avg_test_loss += self.loss(pred_anchor, pred_positive, pred_negative).item() / len(
                    test_loader
                )

                # remove data from GPU
                X_anchor = X_anchor.float().cpu().detach()
                X_positive = X_positive.float().cpu().detach()
                X_negative = X_negative.float().cpu().detach()
                pred_anchor = pred_anchor.float().cpu().detach()
                pred_positive = pred_positive.float().cpu().detach()
                pred_negative = pred_negative.float().cpu().detach()

        return avg_test_loss

    def save(self, model_path):

        print('Saving the model')

        # states (weights + optimizers)
        if self.gpu != None:
            if len(self.gpu) > 1:
                torch.save(self.model.module.state_dict(), model_path + '.pt')
            else:
                torch.save(self.model.state_dict(), model_path + '.pt')
        else:
            torch.save(self.model.state_dict(), model_path)

        # hparams
        with open(f"{model_path}_hparams.yml", 'w') as file:
            yaml.dump(self.hparams, file)

        return True

    def load(self, model_name):
        self.model.load_state_dict(torch.load(model_name + '.pt', map_location=self.device))
        self.model.eval()
        return True

    @classmethod
    def restore(cls, model_name: str, gpu: list, inference: bool):

        if gpu is not None:
            assert all([isinstance(i, int) for i in gpu]), "All gpu indexes should be integer"

        # load hparams
        hparams = yaml.load(open(model_name + "_hparams.yml"), Loader=yaml.FullLoader)

        # construct class
        self = cls(hparams, gpu=gpu, inference=inference)

        # load weights + optimizer state
        self.load(model_name=model_name)

        return self

    ################## Utils #####################

    def __get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def __setup_model(self, inference, gpu):

        # TODO: re-write to pure DDP
        if inference or gpu is None:
            self.device = torch.device('cpu')
            self.model = Encoder_triplet(hparams=self.hparams['model']).to(self.device)
        else:
            if torch.cuda.device_count() > 1:
                if len(gpu) > 1:
                    print("Number of GPUs will be used: ", len(gpu))
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = Encoder_triplet(hparams=self.hparams['model']).to(self.device)
                    self.model = DP(self.model, device_ids=gpu, output_device=gpu[0])
                else:
                    print("Only one GPU will be used")
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = Encoder_triplet(hparams=self.hparams['model']).to(self.device)
            else:
                self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                self.model = Encoder_triplet(hparams=self.hparams['model']).to(self.device)
                print('Only one GPU is available')

        print('Cuda available: ', torch.cuda.is_available())

        return True

    def __setup_model_hparams(self):

        # 1. define losses
        self.loss = TripletLoss()

        # 2. define optimizer
        self.optimizer = eval(f"torch.optim.{self.hparams['optimizer_name']}")(
            params=self.model.parameters(), **self.hparams['optimizer_hparams']
        )

        # 3. define scheduler
        self.scheduler = eval(f"torch.optim.lr_scheduler.{self.hparams['scheduler_name']}")(
            optimizer=self.optimizer, **self.hparams['scheduler_hparams']
        )

        # 4. define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=self.hparams['checkpoint_path'] + f'/checkpoint_{self.start_training}' + '.pt',
            patience=self.hparams['patience'],
            delta=self.hparams['min_delta'],
            is_maximize=False,
        )

        # 5. set gradient clipping
        self.apply_clipping = self.hparams['clipping']  # clipping of gradients

        # 6. Set scaler for optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        return True

    def __seed_everything(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
