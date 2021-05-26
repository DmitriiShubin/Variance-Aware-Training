# basic libs
import numpy as np
from tqdm import tqdm
import os
import yaml
import random

# pytorch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP

# custom modules
from metrics import Kappa
from utils.pytorchtools import EarlyStopping
from time import time

# model
from .structure import EfficientNet
from utils.post_processing_regression import Post_Processing


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
        self.postprocessing = Post_Processing()

        # ininialize model architecture
        self.__setup_model(inference=inference, gpu=gpu)

        # define model parameters
        self.__setup_model_hparams()

        # declare preprocessing object
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
            avg_adv_loss = 0.0

            for X_batch, y_batch in tqdm(train_loader):

                # push the data into the GPU
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                pred = self.model(X_batch)

                # process main loss
                pred = pred.reshape(-1)
                y_batch = y_batch.reshape(-1)
                train_loss = self.loss(pred, y_batch)

                # calc loss
                avg_loss += train_loss.item() / len(train_loader)

                # remove data from GPU
                y_batch = y_batch.float().cpu().detach().numpy()
                pred = pred.float().cpu().detach().numpy()
                X_batch = X_batch.float().cpu().detach().numpy()

                # gradient clipping
                if self.apply_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                # backprop
                train_loss.backward()

                # iptimizer step
                self.optimizer.step()

                y_batch = self.postprocessing.run(y_batch)
                pred = self.postprocessing.run(pred)

                # calculate a step for metrics
                self.metric.calc_running_score(labels=y_batch, outputs=pred)

            # calc train metrics
            metric_train = self.metric.compute()

            # evaluate the model
            print('Model evaluation')

            # val mode
            self.model.eval()
            self.optimizer.zero_grad()
            avg_val_loss = 0.0

            with torch.no_grad():

                for X_batch, y_batch in tqdm(valid_loader):

                    # push the data into the GPU
                    X_batch = X_batch.float().to(self.device)
                    y_batch = y_batch.float().to(self.device)

                    # get predictions
                    pred = self.model(X_batch)

                    # calculate main loss
                    pred = pred.reshape(-1)
                    y_batch = y_batch.reshape(-1)

                    avg_val_loss += self.loss(pred, y_batch).item() / len(valid_loader)

                    # remove data from GPU
                    X_batch = X_batch.float().cpu().detach().numpy()
                    pred = pred.float().cpu().detach().numpy()
                    y_batch = y_batch.float().cpu().detach().numpy()

                    y_batch = self.postprocessing.run(y_batch)
                    pred = self.postprocessing.run(pred)

                    # calculate a step for metrics
                    self.metric.calc_running_score(labels=y_batch, outputs=pred)

            # calc val metrics
            metric_val = self.metric.compute()

            # early stopping for scheduler
            if self.hparams['scheduler_name'] == 'ReduceLROnPlateau':
                self.scheduler.step(metric_val)
            else:
                self.scheduler.step()

            es_result = self.early_stopping(score=metric_val, model=self.model, threshold=None)

            # print statistics
            if self.hparams['verbose_train']:
                print(
                    '| Epoch: ',
                    epoch + 1,
                    '| Train_loss: ',
                    avg_loss,
                    '| Val_loss: ',
                    avg_val_loss,
                    '| Adv_loss: ',
                    avg_adv_loss,
                    '| Metric_train: ',
                    metric_train,
                    '| Metric_val: ',
                    metric_val,
                    '| Current LR: ',
                    self.__get_lr(self.optimizer),
                )

            # add data to tensorboard
            writer.add_scalars(
                'Loss', {'Train_loss': avg_loss, 'Val_loss': avg_val_loss}, epoch,
            )
            writer.add_scalars('Metric', {'Metric_train': metric_train, 'Metric_val': metric_val}, epoch)

            # early stopping procesudre
            if es_result == 2:
                print("Early Stopping")
                print(f'global best val_loss model score {self.early_stopping.best_score}')
                break
            elif es_result == 1:
                print(f'save global val_loss model score {metric_val}')

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
            X_test, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0,
        )

        error_samplewise = []

        self.metric.reset()

        print('Getting predictions')
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(tqdm(test_loader)):
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                pred = self.model(X_batch)

                # calculate main loss
                pred = pred.reshape(-1)
                y_batch = y_batch.reshape(-1)

                pred = pred.cpu().detach().numpy()
                X_batch = X_batch.cpu().detach().numpy()
                y_batch = y_batch.cpu().detach().numpy()

                y_batch = self.postprocessing.run(y_batch)
                pred = self.postprocessing.run(pred)

                self.metric.calc_running_score(labels=y_batch, outputs=pred)

        fold_score = self.metric.compute()

        return fold_score

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
            self.model = EfficientNet.from_pretrained(
                self.hparams['model']['pre_trained_model'], num_classes=self.hparams['model']['n_classes']
            ).to(self.device)
            # self.model.freeze_layers()
        else:
            if torch.cuda.device_count() > 1:
                if len(gpu) > 1:
                    print("Number of GPUs will be used: ", len(gpu))
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = EfficientNet.from_pretrained(
                        self.hparams['model']['pre_trained_model'],
                        num_classes=self.hparams['model']['n_classes'],
                    ).to(self.device)
                    self.model = DP(self.model, device_ids=gpu, output_device=gpu[0])
                    # self.model.module.freeze_layers()
                else:
                    print("Only one GPU will be used")
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = EfficientNet.from_pretrained(
                        self.hparams['model']['pre_trained_model'],
                        num_classes=self.hparams['model']['n_classes'],
                    ).to(self.device)
                    # self.model.freeze_layers()
            else:
                self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                self.model = EfficientNet.from_pretrained(
                    self.hparams['model']['pre_trained_model'],
                    num_classes=self.hparams['model']['n_classes'],
                ).to(self.device)
                # self.model.freeze_layers()
                print('Only one GPU is available')

        if len(gpu) > 1:
            self.model.module.load_self_supervised_model(
                type_pretrain=self.hparams['model']['type_pretrain'],
                pre_trained_model=self.hparams['model']['pre_trained_model'],
                pre_trained_model_ssl=self.hparams['model']['pre_trained_model_ssl'],
                device=self.device,
            )
        else:
            self.model.load_self_supervised_model(
                type_pretrain=self.hparams['model']['type_pretrain'],
                pre_trained_model=self.hparams['model']['pre_trained_model'],
                pre_trained_model_ssl=self.hparams['model']['pre_trained_model_ssl'],
                device=self.device,
            )

        if self.hparams['model']['freeze']:
            if len(gpu)>1:
                self.model.module.freeze_layers()
            else:
                self.model.freeze_layers()

        print('Cuda available: ', torch.cuda.is_available())

        return True

    def __setup_model_hparams(self):

        # 1. define losses
        self.loss = nn.MSELoss()

        # 2. define model metric
        self.metric = Kappa()

        # 3. define optimizer
        self.optimizer = eval(f"torch.optim.{self.hparams['optimizer_name']}")(
            params=self.model.parameters(), **self.hparams['optimizer_hparams']
        )

        # 4. define scheduler
        self.scheduler = eval(f"torch.optim.lr_scheduler.{self.hparams['scheduler_name']}")(
            optimizer=self.optimizer, **self.hparams['scheduler_hparams']
        )

        # 5. define early stopping
        self.early_stopping = EarlyStopping(
            checkpoint_path=self.hparams['checkpoint_path'] + f'/checkpoint_{self.start_training}' + '.pt',
            patience=self.hparams['patience'],
            delta=self.hparams['min_delta'],
            is_maximize=True,
        )

        # 6. set gradient clipping
        self.apply_clipping = self.hparams['clipping']  # clipping of gradients

        # 7. Set scaler for optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        return True

    def __seed_everything(self, seed):
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
