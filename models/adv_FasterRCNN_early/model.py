# basic libs
import numpy as np
from tqdm import tqdm
import os
import yaml
import random

# pytorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom modules
from metrics import AP
from utils.pytorchtools import EarlyStopping
from torch.nn.parallel import DataParallel as DP
from time import time
from utils.loss_functions import f1_loss

# model
from models.adv_FasterRCNN_early.structure import FasterRCNN
from utils.post_processing import Post_Processing


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
        self.postprocessing = Post_Processing()

        # define model parameters
        self.__setup_model_hparams()

        # declare preprocessing object
        self.__seed_everything(42)

    def fit(self, train, valid, pretrain):

        # setup train and val dataloaders
        train_loader = DataLoader(
            train,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_fn

        )
        valid_loader = DataLoader(
            valid,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=self.hparams['num_workers'],
            collate_fn=self.collate_fn
        )

        adv_loader = DataLoader(pretrain, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0)

        # tensorboard
        writer = SummaryWriter(f"runs/{self.hparams['model_name']}_{self.start_training}")

        print('Start training the model')
        for epoch in range(self.hparams['n_epochs']):

            # training mode
            self.model.train()
            avg_loss = 0.0
            avg_adv_loss = 0.0

            for X_batch, y_batch, X_batch_adv, y_batch_adv in tqdm(train_loader):

                X_batch_adv = torch.stack(X_batch_adv,dim=0)
                y_batch_adv = torch.stack(y_batch_adv, dim=0)
                sample = np.round(np.random.uniform(size=X_batch_adv.shape[0]), 2)
                X_batch_adv_train_val, _, _, _ = next(iter(adv_loader))
                X_batch_adv_train_val = X_batch_adv_train_val[: len(X_batch)]
                X_batch_adv[sample >= 0.5] = X_batch_adv_train_val[sample >= 0.5]
                y_batch_adv[sample >= 0.5] = 1
                y_batch_adv[sample < 0.5] = 0

                # push the data into the GPU
                X_batch = list(X.to(self.device) for X in X_batch)
                y_batch = [{k: v.to(self.device) for k, v in t.items()} for t in y_batch]
                X_batch_adv = X_batch_adv.float().to(self.device)
                y_batch_adv = y_batch_adv.float().to(self.device)

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                losses, pred_adv = self.model(x1=X_batch,x2=X_batch_adv,target=y_batch, train=True)

                # process main loss
                train_loss = (
                        self.hparams['model']['l_1'] * losses['loss_classifier']
                        + self.hparams['model']['l_2'] * losses['loss_box_reg']
                        + self.hparams['model']['l_3'] * losses['loss_objectness']
                        + self.hparams['model']['l_4'] * losses['loss_rpn_box_reg']
                )

                # process loss_2
                pred_adv = pred_adv.reshape(-1)
                y_batch_adv = y_batch_adv.reshape(-1)
                adv_loss = self.loss_adv(pred_adv, y_batch_adv)

                # calc loss
                avg_loss += train_loss.item() / len(train_loader)
                avg_adv_loss += adv_loss.item() / len(train_loader)

                train_loss = train_loss + self.hparams['model']['alpha'] * adv_loss

                # remove data from GPU
                losses = losses.float().cpu().detach().numpy()
                X_batch_adv = X_batch_adv.float().cpu().detach().numpy()
                y_batch_adv = y_batch_adv.cpu().detach().numpy()
                pred_adv = pred_adv.cpu().detach().numpy()
                X_batch = [X.cpu().detach().numpy() for X in X_batch]
                y_batch = [{k: v.cpu().detach().numpy() for k, v in t.items()} for t in y_batch]


                # gradient clipping
                if self.apply_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

                # backprop
                train_loss.backward()


            # evaluate the model
            print('Model evaluation')

            # val mode
            self.model.eval()
            self.optimizer.zero_grad()
            avg_val_loss = 0.0

            with torch.no_grad():

                for X_batch, y_batch,_,_ in tqdm(valid_loader):
                    # push the data into the GPU
                    X_batch = list(X.to(self.device) for X in X_batch)
                    y_batch = [{k: v.to(self.device) for k, v in t.items()} for t in y_batch]

                    # clean gradients from the previous step
                    self.optimizer.zero_grad()

                    # get model predictions
                    losses = self.model(X_batch, y_batch)

                    # process main loss
                    train_loss = (
                            self.hparams['model']['l_1'] * losses['loss_classifier']
                            + self.hparams['model']['l_2'] * losses['loss_box_reg']
                            + self.hparams['model']['l_3'] * losses['loss_objectness']
                            + self.hparams['model']['l_4'] * losses['loss_rpn_box_reg']
                    )
                    self.optimizer.zero_grad()

                    # calc loss
                    avg_val_loss += train_loss.item() / len(valid_loader)

                    # remove data from GPU
                    X_batch = [X.cpu().detach().numpy() for X in X_batch]
                    y_batch = [{k: v.cpu().detach().numpy() for k, v in t.items()} for t in y_batch]


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
                    '| Adv_loss: ',
                    avg_adv_loss,
                    '| Current LR: ',
                    self.__get_lr(self.optimizer),
                )

            # add data to tensorboard
            writer.add_scalars(
                'Loss', {'Train_loss': avg_loss, 'Val_loss': avg_val_loss}, epoch,
            )
            #writer.add_scalars('Metric', {'Metric_train': metric_train, 'Metric_val': metric_val}, epoch)

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

    def predict(self, X_test, objectness_threshold, nms_threshold):
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
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

        self.metric.reset_matric()

        print('Getting predictions')
        with torch.no_grad():
            for index, (X_batch, y_batch,_,_) in enumerate(tqdm(test_loader)):
                # push the data into the GPU
                X_batch = list(X.to(self.device) for X in X_batch)
                y_batch = [{k: v.to(self.device) for k, v in t.items()} for t in y_batch]

                # clean gradients from the previous step
                self.optimizer.zero_grad()

                # get model predictions
                preds = self.model(X_batch)

                # remove data from GPU
                X_batch = [X.cpu().detach().numpy() for X in X_batch]
                y_batch = [{k: v.cpu().detach().numpy() for k, v in t.items()} for t in y_batch]
                preds = [{k: v.cpu() for k, v in t.items()} for t in preds]

                bboxes, scores, classes = self.postprocessing.run(preds, objectness_threshold, nms_threshold)

                # calculate a step for metrics
                self.metric.calc_running_score(y_batch=y_batch, bboxes=bboxes, scores=scores, classes=classes)

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
            self.model = FasterRCNN(hparams=self.hparams['model'], device=self.device).to(self.device)
            self.model.build_adv_model(device=self.device)
        else:
            if torch.cuda.device_count() > 1:
                if len(gpu) > 1:
                    print("Number of GPUs will be used: ", len(gpu))
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = FasterRCNN(hparams=self.hparams['model'], device=self.device).to(
                        self.device
                    )
                    self.model.build_adv_model(device=self.device)
                    self.model = DP(self.model, device_ids=gpu, output_device=gpu[0])
                    # self.model.module.freeze_layers()
                else:
                    print("Only one GPU will be used")
                    self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                    self.model = FasterRCNN(hparams=self.hparams['model'], device=self.device).to(
                        self.device
                    )
                    self.model.build_adv_model(device=self.device)
            else:
                self.device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() else "cpu")
                self.model = FasterRCNN(hparams=self.hparams['model'], device=self.device).to(self.device)
                self.model.build_adv_model(device=self.device)
                print('Only one GPU is available')

        print('Cuda available: ', torch.cuda.is_available())

        return True

    def __setup_model_hparams(self):


        self.loss_adv = nn.BCELoss()

        # 2. define model metric
        self.metric = AP(n_classes=self.hparams['model']['n_classes'])  #

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
            is_maximize=False,
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

    def collate_fn(self, batch):
        return tuple(zip(*batch))