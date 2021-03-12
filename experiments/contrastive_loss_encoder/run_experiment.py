import click
from utils.update_hparams import update_hparams
from utils.logger import Logger
from models.encoder_triplet import Model
import yaml
import os
from experiments.contrastive_loss_encoder.data_generator import Dataset_train
from experiments.contrastive_loss_encoder.train_pipeline import TrainPipeline


def run(
    batch_size=None,
    lr=None,
    n_epochs=None,
    gpu='2,3,4,5',
    dropout=None,
    experiment='./experiments/contrastive_loss_encoder/config_brats.yml',
):

    # load hyperparameters
    hparams = yaml.load(open(experiment))

    # crate folders
    for f in [hparams['debug_path'], hparams['model_path'], hparams['checkpoint_path']]:
        os.makedirs(f, exist_ok=True)

    # process gpu selection string
    if gpu is not None:
        gpu = [int(i) for i in gpu.split(",")]

    hparams = update_hparams(
        hparams=hparams, dropout=dropout, batch_size=batch_size, lr=lr, n_epochs=n_epochs,
    )

    logger = Logger()

    # run cross-val
    cross_val = TrainPipeline(hparams=hparams, gpu=gpu, model=Model, Dataset_train=Dataset_train)
    loss_val, start_training = cross_val.train()

    # save logs
    logger.kpi_logger.info('=============================================')
    logger.kpi_logger.info(f'Datetime = {start_training}')
    logger.kpi_logger.info(f'Model loss, val = {loss_val}')
    logger.kpi_logger.info(f'Experiment = {experiment}')
    logger.kpi_logger.info(f"Batch size = {hparams['batch_size']}")
    logger.kpi_logger.info(f"Lr = {hparams['optimizer_hparams']['lr']}")
    logger.kpi_logger.info(f"N epochs = {hparams['n_epochs']}")
    logger.kpi_logger.info(f'GPU = {gpu}')
    logger.kpi_logger.info(f"Dropout rate = {hparams['model']['dropout_rate']}")
    logger.kpi_logger.info(f"Model name: = {hparams['model_name']}")
    logger.kpi_logger.info('=============================================')
