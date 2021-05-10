from utils.update_hparams import update_hparams
from utils.logger import Logger

import yaml
import os

from experiments.detection.adversarial_network_train_val_late.data_generator import Dataset_train
from experiments.detection.adversarial_network_train_val_late.train_pipeline import TrainPipeline
from models.adv_FasterRCNN_late import Model


def run(
    batch_size=None,
    lr=None,
    n_epochs=None,
    gpu='1',
    dropout=None,
    experiment='./experiments/adversarial_network_train_val_early/config_brats_2.yml',
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
    fold_scores_val, fold_scores_test, start_training = cross_val.train()

    # save logs
    logger.kpi_logger.info('=============================================')
    logger.kpi_logger.info(f'Datetime = {start_training}')
    logger.kpi_logger.info(f'Model metric, val = {fold_scores_val}')
    logger.kpi_logger.info(f'Model metric, test = {fold_scores_test}')
    logger.kpi_logger.info(f'Experiment = {experiment}')
    logger.kpi_logger.info(f"Batch size = {hparams['batch_size']}")
    logger.kpi_logger.info(f"Lr = {hparams['optimizer_hparams']['lr']}")
    logger.kpi_logger.info(f"N epochs = {hparams['n_epochs']}")
    logger.kpi_logger.info(f'GPU = {gpu}')
    logger.kpi_logger.info(f"Model name: = {hparams['model_name']}")
    logger.kpi_logger.info('=============================================')


if __name__ == "__main__":
    run()
