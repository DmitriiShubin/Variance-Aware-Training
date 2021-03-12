import click
from utils.update_hparams import update_hparams
from experiments.baseline.train_pipeline import TrainPipeline
from utils.logger import Logger
from models.encoder_triplet import Model
import yaml
import os
from experiments.triplet_loss_encoder.data_generator import Dataset_train

# @click.command()
# @click.option('--batch_size', default=None, help='batch size')
# @click.option('--lr', default=None, help='learning rate')
# @click.option('--n_epochs', default=None, help='number of epoches to run')
# @click.option('--gpu', default='0', help='list of GPUs will be used for training')
# @click.option('--dropout', default=None, help='')
# @click.option('--hparams', default='./experiments/baseline/config_prostate_UB.yml', help='')
def run(
    batch_size=None,
    lr=None,
    n_epochs=None,
    gpu='2,3',
    dropout=None,
    experiment='./experiments/contrastive_learning_pretrain/config.yml',
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
    logger.kpi_logger.info(f"Dropout rate = {hparams['model']['dropout_rate']}")
    logger.kpi_logger.info(f"Model name: = {hparams['model_name']}")
    logger.kpi_logger.info('=============================================')
