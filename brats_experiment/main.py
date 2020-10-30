import logging
import click
import time

# import modules
from cv_pipeline import CVPipeline

from config import hparams

@click.command()
@click.option('--start_fold', default=hparams['start_fold'], help='fold to train')
@click.option('--alpha', default=hparams['model']['alpha'], help='fold to train')
@click.option('--batch_size', default=hparams['batch_size'], help='batch size')
@click.option('--lr', default=hparams['lr'], help='learning rate')
@click.option('--n_epochs', default=hparams['n_epochs'], help='number of epoches to run')
@click.option('--p_proc', default=False, help='does it need to run preprocessing?')
@click.option('--train', default=True, help='does it need to train the model?')
@click.option('--gpu', default='0,1,2', help='list of GPUs will be used for training')
def main(start_fold, batch_size, lr, n_epochs, p_proc, train, gpu,alpha):

    # update hparams

    gpu = [int(i) for i in gpu.split(",")]

    hparams['lr'] = lr
    hparams['batch_size'] = batch_size
    hparams['start_fold'] = int(start_fold)
    hparams['n_epochs'] = n_epochs
    hparams['model']['alpha'] = alpha

    if train:
        cross_val = CVPipeline(hparams=hparams, gpu=gpu)

        score = cross_val.train()

        logger = logging.getLogger('Training pipeline')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('training.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info('=============================================')
        logger.info(f'Datetime = {time.time()}')
        logger.info(f'Model metric = {score}')
        logger.info(f'Model fold = {start_fold}')
        logger.info(f'Train = {train}')
        logger.info(f'Preproc = {p_proc}')
        logger.info(f'Model fold = {batch_size}')
        logger.info(f'Model fold = {lr}')
        logger.info(f'Model fold = {n_epochs}')
        logger.info(f'GPU = {gpu}')
        logger.info(f"Alpha = {hparams['model']['alpha']}")
        logger.info(f"Model name: = {hparams['model_name']}")
        logger.info('=============================================')


if __name__ == "__main__":
    main()
