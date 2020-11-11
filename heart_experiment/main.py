import logging
import click
import time

# import modules
from cv_pipeline import CVPipeline


@click.command()
@click.option('--start_fold', default=None, help='fold to train')
@click.option('--alpha', default=None, help='fold to train')
@click.option('--batch_size', default=None, help='batch size')
@click.option('--lr', default=None, help='learning rate')
@click.option('--n_epochs', default=None, help='number of epoches to run')
@click.option('--gpu', default='0,1,2', help='list of GPUs will be used for training')
@click.option(
    '--model', default='linknet', help='Model type, one of following: unet, adv_unet, fpn, adv_fpn'
)
def main(start_fold, alpha, batch_size, lr, n_epochs, gpu, model):

    # check model type input
    assert (
        model == 'unet' or model == 'adv_unet' or model == 'linknet' or model == 'adv_linknet'
    ), 'The following set of models is supported: unet, adv_unet, linknet, adv_linknet'

    if model == 'unet':
        from models.linknet import Model, hparams
    elif model == 'adv_unet':
        from models.adv_unet import Model, hparams
    elif model == 'linknet':
        from models.linknet import Model, hparams
    elif model == 'adv_linknet':
        from models.adv_linknet import Model, hparams

    # update hparams
    gpu = [int(i) for i in gpu.split(",")]

    if start_fold is not None:
        hparams['start_fold'] = int(start_fold)

    if alpha is not None:
        hparams['model']['alpha'] = float(alpha)

    if batch_size is not None:
        hparams['batch_size'] = int(batch_size)

    if lr is not None:
        hparams['lr'] = float(lr)

    if n_epochs is not None:
        hparams['n_epochs'] = int(n_epochs)

    print(f'Selected type of the model: {model}')

    cross_val = CVPipeline(hparams=hparams, gpu=gpu, model=Model)

    score, start_training = cross_val.train()

    logger = logging.getLogger('Training pipeline')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('training.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info('=============================================')
    logger.info(f'Datetime = {start_training}')
    logger.info(f'Model metric = {score}')
    logger.info(f"Model fold = {hparams['start_fold']}")
    logger.info(f"Batch size = {hparams['batch_size']}")
    logger.info(f"Lr = {hparams['lr']}")
    logger.info(f"N epochs = {hparams['n_epochs']}")
    logger.info(f'GPU = {gpu}')
    logger.info(f"Alpha = {hparams['model']['alpha']}")
    logger.info(f"Model name: = {hparams['model_name']}")
    logger.info('=============================================')


if __name__ == "__main__":
    main()


