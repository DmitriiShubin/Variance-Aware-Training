import os

hparams = {}
# training params
hparams['n_epochs'] = 2
hparams['lr'] = 1e-5
hparams['batch_size'] = 2
hparams['verbose_train'] = True

# early stopping settings
hparams['min_delta'] = 0.001  # thresold of improvement
hparams['patience'] = 10  # wait for n epoches for emprovement
hparams['n_fold'] = 5  # number of folds for cross-validation
hparams['verbose'] = True  # print score or not
hparams['start_fold'] = 1
hparams['model_name']='ResNet50'

# directories
hparams['model_path'] = './data/model_weights'
hparams['model_path'] += '/ResNet50'
hparams['checkpoint_path'] = hparams['model_path'] + '/checkpoint'

for path in [hparams['model_path'], hparams['checkpoint_path']]:
    os.makedirs(path, exist_ok=True)

# dictionary of hyperparameters
structure_hparams = dict()
# global dropout rate
structure_hparams['alpha'] = 0.1

hparams['model'] = structure_hparams
