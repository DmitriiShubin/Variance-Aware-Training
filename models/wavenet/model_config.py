import os

hparams = {}
# training params
hparams['n_epochs'] = 150
hparams['lr'] = 0.001
hparams['batch_size'] = 32
hparams['verbose_train'] = True

# early stopping settings
hparams['min_delta'] = 0  # thresold of improvement
hparams['patience'] = 30  # wait for n epoches for emprovement
hparams['n_fold'] = 5  # number of folds for cross-validation
hparams['verbose'] = True  # print score or not
hparams['start_fold'] = 1

# directories
hparams['model_path'] = './data/model_weights'
hparams['model_path'] += '/wavenet_model'
hparams['checkpoint_path'] = hparams['model_path'] + '/checkpoint'
hparams['model_name'] = '/wavenet'

for path in [hparams['model_path'], hparams['checkpoint_path']]:
    os.makedirs(path, exist_ok=True)

# dictionary of hyperparameters
structure_hparams = dict()
# global dropout rate
structure_hparams['dropout'] = 0.2
# number of filers for the models
structure_hparams['input_size'] = 4000


hparams['models'] = structure_hparams
