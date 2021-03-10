import os
import yaml

hparams = yaml.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/hparams.yml"))


assert hparams['model']['kernel_size'] // 2 != 0, "Kernel size should be odd"


# create folders for the model
folders = []
folders += [
    hparams['model_path'],
    hparams['checkpoint_path'],
]

for f in folders:
    os.makedirs(f, exist_ok=True)
