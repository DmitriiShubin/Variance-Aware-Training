# ACDC experiments:

# baseline models
from experiments.segmentation.baseline.run_experiment import run as run_baseline_segmentation

# pre-training encoders
from experiments.segmentation.patch_encoder.run_experiment import run as run_pre_training_patch_segmentation
from experiments.segmentation.contrastive_loss_encoder.run_experiment import (
    run as run_pre_training_contrastive_segmentation,
)
from experiments.segmentation.rotation_encoder.run_experiment import (
    run as run_pre_training_rotation_segmentation,
)

# pre-trained models
from experiments.segmentation.pre_trained_patch.run_experiment import (
    run as run_pre_trained_patch_segmentation,
)
from experiments.segmentation.pre_trained_contrastive.run_experiment import (
    run as run_pre_trained_contrastive_segmentation,
)
from experiments.segmentation.pre_trained_rotation.run_experiment import (
    run as run_pre_trained_rotation_segmentation,
)

# adversarial models
from experiments.segmentation.adversarial_network_train_val_early.run_experiment import (
    run as run_adversarial_network_train_val_early,
)
from experiments.segmentation.adversarial_network_train_val_late.run_experiment import (
    run as run_adversarial_network_train_val_late,
)

################################################

# APTOS experiments:

# baseline models
from experiments.regression.baseline.run_experiment import run as run_efficientnet_baseline_regression

# pre-training models
from experiments.classification.patch_encoder.run_experiment import (
    run as run_pre_training_patch_classification,
)
from experiments.classification.rotation_encoder.run_experiment import (
    run as run_pre_training_rotation_classification,
)
from experiments.classification.contrastive_loss_encoder.run_experiment import (
    run as run_pre_training_contrastive_classification,
)

# adversarial models
from experiments.regression.adversarial_network_train_val_early.run_experiment import (
    run as run_efficientnet_adv_early_regression,
)
from experiments.regression.adversarial_network_train_val_late.run_experiment import (
    run as run_efficientnet_adv_late_regression,
)

# pre-trained models
from experiments.regression.pre_trained.run_experiment import run as run_pre_trained_regression

################################################

# HIST experiments:

# baseline models
from experiments.classification.baseline.run_experiment import (
    run as run_efficientnet_baseline_classification,
)

from experiments.classification.adversarial_network_train_val_early.run_experiment import (
    run as run_efficientnet_adv_early_classification,
)


import click


@click.command()
@click.option(
    '--experiment',
    default='./experiments/classification/adversarial_network_train_val_early/config_HIST_2_1.yml',
    help='',
)
@click.option('--gpu', default='7', help='')
def main(experiment, gpu):

    run_pre_training_patch_classification(
        experiment='./experiments/classification/patch_encoder/config_aptos.yml', gpu='6,7'
    )
    run_pre_training_contrastive_classification(
        experiment='./experiments/classification/contrastive_loss_encoder/config_aptos.yml', gpu='6,7'
    )
    run_pre_training_rotation_classification(
        experiment='./experiments/classification/rotation_encoder/config_aptos.yml', gpu='6,7'
    )

    run_pre_training_patch_classification(
        experiment='./experiments/classification/patch_encoder/config_HIST.yml', gpu='6,7'
    )
    run_pre_training_rotation_classification(
        experiment='./experiments/classification/rotation_encoder/config_HIST.yml', gpu='6,7'
    )
    run_pre_training_contrastive_classification(
        experiment='./experiments/classification/contrastive_loss_encoder/config_HIST.yml', gpu='6,7'
    )

    return None


if __name__ == "__main__":
    main()
