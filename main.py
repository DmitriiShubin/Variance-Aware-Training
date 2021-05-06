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


# APTOS experiments:

# baseline models
from experiments.classification.baseline.run_experiment import run as run_efficientnet_baseline

# pre-training models
from experiments.classification.contrastive_loss_encoder.run_experiment import (
    run as run_pre_training_contrastive_classification,
)
from experiments.classification.patch_encoder.run_experiment import (
    run as run_pre_training_patch_classification,
)
from experiments.classification.rotation_encoder.run_experiment import (
    run as run_pre_training_rotation_classification,
)

# adversarial models
from experiments.classification.adversarial_network_train_val_early.run_experiment import (
    run as run_efficientnet_adv_early,
)


# RSNA experiments:

# baseline models
from experiments.detection.baseline.run_experiment import run as run_yolov4_baseline

import click


@click.command()
@click.option('--experiment', default='./experiments/detection/baseline/config_RSNA_2.yml', help='')
@click.option('--gpu', default='5', help='')
def main(experiment, gpu):

    # ACDC

    # baseline, without pre-train
    # run_baseline_segmentation(experiment='./experiments/baseline/config_ACDC_2.yml')
    # run_baseline_segmentation(experiment='./experiments/baseline/config_ACDC_4.yml')
    # run_baseline_segmentation(experiment='./experiments/baseline/config_ACDC_8.yml')
    # run_baseline_segmentation(experiment='./experiments/baseline/config_ACDC_UB.yml')

    # pre-training
    # run_contrastive_pre_train_segmentation(experiment='./experiments/contrastive_loss_encoder/config_ACDC.yml',gpu='7')
    # run_rotation_pre_train_segmentation(experiment='./experiments/rotation_encoder/config_ACDC.yml',gpu='7')
    # run_patch_pre_train_segmentation(experiment='./experiments/patch_encoder/config_ACDC.yml',gpu='6')

    # # pre-trained contrastive
    # run_pre_trained_contrastive_segmentation(experiment='./experiments/pre_trained_contrastive/config_ACDC_2.yml')
    # run_pre_trained_contrastive_segmentation(experiment='./experiments/pre_trained_contrastive/config_ACDC_4.yml')
    # run_pre_trained_contrastive_segmentation(experiment='./experiments/pre_trained_contrastive/config_ACDC_8.yml')

    # # pre-trained rotation
    # run_pre_trained_rotation_segmentation(experiment='./experiments/pre_trained_rotation/config_ACDC_2.yml')
    # run_pre_trained_rotation_segmentation(experiment='./experiments/pre_trained_rotation/config_ACDC_4.yml')
    # run_pre_trained_rotation_segmentation(experiment='./experiments/pre_trained_rotation/config_ACDC_8.yml')

    # # pre-trained patch
    # run_pre_trained_patch_segmentation(experiment='./experiments/pre_trained_patch/config_ACDC_2.yml')
    # run_pre_trained_patch_segmentation(experiment='./experiments/pre_trained_patch/config_ACDC_4.yml')
    # run_pre_trained_patch_segmentation(experiment='./experiments/pre_trained_patch/config_ACDC_8.yml')

    # Single-stage self-supervised early flat
    # run_adversarial_network_train_val_early(
    #     experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_2.yml'
    # )
    # run_adversarial_network_train_val_early(experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_4.yml')
    # run_adversarial_network_train_val_early(experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_8.yml')

    # Single-stage self-supervised late flat
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_2.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_4.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_8.yml')

    ###########################################################################
    # APTOS

    # baseline models
    # run_efficientnet_baseline(experiment='./experiments/classification/baseline/config_aptos_2.yml', gpu='7')
    # run_efficientnet_baseline(experiment='./experiments/classification/baseline/config_aptos_4.yml',gpu='7')
    # run_efficientnet_baseline(experiment='./experiments/classification/baseline/config_aptos_8.yml', gpu='7')

    # adversarial models early
    # for i in range(1,11):
    #     run_efficientnet_adv_early(experiment=f'./experiments/classification/adversarial_network_train_val_early/config_aptos_2_{i}.yml', gpu='6,7')
    # for i in range(1,11):
    #     run_efficientnet_adv_early(experiment=f'./experiments/classification/adversarial_network_train_val_early/config_aptos_4_{i}.yml', gpu='6,7')
    # for i in range(1,11):
    #     run_efficientnet_adv_early(experiment=f'./experiments/classification/adversarial_network_train_val_early/config_aptos_8_{i}.yml', gpu='6,7')
    #

    # pre-training models
    # run_pre_training_contrastive_classification(experiment=f'./experiments/classification/contrastive_loss_encoder/config_aptos.yml', gpu='6,7')
    # run_pre_training_patch_classification(experiment=f'./experiments/classification/patch_encoder/config_aptos.yml', gpu='6,7')
    # run_pre_training_rotation_classification(experiment=f'./experiments/classification/rotation_encoder/config_aptos.yml', gpu='6,7')
    ###########################################################################
    # RSNA

    # run_yolov4_baseline(experiment='./experiments/detection/baseline/config_RSNA_2.yml', gpu='0')

    run_yolov4_baseline(experiment=experiment, gpu=gpu)

    return None


if __name__ == "__main__":
    main()
