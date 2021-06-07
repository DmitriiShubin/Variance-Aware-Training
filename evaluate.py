# ACDC experiments:

# baseline models
from experiments.segmentation.baseline.run_experiment import run as run_baseline_segmentation

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
    run as run_adv_early_segmentation,
)
from experiments.segmentation.adversarial_network_train_val_late.run_experiment import (
    run as run_adv_late_segmentation,
)

################################################

# HIST experiments:

# baseline models
from experiments.classification.baseline.run_experiment import run as run_baseline_classification

# pre-trained models
from experiments.classification.pre_trained.run_experiment import run as run_pre_trained_classification

# adversarial models
from experiments.classification.adversarial_network_train_val_early.run_experiment import (
    run as run_adv_early_classification,
)
from experiments.classification.adversarial_network_train_val_late.run_experiment import (
    run as run_adv_late_classification,
)

################################################

# APTOS experiments:

# baseline models
from experiments.regression.baseline.run_experiment import run as run_baseline_regression

# pre-trained models
from experiments.regression.pre_trained.run_experiment import run as run_pre_trained_regression

# adversarial models
from experiments.regression.adversarial_network_train_val_early.run_experiment import (
    run as run_adv_early_regression,
)
from experiments.regression.adversarial_network_train_val_late.run_experiment import (
    run as run_adv_late_regression,
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


    # ACDC

    # baseline, without pre-train
    run_baseline_segmentation(experiment='./experiments/segmentation/baseline/config_ACDC_2_eval.yml', eval=True)
    run_baseline_segmentation(experiment='./experiments/segmentation/baseline/config_ACDC_4_eval.yml', eval=True)
    run_baseline_segmentation(experiment='./experiments/segmentation/baseline/config_ACDC_8_eval.yml', eval=True)
    run_baseline_segmentation(experiment='./experiments/segmentation/baseline/config_ACDC_UB_eval.yml', eval=True)

    # pre-trained contrastive
    run_pre_trained_contrastive_segmentation(
        experiment='./experiments/segmentation/pre_trained_contrastive/config_ACDC_2_eval.yml'
        , eval=True)
    run_pre_trained_contrastive_segmentation(
        experiment='./experiments/segmentation/pre_trained_contrastive/config_ACDC_4_eval.yml'
        , eval=True)
    run_pre_trained_contrastive_segmentation(
        experiment='./experiments/segmentation/pre_trained_contrastive/config_ACDC_8_eval.yml'
        , eval=True)

    # pre-trained rotation
    run_pre_trained_rotation_segmentation(
        experiment='./experiments/segmentation/pre_trained_rotation/config_ACDC_2_eval.yml'
        , eval=True)
    run_pre_trained_rotation_segmentation(
        experiment='./experiments/segmentation/pre_trained_rotation/config_ACDC_4_eval.yml'
        , eval=True)
    run_pre_trained_rotation_segmentation(
        experiment='./experiments/segmentation/pre_trained_rotation/config_ACDC_8_eval.yml'
        , eval=True)

    # pre-trained patch
    run_pre_trained_patch_segmentation(
        experiment='./experiments/segmentation/pre_trained_patch/config_ACDC_2_eval.yml'
        , eval=True)
    run_pre_trained_patch_segmentation(
        experiment='./experiments/segmentation/pre_trained_patch/config_ACDC_4_eval.yml'
        , eval=True)
    run_pre_trained_patch_segmentation(
        experiment='./experiments/segmentation/pre_trained_patch/config_ACDC_8_eval.yml'
        , eval=True)

    # variance-aware, ealy agg
    run_adv_early_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_2_eval.yml'
        , eval=True)
    run_adv_early_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_4_eval.yml'
        , eval=True)
    run_adv_early_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_early/config_ACDC_8_eval.yml'
        , eval=True)

    # variance-aware, late agg
    run_adv_late_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_late/config_ACDC_2_eval.yml'
        , eval=True)
    run_adv_late_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_late/config_ACDC_4_eval.yml'
        , eval=True)
    run_adv_late_segmentation(
        experiment='./experiments/segmentation/adversarial_network_train_val_late/config_ACDC_8_eval.yml'
        , eval=True)

    ###########################################################################

    # PCam (classification, eval=True) experiments

    # baseline, without pre-train
    run_baseline_classification(experiment='./experiments/classification/baseline/config_HIST_2_eval.yml', eval=True)
    run_baseline_classification(experiment='./experiments/classification/baseline/config_HIST_4_eval.yml', eval=True)
    run_baseline_classification(experiment='./experiments/classification/baseline/config_HIST_8_eval.yml', eval=True)
    run_baseline_classification(experiment='./experiments/classification/baseline/config_HIST_UB_eval.yml', eval=True)

    # pre-trained contrastive (SimCLR, eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_2_contrastive_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_4_contrastive_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_8_contrastive_eval.yml'
        , eval=True)

    # pre-trained rotation
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_2_rotation_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_4_rotation_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_8_rotation_eval.yml'
        , eval=True)

    # pre-trained patch (CP, eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_2_patch_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_4_patch_eval.yml'
        , eval=True)
    run_pre_trained_classification(
        experiment='./experiments/classification/pre_trained/config_HIST_8_patch_eval.yml'
        , eval=True)

    # variance-aware, ealy agg
    run_adv_early_classification(
        experiment='./experiments/classification/adversarial_network_train_val_early/config_HIST_2_eval.yml'
        , eval=True)
    run_adv_early_classification(
        experiment='./experiments/classification/adversarial_network_train_val_early/config_HIST_4_eval.yml'
        , eval=True)
    run_adv_early_classification(
        experiment='./experiments/classification/adversarial_network_train_val_early/config_HIST_8_eval.yml'
        , eval=True)

    # variance-aware, late agg
    run_adv_late_classification(
        experiment='./experiments/classification/adversarial_network_train_val_late/config_HIST_2_eval.yml'
        , eval=True)
    run_adv_late_classification(
        experiment='./experiments/classification/adversarial_network_train_val_late/config_HIST_4_eval.yml'
        , eval=True)
    run_adv_late_classification(
        experiment='./experiments/classification/adversarial_network_train_val_late/config_HIST_8_eval.yml'
        , eval=True)

    ###########################################################################
    # APTOS

    # baseline, without pre-train
    run_baseline_regression(experiment='./experiments/regression/baseline/config_aptos_2_eval.yml', eval=True)
    run_baseline_regression(experiment='./experiments/regression/baseline/config_aptos_4_eval.yml', eval=True)
    run_baseline_regression(experiment='./experiments/regression/baseline/config_aptos_8_eval.yml', eval=True)
    run_baseline_regression(experiment='./experiments/regression/baseline/config_aptos_UB_eval.yml', eval=True)

    # pre-trained contrastive (SimCLR, eval=True)
    run_pre_trained_regression(
        experiment='./experiments/regression/pre_trained/config_aptos_2_contrastive_eval.yml'
        , eval=True)
    run_pre_trained_regression(
        experiment='./experiments/regression/pre_trained/config_aptos_4_contrastive_eval.yml'
        , eval=True)
    run_pre_trained_regression(
        experiment='./experiments/regression/pre_trained/config_aptos_8_contrastive_eval.yml'
        , eval=True)

    # pre-trained rotation
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_2_rotation_eval.yml',
                               eval=True)
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_4_rotation_eval.yml',
                               eval=True)
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_8_rotation_eval.yml',
                               eval=True)

    # pre-trained patch (CP, eval=True)
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_2_patch_eval.yml',
                               eval=True)
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_4_patch_eval.yml',
                               eval=True)
    run_pre_trained_regression(experiment='./experiments/regression/pre_trained/config_aptos_8_patch_eval.yml',
                               eval=True)

    # variance-aware, ealy agg
    run_adv_early_regression(
        experiment='./experiments/regression/adversarial_network_train_val_early/config_aptos_2_eval.yml'
        , eval=True)
    run_adv_early_regression(
        experiment='./experiments/regression/adversarial_network_train_val_early/config_aptos_4_eval.yml'
        , eval=True)
    run_adv_early_regression(
        experiment='./experiments/regression/adversarial_network_train_val_early/config_aptos_8_eval.yml'
        , eval=True)

    # variance-aware, late agg
    run_adv_late_regression(
        experiment='./experiments/regression/adversarial_network_train_val_late/config_aptos_2_eval.yml'
        , eval=True)
    run_adv_late_regression(
        experiment='./experiments/regression/adversarial_network_train_val_late/config_aptos_4_eval.yml'
        , eval=True)
    run_adv_late_regression(
        experiment='./experiments/regression/adversarial_network_train_val_late/config_aptos_8_eval.yml'
        , eval=True)

    return None


if __name__ == "__main__":
    main()
