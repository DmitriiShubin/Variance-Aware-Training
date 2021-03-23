from experiments.baseline.run_experiment import run as run_baseline
from experiments.triplet_loss_encoder.run_experiment import run as run_triplet_pre_train
from experiments.rotation_encoder.run_experiment import run as run_rotation_pre_train
from experiments.contrastive_loss_encoder.run_experiment import run as run_contrastive_pre_train
from experiments.adversarial_network_train_val_early.run_experiment import (
    run as run_adversarial_network_train_val_early,
)
from experiments.adversarial_network_train_val_late.run_experiment import (
    run as run_adversarial_network_train_val_late
)
from experiments.pre_trained_triplet.run_experiment import run as run_pre_trained_triplet
from experiments.pre_trained_contrastive.run_experiment import run as run_pre_trained_contrastive
import os


def main():

    # BRATS

    # pre-training
    # Self-supervised contrastive
    run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_ACDC.yml',gpu='7')
    # Self-supervised triplet
    run_triplet_pre_train(experiment='./experiments/triplet_loss_encoder/config_ACDC.yml',gpu='7')

    # Self-supervised rotation
    run_rotation_pre_train(experiment='./experiments/rotation_encoder/config_ACDC.yml',gpu='7')

    # baseline, without pre-train
    #run_baseline(experiment='./experiments/baseline/config_ACDC_2.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_4.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_8.yml')
    #run_baseline(experiment='./experiments/baseline/config_ACDC_UB.yml')

    # pre-trained contrastive
    #run_pre_trained_contrastive(experiment='./experiments/pre_trained_contrastive/config_ACDC_2.yml')
    # 4
    # 8
    # UB

    # pre-trained triplet
    #run_pre_trained_triplet(experiment='./experiments/pre_trained_triplet/config_brats_2.yml')
    # 4
    # 8
    # UB

    # pre-train rotation
    # 2
    # 4
    # 8
    # UB

    # pre-train puzzle
    # 2
    # 4
    # 8
    # UB

    # TODO: select alpha grid
    # Single-stage self-supervised late log
    # 2
    # 4
    # 8

    # Single-stage self-supervised early flat
    #run_adversarial_network_train_val_early(experiment='./experiments/adversarial_network_train_val_early/config_ACDC_2.yml')
    #run_adversarial_network_train_val_early(experiment='./experiments/adversarial_network_train_val_early/config_ACDC_4.yml')
    #run_adversarial_network_train_val_early(experiment='./experiments/adversarial_network_train_val_early/config_ACDC_8.yml')



    # Single-stage self-supervised late flat
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_2.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_4.yml')
    # run_adversarial_network_train_val_late(experiment='./experiments/adversarial_network_train_val_late/config_ACDC_8.yml')


    return None


if __name__ == "__main__":
    main()
