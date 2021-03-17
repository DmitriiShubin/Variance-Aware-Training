from experiments.baseline.run_experiment import run as run_baseline
from experiments.triplet_loss_encoder.run_experiment import run as run_triplet_pre_train
from experiments.rotation_encoder.run_experiment import run as run_rotation_pre_train
from experiments.contrastive_loss_encoder.run_experiment import run as run_contrastive_pre_train
from experiments.adversarial_network_train_val_early.run_experiment import (
    run as run_adversarial_network_train_val_early,
)
from experiments.adversarial_network_train_val_late.run_experiment import (
    run as run_adversarial_network_train_val_late,
)
from experiments.rotation_encoder.run_experiment import (
    run as run_adversarial_network_train_val_late,
)
from experiments.pre_trained_triplet.run_experiment import run as run_pre_trained_triplet


def main():

    # BRATS

    # pre-training
    # Self-supervised contrastive
    #run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_brats.yml',gpu='0')

    # Self-supervised triplet
    #run_triplet_pre_train(experiment='./experiments/triplet_loss_encoder/config_brats.yml')

    # Self-supervised rotation
    # run_rotation_pre_train(experiment='./experiments/rotation_encoder/config_brats.yml')

    # baseline, without pre-train
    run_baseline(experiment='./experiments/baseline/config_brats_2.yml')
    # run_baseline(experiment='./experiments/baseline/config_brats_4.yml')
    # run_baseline(experiment='./experiments/baseline/config_brats_8.yml')

    # pre-train contrastive
    # 2
    # 4
    # 8
    # UB

    # pre-train triplet
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

    # Single-stage self-supervised early log
    # 2
    # 4
    # 8

    # Single-stage self-supervised late flat
    # 2
    # 4
    # 8

    # Single-stage self-supervised early flat
    # 2
    # 4
    # 8

    return None


if __name__ == "__main__":
    main()
