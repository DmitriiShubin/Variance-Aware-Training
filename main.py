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


def main():

    # BRATS

    # pre-training
    # Self-supervised contrastive
    # run_contrastive_pre_train(experiment='./experiments/contrastive_loss_encoder/config_brats.yml',n_epochs=1)

    # Self-supervised triplet
    # run_triplet_pre_train(experiment='./experiments/triplet_loss_encoder/config_brats.yml',n_epochs=1)

    # Self-supervised rotation
    run_rotation_pre_train(experiment='./experiments/rotation_encoder/config_brats.yml', n_epochs=1)

    # baseline, without pre-train
    run_baseline(experiment='./experiments/baseline/config_brats_2.yml', n_epochs=1)
    run_baseline(experiment='./experiments/baseline/config_brats_4.yml', n_epochs=1)
    run_baseline(experiment='./experiments/baseline/config_brats_8.yml', n_epochs=1)

    # pre-train contrastive
    # 2
    # 4
    # 8
    # UB

    # pre-train triplet
    # 2
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
