from experiments.baseline.run_experiment import run as run_baseline
from experiments.triplet_loss_encoder.run_experiment import run as run_triplet_pre_train
from experiments.rotation_encoder.run_experiment import run as run_rotation_pre_train
from experiments.contrastive_loss_encoder.run_experiment import run as run_contrastive_pre_train
from experiments.adversarial_network_patientwise_early.run_experiment import (
    run as run_adversarial_network_patientwise_early,
)
from experiments.adversarial_network_patientwise_late.run_experiment import (
    run as run_adversarial_network_patientwise_late,
)
from experiments.adversarial_network_train_val_early.run_experiment import (
    run as run_adversarial_network_train_val_early,
)
from experiments.adversarial_network_train_val_late.run_experiment import (
    run as run_adversarial_network_train_val_late,
)


def main():

    # baseline upper-boundaries
    run_adversarial_network_train_val_late()

    # baseline (without pre-trianing)

    # Self-supervised #1

    # Self-supervised #2

    # Self-supervised #3

    # Single-stage self-supervised

    return None


if __name__ == "__main__":
    main()
