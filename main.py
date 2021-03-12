from experiments.baseline.run_experiment import run as run_baseline
from experiments.triplet_loss_encoder.run_experiment import run as run_triplet_pre_train


def main():

    # baseline upper-boundaries
    run_triplet_pre_train()

    # baseline (without pre-trianing)

    # Self-supervised #1

    # Self-supervised #2

    # Self-supervised #3

    # Single-stage self-supervised

    return None


if __name__ == "__main__":
    main()
