from experiments.baseline.run_experiment import run


def main():

    # baseline upper-boundaries
    run(hparams='./experiments/baseline/config_prostate_UB.yml')

    # baseline (without pre-trianing)

    # Self-supervised #1

    # Self-supervised #2

    # Self-supervised #3

    # Single-stage self-supervised

    return None


if __name__ == "__main__":
    main()
