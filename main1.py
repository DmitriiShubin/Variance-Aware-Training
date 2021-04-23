import click

# baseline models
from experiments.segmentation.baseline.run_experiment import run as run_baseline_segmentation
from experiments.classification.baseline.run_experiment import run as run_baseline_classification





@click.command()
@click.option('--experiment', default=None, help='')
def main(experiment):


    run_baseline_classification(experiment='./experiments/classification/baseline/config_HMT_8.yml')

    return None


if __name__ == "__main__":
    main()
