import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from trainer import init_trainer

RESULTS_DIR = 'results'
CONFIG_DIR = 'config'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'

plt.style.use('seaborn-paper')

def plot_lambda_accuracy(config_list, save_path=None, num_seeds=1, **kwargs):
    """Plots the lambda (robustness regularizer) vs accuracy of SENN#

    Parameters
    ----------
    config_list: list
        List of experiment config files used to vary the lambda.
        If multiple seeds are used then this is a list of lists where the inner lists have a length
        equal to the number of different seeds used and contain the corresponding config files.
    save_path: str
        Path to the location where the plot should be saved.
    num_seeds : int
        The number of different seeds that are used.
    """
    assert type(num_seeds) is int and num_seeds > 0, "num_seeds must be an integer > 0 but is {}".format(num_seeds)
    lambdas = []
    accuracies = []

    path = Path(CONFIG_DIR)
    for config_file in config_list:
        seed_accuracies = []
        for seed in range(num_seeds):
            config_path = path/config_file if num_seeds == 1 else path/config_file[seed]
            trainer = init_trainer(config_path, best_model=True)
            seed_accuracies.append(trainer.test())
            with open(config_path, 'r') as f:
                config = json.load(f)
        print(seed_accuracies)
        lambdas.append(config["robust_reg"])
        accuracies.append(sum(seed_accuracies)/num_seeds)

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(lambdas)), accuracies, "r.-")
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels(lambdas)
    ax.set_xlabel('Robustness Regularization Strength')
    ax.set_ylabel('Prediction Accuracy')

    if save_path is not None:
        plt.savefig(save_path)

    return fig