import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
CONFIG_DIR = 'config'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'

plt.style.use('seaborn-paper')

def create_barplot(relevances, y_pred, save_path='results/relevances.png', concept_names=None, **kwargs):
    """Creates a bar plot of relevances.

    Parameters
    ----------
    relevances: torch.tensor
       The relevances for which the bar plot should be generated. shape: (1, NUM_CONCEPTS, NUM_CLASSES)
    y_pred: torch.tensor (int)
       The prediction of the model for the corresponding relevances. shape: scalar value
    save_path: str
        Path to the location where the bar plot should be saved.
    """
    fig, ax = plt.subplots()

    # Example data
    y_pred = y_pred.item()
    relevances = relevances[0, :, y_pred].squeeze()
    if concept_names is None:
        concept_names = ['Concept {}'.format(i + 1) for i in range(len(relevances))]
    else:
        concept_names = concept_names.copy()
    concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().cpu().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel('Relevances (thetas)')
    ax.set_title('Explanation for prediction: {}'.format(y_pred))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def plot_lambda_accuracy(config_list, save_path, num_seeds=1):
    """Plots the lambda (robustness regularizer) vs accuracy of SENN

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
            with open(path/config_file, 'r') as f:
                config = json.load(f)
                lambdas.append(config["robust_reg"])
                result_dir = Path(RESULTS_DIR)
                results_csv = result_dir / config["exp_name"] / RESULTS_FILENAME
                dataset = config['dataloader']
            seed_accuracies.append(pd.read_csv(results_csv, header=0)['Accuracy'].max())
        accuracies.append(sum(seed_accuracies)/num_seeds)

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(lambdas)), accuracies, "r.-")
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels(lambdas)
    ax.set_xlabel('Robustness Regularization Strength')
    ax.set_ylabel('Prediction Accuracy')
    
    plt.savefig(save_path)
    plt.clf()
