import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import trainer

RESULTS_DIR = 'results'
CONFIG_DIR = 'config'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'

plt.style.use('seaborn-paper')

def create_barplot(ax, relevances, y_pred, x_lim=1.1, title='', x_label='',save_path='results/relevances.png', concept_names=None, **kwargs):
    """Creates a bar plot of relevances.

    Parameters
    ----------
    ax : pyplot axes object
        The axes on which the bar plot should be created.
    relevances: torch.tensor
        The relevances for which the bar plot should be generated. shape: (1, NUM_CONCEPTS, NUM_CLASSES)
    y_pred: torch.tensor (int)
        The prediction of the model for the corresponding relevances. shape: scalar value
    save_path: str
        Path to the location where the bar plot should be saved.
    """
    # Example data
    y_pred = y_pred.item()
    if len(relevances.squeeze().size()) == 2:
        relevances = relevances[:, y_pred]
    relevances = relevances.squeeze()
    if concept_names is None:
        concept_names = ['C. {}'.format(i + 1) for i in range(len(relevances))]
    else:
        concept_names = concept_names.copy()
    concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().cpu().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_title(title, fontsize=18)
    #ax.set_xlabel('Relevances (thetas)')
    #ax.set_title('Explanation for prediction: {}'.format(y_pred))
    #plt.tight_layout()
    #plt.savefig(save_path)
    #plt.clf()


def plot_lambda_accuracy(config_list, save_path=None, num_seeds=1, valid=False, **kwargs):
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
    valid : bool
        If true create plots based on saved validation accuracy (fast approach,
        only recommended if validation set was not used to tune hyper parameters).
        If false (default) the best model is loaded and evaluated on the test set (more runtime extensive).

    """
    assert type(num_seeds) is int and num_seeds > 0, "num_seeds must be an integer > 0 but is {}".format(num_seeds)
    lambdas = []
    accuracies = []

    path = Path(CONFIG_DIR)
    for config_file in config_list:
        seed_accuracies = []
        for seed in range(num_seeds):
            config_path = path/config_file if num_seeds == 1 else path/config_file[seed]
            # if test mode: instanciate trainer that evaluates model on the test set
            if not valid:
                t = trainer.init_trainer(config_path, best_model=True)
                seed_accuracies.append(t.test())
            with open(config_path, 'r') as f:
                config = json.load(f)
                # if validation mode: read top validation accuracy from csv file (a lot faster)
                if valid:
                    result_dir = Path(RESULTS_DIR)
                    results_csv = result_dir / config["exp_name"] / RESULTS_FILENAME
                    seed_accuracies.append(pd.read_csv(results_csv, header=0)['Accuracy'].max())
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
