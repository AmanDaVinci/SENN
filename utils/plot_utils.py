import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trainer import init_trainer

RESULTS_DIR = 'results'
CONFIG_DIR = 'config'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'


def create_barplot(relevances, y_pred, save_path='results/relevances.png', concept_names=None):
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
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    y_pred = y_pred.item()
    relevances = relevances[0, :, y_pred].squeeze()
    if concept_names is None:
        concept_names = ['Concept {}'.format(i + 1) for i in range(len(relevances))]
        concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().cpu().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('Relevances (thetas)')
    ax.set_title('Explanation for prediction: {}'.format(y_pred))

    plt.savefig(save_path)
    plt.clf()


def plot_lambda_accuracy(config_list, save_path=None):
    """Plots the lambda (robustness regularizer) vs accuracy of SENN

    Parameters
    ----------
    config_list: list
        List of experiment config files used to vary the lambda 
    save_path: str
        Path to the location where the plot should be saved.
    """
    lambdas = []
    accuracies = []

    path = Path(CONFIG_DIR)
    for config_file in config_list:
        trainer = init_trainer(path/config_file, best_model=True)
        accuracies.append(trainer.test())
        with open(path / config_file, 'r') as f:
            config = json.load(f)
            lambdas.append(config["robust_reg"])

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(lambdas, accuracies, "r.-")
    ax.set_xlabel('Robustness Regularization Strength')
    ax.set_ylabel('Prediction Accuracy')
    
    if save_path is not None:
        plt.savefig(save_path)

    return fig
