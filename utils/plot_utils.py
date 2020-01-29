import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import trainer

RESULTS_DIR = 'results'
CONFIG_DIR = 'config'
RESULTS_FILENAME = 'accuracies_losses_valid.csv'

plt.style.use('seaborn-paper')

def get_comparison_plot(images, model):
    """Creates a plot that shows similar prototypes with their relevance scores and concept values.

    Parameters
    ----------
    images: torch.Tensor
       An array with the images to be compared
    model: models.senn
       A senn model to be used for the visualizations

    Returns
    ----------
    fig: matplotlib.pyplot
        The figure that contains the plots
    """

    def get_colors(values):
        colors = ['b' if v > 0 else 'r' for v in values]
        colors.reverse()
        return colors

    model.eval()
    with torch.no_grad():
        y_pred, (concepts, relevances), _ = model(images)
    y_pred = y_pred.argmax(1)

    fig, axes = plt.subplots(nrows=3, ncols=len(images))

    PROTOTYPE_ROW = 0
    RELEVANCE_ROW = 1
    CONCEPT_ROW = 2

    concepts_min = concepts.min().item()
    concepts_max = concepts.max().item()
    concept_lim = -concepts_min if -concepts_min > concepts_max else concepts_max

    for i in range(len(images)):
        prediction_index = y_pred[i].item()
        concept_names = [f'C{i+1}' for i in range(concepts.shape[1] - 1, -1, -1)]

        # plot the input image
        axes[PROTOTYPE_ROW, i].imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray')
        axes[PROTOTYPE_ROW, i].set_title(f"Prediction: {prediction_index}")
        axes[PROTOTYPE_ROW, i].axis('off')

        # plot the relevance scores
        rs = relevances[i, :, prediction_index]
        colors_r = get_colors(rs)
        axes[RELEVANCE_ROW, i].barh(np.arange(len(rs)),
                                    np.flip(rs.detach().numpy()),
                                    align='center', color=colors_r)

        axes[RELEVANCE_ROW, i].set_yticks(np.arange(len(concept_names)))
        axes[RELEVANCE_ROW, i].set_yticklabels(concept_names)
        axes[RELEVANCE_ROW, i].set_xlim(-1.1, 1.1)

        # plot the concept values
        cs = concepts[i].flatten()
        colors_c = get_colors(cs)
        axes[CONCEPT_ROW, i].barh(np.arange(len(cs)),
                                  np.flip(cs.detach().numpy()),
                                  align='center', color=colors_c)

        axes[CONCEPT_ROW, i].set_yticks(np.arange(len(concept_names)))
        axes[CONCEPT_ROW, i].set_yticklabels(concept_names)
        axes[CONCEPT_ROW, i].set_xlim(-concept_lim - 0.2, concept_lim + 0.2)

        # Only show titles for the leftmost plots
        if i == 0:
            axes[CONCEPT_ROW, i].set_ylabel("Concepts scores")
            axes[RELEVANCE_ROW, i].set_ylabel("Relevance scores")

    return fig


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
