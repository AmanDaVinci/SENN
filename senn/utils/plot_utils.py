import json
from os import path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import trainer
from .concept_representations import highest_activations, highest_contrast, filter_concepts

RESULTS_DIR = 'results'
CONFIG_DIR = 'configs'
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
        concept_names = [f'C{i + 1}' for i in range(concepts.shape[1] - 1, -1, -1)]

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


def create_barplot(ax, relevances, y_pred, x_lim=1.1, title='', x_label='', concept_names=None, **kwargs):
    """Creates a bar plot of relevances.

    Parameters
    ----------
    ax : pyplot axes object
        The axes on which the bar plot should be created.
    relevances: torch.tensor
        The relevances for which the bar plot should be generated. shape: (1, NUM_CONCEPTS, NUM_CLASSES)
    y_pred: torch.tensor (int)
        The prediction of the model for the corresponding relevances. shape: scalar value
    x_lim: float
        the limits of the plot
    title: str
        the title of the plot
    x_label: str
        the label of the X-axis of the plot
    concept_names: list[str]
        the names of each feature on the plot
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
        List of experiment configs files used to vary the lambda.
        If multiple seeds are used then this is a list of lists where the inner lists have a length
        equal to the number of different seeds used and contain the corresponding configs files.
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
    std_seeds = []

    path = Path(CONFIG_DIR)
    for config_file in config_list:
        seed_accuracies = []
        for seed in range(num_seeds):
            config_path = path / config_file if num_seeds == 1 else path / config_file[seed]
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
        std_seeds.append(np.std(seed_accuracies))
        accuracies.append(sum(seed_accuracies) / num_seeds)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.errorbar(np.arange(len(lambdas)), accuracies, std_seeds, color='r', marker='o')
    ax.set_xticks(np.arange(len(lambdas)))
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(lambdas, fontsize=12)
    ax.set_xlabel('Robustness Regularization Strength', fontsize=18)
    ax.set_ylabel('Prediction Accuracy', fontsize=18)
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    return fig


def show_explainations(model, test_loader, dataset, num_explanations=2, save_path=None, batch_size=128, concept_names=None,
                       **kwargs):
    """Generates some explanations of model predictions.

    Parameters
    ----------
    model : torch nn.Module
        model to visualize
    test_loader: Dataloader object
        Test set dataloader to iterate over test set.
    dataset : str
        Name of the dataset used.
    save_path : str
        Directory where the figures are saved. If None a figure is showed instead.
    batch_size : int
        batch_size of test loader
    """
    device = 'cuda:0' if next(model.parameters()).is_cuda else 'cpu'
    model.eval()

    # select test example
    (test_batch, test_labels) = next(iter(test_loader))
    test_batch = test_batch.float().to(device)

    # feed test batch to model to obtain explanation
    y_pred, (concepts, relevances), _ = model(test_batch)
    if len(y_pred.size()) > 1:
        y_pred = y_pred.argmax(1)

    concepts_min = concepts.min().item()
    concepts_max = concepts.max().item()
    concept_lim = abs(concepts_min) if abs(concepts_min) > abs(concepts_max) else abs(concepts_max)

    plt.style.use('seaborn-paper')
    batch_idx = np.random.randint(0, batch_size - 1, num_explanations)
    for i in range(num_explanations):
        if concept_names is not None:
            gridsize = (1, 2)
            fig = plt.figure(figsize=(12, 6))
            ax1 = plt.subplot2grid(gridsize, (0, 0))
            ax2 = plt.subplot2grid(gridsize, (0, 1))

            create_barplot(ax1, relevances[batch_idx[i]], y_pred[batch_idx[i]], x_label='Relevances (theta)',
                           concept_names=concept_names, **kwargs)
            ax1.xaxis.set_label_position('top')
            ax1.tick_params(which='major', labelsize=12)

            create_barplot(ax2, concepts[batch_idx[i]], y_pred[batch_idx[i]], x_lim=concept_lim,
                           x_label='Concepts/Raw Inputs', concept_names=concept_names, **kwargs)
            ax2.xaxis.set_label_position('top')
            ax2.tick_params(which='major', labelsize=12)

        else:
            gridsize = (1, 3)
            fig = plt.figure(figsize=(9, 3))
            ax1 = plt.subplot2grid(gridsize, (0, 0))
            ax2 = plt.subplot2grid(gridsize, (0, 1))
            ax3 = plt.subplot2grid(gridsize, (0, 2))

            # figure of example image
            ax1.imshow(test_batch[batch_idx[i]].squeeze().cpu(), cmap='gray')
            ax1.set_axis_off()
            ax1.set_title(f'Input Prediction: {y_pred[batch_idx[i]].item()}', fontsize=18)

            create_barplot(ax2, relevances[batch_idx[i]], y_pred[batch_idx[i]], x_label='Relevances (theta)', **kwargs)
            ax2.xaxis.set_label_position('top')
            ax2.tick_params(which='major', labelsize=12)

            create_barplot(ax3, concepts[batch_idx[i]], y_pred[batch_idx[i]], x_lim=concept_lim,
                           x_label='Concept activations (h)', **kwargs)
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(which='major', labelsize=12)

        plt.tight_layout()

        plt.show() if save_path is None else plt.savefig(path.join(save_path, 'explanation_{}.png'.format(i)))
        plt.close('all')


def show_prototypes(model, test_loader, representation_type='activation', save_path=None, **kwargs):
    """Generates prototypes for concept representation.

    Parameters
    ----------
    model : torch nn.Module
        model to visualize
    test_loader: Dataloader object
        Test set dataloader to iterate over test set.
    representation_type : str
        Name of the representation type used.
    save_path : str
        Directory where the figures are saved. If None a figure is showed instead.
    """
    if representation_type == 'activation':
        highest_activations(model, test_loader, save_path=save_path)
    elif representation_type == 'contrast':
        highest_contrast(model, test_loader, save_path=save_path)
    elif representation_type == 'filter':
        filter_concepts(model, save_path=save_path)
