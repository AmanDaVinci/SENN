import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_barplot(relevances, y_pred, save_path='results/relevances.png'):
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
    concept_names = ['Concept {}'.format(i + 1) for i in range(len(relevances))]
    concept_names.reverse()
    y_pos = np.arange(len(concept_names))
    colors = ['b' if r > 0 else 'r' for r in relevances]
    colors.reverse()

    ax.barh(y_pos, np.flip(relevances.detach().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('Relevances (thetas)')
    ax.set_title('Explanation for prediction: {}'.format(y_pred))

    plt.savefig(save_path)
    plt.clf()


def plot_lambda_accuracy(config_list, save_path):
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

    for config_file in config_list:
        with open(config_file, 'r') as f:
            config = json.load(f)
            lambdas.append(config["robust_reg"])
            path = Path(config["checkpoint_dir"])
            results_csv = path / config["experiment_dir"] / "accuracies_losses_valid.csv"
            dataset = config['dataloader']
        max_accuracy = pd.read_csv(results_csv, header=0)['Accuracy'].max()
        accuracies.append(max_accuracy)
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(lambdas, accuracies, "r.-")
    ax.set_xlabel('Robustness Regularization Strength')
    ax.set_ylabel('Prediction Accuracy')
    
    plt.savefig(save_path)
    plt.clf()