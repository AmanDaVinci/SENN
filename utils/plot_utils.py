import numpy as np
import matplotlib.pyplot as plt


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

    ax.barh(y_pos, np.flip(relevances.detach().numpy()), align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('Relevances (thetas)')
    ax.set_title('Explanation for prediction: {}'.format(y_pred))

    plt.savefig(save_path)
    plt.clf()
