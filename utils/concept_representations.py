import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

def highest_activations(model, test_loader, num_concepts=5, num_prototypes=6, save_path="results/concepts.png"):
    model.eval()
    activations = []
    for i, (x, labels) in enumerate(test_loader):
        x = x.float().to("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            _, (concepts, parameters), _ = model(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    top_test, top_test_idx = torch.topk(activations, num_prototypes, 0)

    top_examples = [test_loader.dataset.data[top_test_idx[:, concept]] for concept in range(num_concepts)]
    # flatten list and ensure correct image shape
    top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]
    concept_names.reverse()

    start = 0.0
    end = num_concepts * top_examples[0].size(1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start+0.5*stepsize, end-0.5*stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))
    ax.set_title('Basis concepts: ')
    save(make_grid(top_examples, nrow=num_prototypes, pad_value=1),save_path)
    plt.rcdefaults()

def save(img, save_path):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.savefig(save_path)
    plt.clf()