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

    start = 0.0
    end = num_concepts * x.size(-1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start+0.5*stepsize, end-0.5*stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))
    ax.set_title('Basis concepts: ')
    save(make_grid(top_examples, nrow=num_prototypes, pad_value=1), save_path)
    plt.rcdefaults()

def highest_contrast(model, test_loader, num_concepts=5, num_prototypes=6, save_path="results/concepts.png"):
    model.eval()
    activations = []
    for i, (x, labels) in enumerate(test_loader):
        x = x.float().to("cuda:0" if next(model.parameters()).is_cuda else "cpu")
        with torch.no_grad():
            _, (concepts, parameters), _ = model(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    contrast_scores = torch.empty_like(activations)
    for c in range(num_concepts):
        contrast_scores[:, c] = activations[:, c] - (activations[:, :c].sum() + activations[:, c:].sum())

    top_test, top_test_idx = torch.topk(contrast_scores, num_prototypes, 0)

    top_examples = [test_loader.dataset.data[top_test_idx[:, concept]] for concept in range(num_concepts)]
    # flatten list and ensure correct image shape
    top_examples = [img.unsqueeze(0) if len(img.size()) == 2 else img for sublist in top_examples for img in sublist]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    start = 0.0
    end = num_concepts * x.size(-1)
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start+0.5*stepsize, end-0.49*stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} data examples with highest contrast per concept'.format(num_prototypes))
    ax.set_title('Basis concepts: ')
    save(make_grid(top_examples, nrow=num_prototypes, pad_value=1), save_path)
    plt.rcdefaults()

def filter_concepts(model, num_concepts=5, num_prototypes=10, save_path="results/concepts.png"):
    model.eval()
    plt.rcdefaults()
    fig, ax = plt.subplots()
    concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

    filters = [f for f in model.conceptizer.encoder[-2][0].weight.data.clone()]
    imgs = [dim.unsqueeze(0) for f in filters for dim in f]

    start = 0.0
    end = num_concepts * filters[0].size(-1) + 2
    stepsize = abs(end - start) / num_concepts
    ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
    ax.set_yticklabels(concept_names)
    plt.xticks([])
    ax.set_xlabel('{} dimensions of concept filters'.format(num_prototypes))
    ax.set_title('Basis concepts: ')

    save(make_grid(imgs, nrow=num_prototypes, normalize=True, padding=1, pad_value=1), save_path)
    plt.rcdefaults()

def save(img, save_path):
    img = img.clone().squeeze()
    npimg = img.numpy()
    if len(npimg.shape) == 2:
        plt.imsave(save_path, npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.savefig(save_path)
    plt.clf()