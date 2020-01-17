import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def highest_activations(model, test_loader, num_concepts=5, num_prototypes=5):
    model.eval()
    activations = []
    for i, (x, labels) in enumerate(test_loader):
        x = x.float().to(model.device)
        with torch.no_grad():
            _, (concepts, parameters), _ = model(x)
            activations.append(concepts.squeeze())
    activations = torch.cat(activations)

    top_activations = torch.empty(num_concepts)
    top_idxs = torch.empty(num_concepts)
    top_examples = []
    for c in range(num_concepts):
        top_activations[c], top_idxs[c] = torch.topk(activations, num_prototypes, 0)
        for p in range(num_prototypes):
            top_examples.append(test_loader.dataset[top_idxs[p]])
    show(make_grid(top_examples, nrow=num_prototypes, pad_value=1))

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')