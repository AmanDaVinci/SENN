import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_mnist(data_path, batch_size, num_workers=4, valid_size=0.1):
    """
    Load mnist data.

    Loads mnist dataset and performs the following preprocessing operations:
        - converting to tensor
        - normalization so that values are in [-1, 1]

    Parameters
    ----------
    data_path: str
        x
    batch_size: int
        x
    num_workers: int
        x
    valid_size
        x

    Returns
    -------
    train_loader
        x
    valid_loader
        x
    test_loader
        x
    """

    transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_size = len(train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader
