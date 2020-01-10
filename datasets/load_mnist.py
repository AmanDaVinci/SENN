import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_mnist(data_path, batch_size, num_workers=4):
    """
    Load mnist data.

    Loads mnist dataset and performs the following preprocessing operations:
        - padding
        - centercropping
        - converting to tensor
        - normalization so that values are in ()

    Parameters
    ----------
    data_path: str
        x
    batch_size: int
        x
    num_workers: int
        x 

    Returns
    -------
    train_loader
        x
    test_loader
        x
    """
    train_trans = transforms.Compose([transforms.Pad(int(np.ceil(28 * 0.05)), padding_mode='edge'),
                                      transforms.CenterCrop(28),  
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(data_path, train=True, transform=train_trans, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, drop_last=True)
    
    test_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST(data_path, train=False, transform=test_trans, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader