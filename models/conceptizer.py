import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.utils as vutils

import numpy as np

class IdentityConceptizer(nn.Module):
    def __init__(self, **kwargs):
        """
        Basic Identity Conceptizer that returns the unchanged input features.
        """
        super().__init__()

    def encode(self, x):
        """Encoder of Identity Conceptizer.

        Returns the unchanged input features as concepts (use of raw features -> no concept computation).

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, INPUT_FEATURES).

        Returns
        -------
        concepts : torch.Tensor
            Unchanged input features (identical to x)
        """
        return x

    def decode(self, z):
        """Decoder of Identity Conceptizer.

        Returns the unchanged input features as concepts (use of raw features -> no concept computation).

        Parameters
        ----------
        z : torch.Tensor
            Output of encoder (identical to encoder input x), size: (BATCH, INPUT_FEATURES).

        Returns
        -------
        reconst : torch.Tensor
            Unchanged input features (identical to x)
        """
        return z

class MNISTConceptizer(nn.Module):
    def __init__(self, **kwargs):
        pass


