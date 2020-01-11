import torch.nn as nn
import torchvision.utils as vutils


class CompasParameterizer(nn.Module):
    def __init__(self, hidden_sizes=(10, 5, 5, 10), **kwargs):
        """Parameterizer for compas dataset.
        
        Solely consists of fully connected modules.

        Parameters
        ----------
        hidden_sizes : iterable of int
            Indicates the size of each layer in the network. The first element corresponds to
            the number of input features.
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(h, h_next))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of compas parameterizer.

        Computes relevance parameters theta.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        parameters : torch.Tensor
            Relevance scores associated with concepts. Of shape (NUM_CONCEPTS, *)
        """
        return self.layers(x)


class MNISTParameterizer(nn.Module):
    def __init__(self, **kwargs):
        pass


