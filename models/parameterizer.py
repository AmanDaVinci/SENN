import torch
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
    def __init__(self, cl_sizes=(1, 10, 20), kernel_size=5, hidden_sizes=(10, 5, 5, 10), dropout=0.5, **kwargs):
        """Parameterizer for MNIST dataset.

        Consists of convolutional as well as fully connected modules.

        Parameters
        ----------
        cl_sizes : iterable of int
            Indicates the number of kernels of each convolutional layer in the network. The first element corresponds to
            the number of input channels.
        kernel_size : int
            Indicates the size of the kernel window for the convolutional layers.
        hidden_sizes : iterable of int
            Indicates the size of each fully connected layer in the network. The first element corresponds to
            the number of input features.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.cl_sizes = cl_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout

        cl_layers = []
        for h, h_next in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(nn.Conv2d(h, h_next, kernel_size=self.kernel_size))
            # TODO: maybe adaptable parameters for pool kernel size and stride
            cl_layers.append(nn.MaxPool2d(3, stride=2))
            cl_layers.append(nn.ReLU())
        cl_layers.append(nn.Dropout2d(self.dropout))
        self.cl_layers = nn.Sequential(*cl_layers)

        fc_layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            fc_layers.append(nn.Linear(h, h_next))
            fc_layers.append(nn.Dropout(self.dropout))
            fc_layers.append(nn.ReLU())
        fc_layers.pop()
        fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Forward pass of MNIST parameterizer.

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
        return self.fc_layers(torch.flatten(self.cl_layers(x)))



