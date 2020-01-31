import torch
import torch.nn as nn


class LinearParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, hidden_sizes=(10, 5, 5, 10), dropout=0.5, **kwargs):
        """Parameterizer for compas dataset.
        
        Solely consists of fully connected modules.

        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        hidden_sizes : iterable of int
            Indicates the size of each layer in the network. The first element corresponds to
            the number of input features.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        layers = []
        for h, h_next in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(nn.Linear(h, h_next))
            layers.append(nn.Dropout(self.dropout))
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
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        return self.layers(x).view(x.size(0), self.num_concepts, self.num_classes)


class ConvParameterizer(nn.Module):
    def __init__(self, num_concepts, num_classes, cl_sizes=(1, 10, 20), kernel_size=5, hidden_sizes=(10, 5, 5, 10), dropout=0.5,
                 **kwargs):
        """Parameterizer for MNIST dataset.

        Consists of convolutional as well as fully connected modules.

        Parameters
        ----------
        num_concepts : int
            Number of concepts that should be parameterized (for which the relevances should be determined).
        num_classes : int
            Number of classes that should be distinguished by the classifier.
        cl_sizes : iterable of int
            Indicates the number of kernels of each convolutional layer in the network. The first element corresponds to
            the number of input channels.
        kernel_size : int
            Indicates the size of the kernel window for the convolutional layers.
        hidden_sizes : iterable of int
            Indicates the size of each fully connected layer in the network. The first element corresponds to
            the number of input features. The last element must be equal to the number of concepts multiplied with the
            number of output classes.
        dropout : float
            Indicates the dropout probability.
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.cl_sizes = cl_sizes
        self.kernel_size = kernel_size
        self.dropout = dropout

        cl_layers = []
        for h, h_next in zip(cl_sizes, cl_sizes[1:]):
            cl_layers.append(nn.Conv2d(h, h_next, kernel_size=self.kernel_size))
            # TODO: maybe adaptable parameters for pool kernel size and stride
            cl_layers.append(nn.MaxPool2d(2, stride=2))
            cl_layers.append(nn.ReLU())
        # dropout before maxpool
        cl_layers.insert(-2, nn.Dropout2d(self.dropout))
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
            Relevance scores associated with concepts. Of shape (BATCH, NUM_CONCEPTS, NUM_CLASSES)
        """
        cl_output = self.cl_layers(x)
        flattened = cl_output.view(x.size(0), -1)
        return self.fc_layers(flattened).view(-1, self.num_concepts, self.num_classes)
