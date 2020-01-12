import torch
import torch.nn as nn
import torch.optim as opt
import torchvision.utils as vutils

import numpy as np

class Conceptizer(nn.Module):
    def __init__(self):
        """
        A general Conceptizer meta-class. Children of the Conceptizer class
        should implement encode() and decode() functions.
        """
        super(Conceptizer, self).__init__()

    def forward(self, x):
        """
        Forward pass of the general conceptizer.

        Computes concepts present in the input.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.

        Returns
        -------
        encoded : torch.Tensor
            Encoded concepts (batch_size, concept_number, concept_dimension)
        decoded : torch.Tensor
            Reconstructed input (batch_size, *)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded.view_as(x)


class Conceptizer_CNN(Conceptizer):
    def __init__(self, image_size, concept_num, concept_dim, image_channels=1, encoder_channels=[10],
                 decoder_channels=[16, 8], kernel_size=5, stride_conv=1, stride_pool=2, stride_upsample=2, padding=0):
        """
        CNN Autoencoder used to learn the concepts, present in an input image

        Parameters
        ----------
        image_size : int
            the width of the input image
        concept_num : int
            the number of concepts
        concept_dim :
            the dimension of each concept to be learned
        image_channels : int
            the number of channels of the input images
        encoder_channels : list[int]
            a list with the number of channels for the hidden convolutional layers
        decoder_channels : list[int]
            a list with the number of channels for the hidden upsampling layers
        kernel_size : int
            the size of the kernels to be used for convolution and upsampling
        stride_conv : int
            the stride of the convolutional layers
        stride_pool : int
            the stride of the pooling layers
        stride_upsample : int
            the stride of the upsampling layers
        padding : int
            the padding to be used by the convolutional and upsampling layers
        """
        super(Conceptizer_CNN, self).__init__()
        self.concept_num = concept_num
        self.dout = image_size

        encoder_channels.insert(0, image_channels)
        encoder_channels.append(concept_num)

        decoder_channels.insert(0, concept_num)
        decoder_channels.append(image_channels)

        # Encoder implementation
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(self.conv_block(in_channels=encoder_channels[i],
                                                out_channels=encoder_channels[i + 1],
                                                kernel_size=kernel_size,
                                                stride_conv=stride_conv,
                                                stride_pool=stride_pool,
                                                padding=padding))
            self.dout = (self.dout - kernel_size + 2 * padding + stride_conv * stride_pool) // (stride_conv * stride_pool)

        self.encoder.append(Flatten())
        self.encoder.append(nn.Linear(self.dout ** 2, concept_dim))

        # Decoder implementation
        self.unlinear = nn.Linear(concept_dim, self.dout ** 2)
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            # the last activation needs to be Tanh
            activation = nn.ReLU(inplace=True) if i != (len(decoder_channels) - 2) else nn.Tanh()
            self.decoder.append(self.upsample_block(in_channels=decoder_channels[i],
                                                    out_channels=decoder_channels[i + 1],
                                                    kernel_size=kernel_size,
                                                    stride_conv=stride_upsample if i % 2 == 0 else 1,
                                                    padding=padding,
                                                    activation=activation))

    def encode(self, x):
        """
        The encoder part of the autoencoder which takes an Image as an input
        and learns its hidden representations (concepts)

        Parameters
        ----------
        x : Image (batch_size, channels, width, height)

         Returns
        -------
        encoded : torch.Tensor (batch_size, concept_number, concept_dimension)
            the concepts representing an image

        """
        encoded = x
        for module in self.encoder:
            encoded = module(encoded)
        return encoded

    def decode(self, z):
        """
        The decoder part of the autoencoder which takes a hidden representation as an input
        and tries to reconstruct the original image

        Parameters
        ----------
        z : torch.Tensor (batch_size, channels, width, height)
            the concepts in an image

        Returns
        -------
        reconst : torch.Tensor (batch_size, channels, width, height)
            the reconstructed image

        """
        reconst = self.unlinear(z)
        reconst = reconst.view(-1, self.concept_num, self.dout, self.dout)
        for module in self.decoder:
            reconst = module(reconst)
        return reconst

    def conv_block(self, in_channels, out_channels, kernel_size, stride_conv, stride_pool, padding):
        """
        A helper function that constructs a convolution block with pooling and activation

        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_conv : int
            the stride of the deconvolution
        stride_pool : int
            the stride of the pooling layer
        padding : int
            the size of padding

        Returns
        -------
        sequence : nn.Sequence
            a sequence of convolutional, pooling and activation modules
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride_conv,
                      padding=padding),
            # nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=stride_pool,
                         padding=padding),
            nn.ReLU(inplace=True)
        )

    def upsample_block(self, in_channels, out_channels, kernel_size, stride_conv, padding, activation=nn.ReLU(inplace=True)):
        """
        A helper function that constructs an upsampling block with activations

        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of output channels
        kernel_size : int
            the size of the convolutional kernel
        stride_conv : int
            the stride of the deconvolution
        padding : int
            the size of padding
        activation : nn.Module
            the activation to be used

        Returns
        -------
        sequence : nn.Sequence
            a sequence of deconvolutional and activation modules
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride_conv,
                               padding=padding),
            activation
        )


class Flatten(nn.Module):
    def forward(self, input):
        """
        Flattens the inputs to only 3 dimensions, preserving the sizes of the 1st and 2nd.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (dim1, dim2, *).

        Returns
        -------
        flattened : torch.Tensor
            Flattened input (dim1, dim2, dim3)
        """
        return input.view(input.size(0), input.size(1), -1)
