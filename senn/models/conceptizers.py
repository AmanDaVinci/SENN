from abc import abstractmethod

import torch
import torch.nn as nn


class Conceptizer(nn.Module):
    def __init__(self):
        """
        A general Conceptizer meta-class. Children of the Conceptizer class
        should implement encode() and decode() functions.
        """
        super(Conceptizer, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

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

    @abstractmethod
    def encode(self, x):
        """
        Abstract encode function to be overridden.
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, *). Only restriction on the shape is that
            the first dimension should correspond to the batch size.
        """
        pass

    @abstractmethod
    def decode(self, encoded):
        """
        Abstract decode function to be overridden.
        Parameters
        ----------
        encoded : torch.Tensor
            Latent representation of the data
        """
        pass


class IdentityConceptizer(Conceptizer):
    def __init__(self, **kwargs):
        """
        Basic Identity Conceptizer that returns the unchanged input features.
        """
        super().__init__()

    def encode(self, x):
        """Encoder of Identity Conceptizer.

        Leaves the input features unchanged  but reshapes them to three dimensions
        and returns them as concepts (use of raw features -> no concept computation)

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, INPUT_FEATURES).

        Returns
        -------
        concepts : torch.Tensor
            Unchanged input features but with extra dimension (BATCH, INPUT_FEATURES, 1).
        """
        return x.unsqueeze(-1)

    def decode(self, z):
        """Decoder of Identity Conceptizer.

        Simulates reconstruction of the original input x by undoing the reshaping of the encoder.
        The 'reconstruction' is identical to the input of the conceptizer.

        Parameters
        ----------
        z : torch.Tensor
            Output of encoder (input x reshaped to three dimensions), size: (BATCH, INPUT_FEATURES, 1).

        Returns
        -------
        reconst : torch.Tensor
            Unchanged input features (identical to x)
        """
        return z.squeeze(-1)


class VaeConceptizer(nn.Module):
    """Variational Auto Encoder to generate basis concepts

    Concepts should be independently sensitive to single generative factors,
    which will lead to better interpretability and fulfill the "diversity" 
    desiderata for basis concepts in a Self-Explaining Neural Network.
    VAE can be used to learn disentangled representations of the basis concepts 
    by emphasizing the discovery of latent factors which are disentangled. 
    """

    def __init__(self, image_size, num_concepts, **kwargs):
        """Initialize Variational Auto Encoder

        Parameters
        ----------
        image_size : int
            size of the width or height of an image, assumes square image
        num_concepts : int
            number of basis concepts to learn in the latent distribution space
        """
        super().__init__()
        self.in_dim = image_size * image_size
        self.z_dim = num_concepts
        self.encoder = VaeEncoder(self.in_dim, self.z_dim)
        self.decoder = VaeDecoder(self.in_dim, self.z_dim)

    def forward(self, x):
        """Forward pass through the encoding, sampling and decoding step

        Parameters
        ----------
        x : torch.tensor 
            input of shape [batch_size x ... ], which will be flattened

        Returns
        -------
        concept_mean : torch.tensor
            mean of the latent distribution induced by the posterior input x
        x_reconstruct : torch.tensor
            reconstruction of the input in the same shape
        """
        concept_mean, concept_logvar = self.encoder(x)
        concept_sample = self.sample(concept_mean, concept_logvar)
        x_reconstruct = self.decoder(concept_sample)
        return (concept_mean.unsqueeze(-1),
                concept_logvar.unsqueeze(-1),
                x_reconstruct.view_as(x))

    def sample(self, mean, logvar):
        """Samples from the latent distribution using reparameterization trick

        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is drawn from a standard normal distribution
        
        Parameters
        ----------
        mean : torch.tensor
            mean of the latent distribution of shape [batch_size x z_dim]
        log_var : torch.tensor
            diagonal log variance of the latent distribution of shape [batch_size x z_dim]
        
        Returns
        -------
        z : torch.tensor
            sample latent tensor of shape [batch_size x z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        return z


class VaeEncoder(nn.Module):
    """Encoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron

        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(100, z_dim)
        self.logvar_layer = nn.Linear(100, z_dim)

    def forward(self, x):
        """Forward pass of the encoder
        """
        x = self.FC(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar


class VaeDecoder(nn.Module):
    """Decoder of a VAE"""

    def __init__(self, in_dim, z_dim):
        """Instantiate a multilayer perceptron

        Parameters
        ----------
        in_dim: int
            dimension of the input data
        z_dim: int
            latent dimension of the encoder output
        """
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.FC = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim)
        )

    def forward(self, x):
        """Forward pass of a decoder"""
        x_reconstruct = torch.sigmoid(self.FC(x))
        return x_reconstruct


class ConvConceptizer(Conceptizer):
    def __init__(self, image_size, num_concepts, concept_dim, image_channels=1, encoder_channels=(10,),
                 decoder_channels=(16, 8), kernel_size_conv=5, kernel_size_upsample=(5, 5, 2),
                 stride_conv=1, stride_pool=2, stride_upsample=(2, 1, 2),
                 padding_conv=0, padding_upsample=(0, 0, 1), **kwargs):
        """
        CNN Autoencoder used to learn the concepts, present in an input image

        Parameters
        ----------
        image_size : int
            the width of the input image
        num_concepts : int
            the number of concepts
        concept_dim : int
            the dimension of each concept to be learned
        image_channels : int
            the number of channels of the input images
        encoder_channels : tuple[int]
            a list with the number of channels for the hidden convolutional layers
        decoder_channels : tuple[int]
            a list with the number of channels for the hidden upsampling layers
        kernel_size_conv : int, tuple[int]
            the size of the kernels to be used for convolution
        kernel_size_upsample : int, tuple[int]
            the size of the kernels to be used for upsampling
        stride_conv : int, tuple[int]
            the stride of the convolutional layers
        stride_pool : int, tuple[int]
            the stride of the pooling layers
        stride_upsample : int, tuple[int]
            the stride of the upsampling layers
        padding_conv : int, tuple[int]
            the padding to be used by the convolutional layers
        padding_upsample : int, tuple[int]
            the padding to be used by the upsampling layers
        """
        super(ConvConceptizer, self).__init__()
        self.num_concepts = num_concepts
        self.filter = filter
        self.dout = image_size

        # Encoder params
        encoder_channels = (image_channels,) + encoder_channels
        kernel_size_conv = handle_integer_input(kernel_size_conv, len(encoder_channels))
        stride_conv = handle_integer_input(stride_conv, len(encoder_channels))
        stride_pool = handle_integer_input(stride_pool, len(encoder_channels))
        padding_conv = handle_integer_input(padding_conv, len(encoder_channels))
        encoder_channels += (num_concepts,)

        # Decoder params
        decoder_channels = (num_concepts,) + decoder_channels
        kernel_size_upsample = handle_integer_input(kernel_size_upsample, len(decoder_channels))
        stride_upsample = handle_integer_input(stride_upsample, len(decoder_channels))
        padding_upsample = handle_integer_input(padding_upsample, len(decoder_channels))
        decoder_channels += (image_channels,)

        # Encoder implementation
        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(self.conv_block(in_channels=encoder_channels[i],
                                                out_channels=encoder_channels[i + 1],
                                                kernel_size=kernel_size_conv[i],
                                                stride_conv=stride_conv[i],
                                                stride_pool=stride_pool[i],
                                                padding=padding_conv[i]))
            self.dout = (self.dout - kernel_size_conv[i] + 2 * padding_conv[i] + stride_conv[i] * stride_pool[i]) // (
                    stride_conv[i] * stride_pool[i])

        if self.filter and concept_dim == 1:
            self.encoder.append(ScalarMapping((self.num_concepts, self.dout, self.dout)))
        else:
            self.encoder.append(Flatten())
            self.encoder.append(nn.Linear(self.dout ** 2, concept_dim))

        # Decoder implementation
        self.unlinear = nn.Linear(concept_dim, self.dout ** 2)
        self.decoder = nn.ModuleList()
        decoder = []
        for i in range(len(decoder_channels) - 1):
            decoder.append(self.upsample_block(in_channels=decoder_channels[i],
                                               out_channels=decoder_channels[i + 1],
                                               kernel_size=kernel_size_upsample[i],
                                               stride_deconv=stride_upsample[i],
                                               padding=padding_upsample[i]))
            decoder.append(nn.ReLU(inplace=True))
        decoder.pop()
        decoder.append(nn.Tanh())
        self.decoder = nn.ModuleList(decoder)

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
        reconst = reconst.view(-1, self.num_concepts, self.dout, self.dout)
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

    def upsample_block(self, in_channels, out_channels, kernel_size, stride_deconv, padding):
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
        stride_deconv : int
            the stride of the deconvolution
        padding : int
            the size of padding

        Returns
        -------
        sequence : nn.Sequence
            a sequence of deconvolutional and activation modules
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride_deconv,
                               padding=padding),
        )


class Flatten(nn.Module):
    def forward(self, x):
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
        return x.view(x.size(0), x.size(1), -1)


def handle_integer_input(input, desired_len):
    """
    Checks if the input is an integer or a list.
    If an integer, it is replicated the number of  desired times
    If a tuple, the tuple is returned as it is

    Parameters
    ----------
    input : int, tuple
        The input can be either a tuple of parameters or a single parameter to be replicated
    desired_len : int
        The length of the desired list

    Returns
    -------
    input : tuple[int]
        a tuple of parameters which has the proper length.
    """
    if type(input) is int:
        return (input,) * desired_len
    elif type(input) is tuple:
        if len(input) != desired_len:
            raise AssertionError("The sizes of the parameters for the CNN conceptizer do not match."
                                 f"Expected '{desired_len}', but got '{len(input)}'")
        else:
            return input
    else:
        raise TypeError(f"Wrong type of the parameters. Expected tuple or int but got '{type(input)}'")


class ScalarMapping(nn.Module):
    def __init__(self, conv_block_size):
        """
        Module that maps each filter of a convolutional block to a scalar value

        Parameters
        ----------
        conv_block_size : tuple (int iterable)
            Specifies the size of the input convolutional block: (NUM_CHANNELS, FILTER_HEIGHT, FILTER_WIDTH)
        """
        super().__init__()
        self.num_filters, self.filter_height, self.filter_width = conv_block_size

        self.layers = nn.ModuleList()
        for _ in range(self.num_filters):
            self.layers.append(nn.Linear(self.filter_height * self.filter_width, 1))

    def forward(self, x):
        """
        Reduces a 3D convolutional block to a 1D vector by mapping each 2D filter to a scalar value.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape (BATCH, CHANNELS, HEIGHT, WIDTH).

        Returns
        -------
        mapped : torch.Tensor
            Reduced input (BATCH, CHANNELS, 1)
        """
        x = x.view(-1, self.num_filters, self.filter_height * self.filter_width)
        mappings = []
        for f, layer in enumerate(self.layers):
            mappings.append(layer(x[:, [f], :]))
        return torch.cat(mappings, dim=1)
