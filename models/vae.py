import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim=20):
        """
        VAE encoder assuming a gaussian latent space.

        Input
        -----
        input_dim: int
            Dimensionality of input vectors.
        hidden_dim: iterable of int
            Dimensionality of hidden layers
        z_dim: int
            Dimensionality of the latent space.
        """
        super().__init__()

        self.input_to_hidden = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 124),
            nn.ReLU()
        )
        self.hidden_to_mean = nn.Linear(124, z_dim)
        self.hidden_to_log_std = nn.Linear(124, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Input
        -----
        input: tensor, shape (batch_size, input_dim)
            Contains input data.

        Returns
        -------
        mean:
            tensor of shape (batch_size, z_dim). Contains means of latent distribution.
        std:
            tensor of shape (batch_size, z_dim). Contains standard deviations of
            latent distribution.

        Make sure that any constraints are enforced.
        """
        hidden = self.input_to_hidden(input)
        mean = self.hidden_to_mean(hidden)
        # do we need to take square root for variance? I think not
        std = torch.exp(self.hidden_to_log_std(hidden))

        return mean, std


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim=20):
        """
        VAE decoder assuming Bernoulli output.

        Input
        -----
        input_dim: int
            Dimensionality of input vectors (equivalently, output vectors).
            Dimensionality of hidden layer
        z_dim: int
            Dimensionality of the latent space.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 124),
            nn.ReLU(),
            nn.Linear(124, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, input):
        """
        Perform forward pass of decoder.

        Input
        -----
        input: tensor, shape (batch_size, z_dim)
            Contains encoded latent vector.

        Returns
        -------
        mean:
            tensor of shape (batch_size, input_dim).
            Means of bernoulli distribution for every pixel.
        """
        mean = torch.sigmoid(self.layers(input))

        return mean


class VAE(nn.Module):
    def __init__(self, z_dim=20, device='cpu'):
        """
        A Variational Autoencoder.

        Input
        -----
        input_dim: int
            Dimensionality of input vectors (equivalently, output vectors).
        hidden_dim: int
            Dimensionality of hidden layer
        z_dim: int
            Dimensionality of the latent space.
        device : str, torch.Device
            cuda or cpu.
        """
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, input):
        """
        Perform an encoding and decoding step and return the
        negative average elbo for the given batch.

        Input
        -----
        input: tensor, shape (batch_size, input_dim)
            Contains input data.

        Returns
        -------
        float, negative average elbo for the given batch
        """
        mean, std = self.encoder(input)

        # sample from standard normal distribution
        sample = torch.randn_like(mean)
        latent = mean + std * sample

        reconstruction = self.decoder(latent)

        average_negative_elbo = negative_elbo(input, reconstruction, mean, std).mean()

        return average_negative_elbo

    def sample(self, n_samples=1, noise_values=None):
        """
        Sample n_samples from the model.

        Return both images sampled from bernoulli distributions, as well as
        the means for these bernoullis (as these are used to plot the data manifold).

        Input
        -----
        n_samples: int
            How many samples should be returned.
	noise_values: tensor shape (n_samples, input_dim)
            if specified, uses these exact values as latent variable values instead
            of sampling randomly.

        Returns
        -------
        sampled_ims:
            tensor shape(n_samples, input_dim)
            images sampled from a bernoulli distribution, where parameters
            are given by decoder network.
        im_means:
            tensor shape(n_samples, input_dim)
            bernoulli parameters for every pixel given by decoder network.
        """
        if noise_values is not None:
            assert noise_values.shape[1] == self.z_dim, 'input noise shape should be equal to z_dim'
            in_noise = noise_values
        else:
            in_noise = torch.randn(n_samples, self.z_dim)
        in_noise = in_noise.to(device=self.device)

        # when sampling, we don't train (is this correct?)
        with torch.no_grad():
            im_means = self.decoder(in_noise)

        sampled_ims = (torch.rand(im_means.shape, device=self.device) < im_means).long()
        return sampled_ims, im_means


def negative_elbo(input, reconstruction, mean, std):
    """
    Return negative elbo.
    
    Input
    -----
    input: tensor, shape (batch_size, input_dim)
        Contains input data.
    reconstruction: tensor, shape (batch_size, input_dim)
        Reconstruction of input data generated by decoder network.
    mean: tensor, shape (batch_size, latent_dim)
        means of the variational posterior gaussian distribution
        latent_dim is the dimensionality of the latent space
    std: tensor, shape (batch_size, latent_dim)
        standard deviations of the variational posterior gaussian distribution
        latent_dim is the dimensionality of the latent space
        
    Returns
    -------
    tensor:
        shape (batch_size)
        Negative elbo for each sample in the batch.
    """
    reconstruction_loss = - (input * torch.log(reconstruction) + 
                             (1 - input) * torch.log(1 - reconstruction)).sum(1)
    regularization_loss = kl_divergence(mean, std)
    
    return reconstruction_loss + regularization_loss


def kl_divergence(mean, std):
    """
    Return KL-divergence KL(q||p) of a gaussian q and a unit gaussian p.
    
    The d-dimensional gaussian q is specified by mean and std.
    
    Input
    -----
    mean: tensor, shape (batch_size, latent_dim)
        means of the gaussian distribution q
        latent_dim is the dimensionality of the latent space
    std: tensor, shape (batch_size, latent_dim)
        standard deviations of the gaussian distribution q
        latent_dim is the dimensionality of the latent space
    """
    return ((mean ** 2 + std ** 2 - 1) / 2 - torch.log(std)).sum(1)

