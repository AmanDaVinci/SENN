import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

MODEL_FILENAME = "./MNIST_autoencoder_pretrained.pt"

# TODO: Needs Docstrings
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            Flatten(),
            nn.Linear(32, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            Deflatten(),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Deflatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 8, 2, 2)


def get_most_similar(latents, query, number):
    knn = NearestNeighbors(n_neighbors=number).fit(latents)
    return knn.kneighbors(query.reshape(1, -1) if len(query.shape) == 1 else query)


class AETrainer:
    def __init__(self, dataloader, batch_size, **kwargs):
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.model = AutoEncoder()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, epochs):
        for epoch in range(epochs):
            for x, _ in self.dataloader:
                _, decoded = self.model(x)
                loss = self.criterion(decoded, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch}/{epochs}], Loss:{loss.item():.4f}')

    def save_model(self, filename):
        state = {'model_state': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}

        torch.save(state, filename)

    def load_model(self, filename):
        loaded_state = torch.load(filename)
        self.model.load_state_dict(loaded_state["model_state"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])

    def get_latent_reps(self, dataloader):
        latents = []
        for x, _ in dataloader:
            encoded, _ = self.model(x)
            latents.extend(encoded.view(self.batch_size, -1).detach().numpy())
        return np.array(latents)
