import torch.optim as optim

import matplotlib.pyplot as plt
import os

from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import urllib.request
import shutil


class CompasDataset(Dataset):
    def __init__(self, compas_path):
        """ProPublica Compas dataset.

        Dataset is read in from `two-years-processed` version of the data. Variables are
        created as done in https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm,
        under figure `Risk of General Recidivism Logistic Model`.

        Parameters
        ----------
        compas_path : str
            Location of Compas data.
        """
        df = pd.read_csv(compas_path)

        # preprocessed df
        pre_df = pd.DataFrame()
        pre_df['Two_Year_Recidivism'] = df['two_year_recid']
        pre_df['Number_of_Priors'] = df['priors_count'] / df['priors_count'].max()
        pre_df['Age_Above_FourtyFive'] = df.age_cat == 'Greater than 45'
        pre_df['Age_Below_TwentyFive'] = df.age_cat == 'Less than 25'
        pre_df['African_American'] = df['race'] == 'African-American'
        pre_df['Asian'] = df['race'] == 'Asian'
        pre_df['Hispanic'] = df['race'] == 'Hispanic'
        pre_df['Native_American'] = df['race'] == 'Native American'
        pre_df['Other'] = df['race'] == 'Other'
        pre_df['Female'] = df['sex'] == 'Female'
        pre_df['Misdemeanor'] = df['c_charge_degree'] == 'M'

        self.X = pre_df
        self.y = df.is_recid.values.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return (self.X.iloc[idx].values.astype(float), self.y[idx])


def load_compas(data_path="datasets/data/compas.csv", train_percent=0.8, batch_size=200, num_workers=4, valid_size=0.1,
                **kwargs):
    """Return compas dataloaders.

    Parameters
    ----------
    data_path : str
        Path of compas data.
    train_percent : float
        What percentage of samples should be used as the training set. The rest is used
        for the test set.
    batch_size : int
        Number of samples in minibatches.

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    if not os.path.isfile(data_path):
        download_compas_data(data_path)
    dataset = CompasDataset(data_path)

    # Split into training and test
    train_size = int(train_percent * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    indices = list(range(train_size))
    validation_split = int(valid_size * train_size)
    train_sampler = SubsetRandomSampler(indices[validation_split:])
    valid_sampler = SubsetRandomSampler(indices[:validation_split])

    # Dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader


def download_compas_data(store_path):
    """Download compas data `compas-scores-two-years_processed.csv` from public github repo.

    Parameters
    ----------
    store_path : str
        Data storage location.
    """
    # Download the file from `url` and save it locally under `file_name`:
    url = 'https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv'
    with urllib.request.urlopen(url) as response, open(store_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x).squeeze())


net = Net()
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()
trainloader, _, _ = load_compas()

def train_epoch(model, opt, criterion):
    model.train()
    losses = []
    accuracies = []

    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        opt.zero_grad()
        # (1) Forward
        y_hat = net(inputs.float())
        # (2) Compute diff
        loss = criterion(y_hat, labels)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
        accuracies.append(accuracy(y_hat,labels))
    return losses, accuracies

def accuracy(ys,labels):
    correct_pred = labels == torch.round(ys)
    return float(sum(correct_pred))/len(correct_pred)

e_losses = []
e_accuracies = []
num_epochs = 20
for e in range(num_epochs):
    print('Epoch {}:'.format(e))
    losses, accuracies = train_epoch(net, opt, criterion)
    e_losses.append(losses)
    e_accuracies.append(accuracies)
e_losses = [item for sublist in e_losses for item in sublist]
e_accuracies = [item for sublist in e_accuracies for item in sublist]

plt.plot(range(1, len(e_losses) + 1), e_losses)
plt.show()

plt.plot(range(1, len(e_accuracies) + 1), e_accuracies)
plt.show()
