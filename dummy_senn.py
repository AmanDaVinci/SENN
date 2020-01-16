import torch.optim as optim

import matplotlib.pyplot as plt
import os

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import urllib.request
import shutil


class CompasDataset(Dataset):
    def __init__(self, compas_path, verbose=True):
        """ProPublica Compas dataset.

        Dataset is read in from preprocessed compas data: `propublica_data_for_fairml.csv`
        from fairml github repo.
        Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'

        Following approach of Alvariz-Melis et al (SENN).

        Parameters
        ----------
        compas_path : str
            Location of Compas data.
        """
        df = pd.read_csv(compas_path)

        # don't know why square root
        df['Number_of_Priors'] = (df['Number_of_Priors'] / df['Number_of_Priors'].max()) ** (1 / 2)
        # get target
        compas_rating = df.score_factor.values  # This is the target?? (-_-)
        df = df.drop('score_factor', axis=1)

        pruned_df, pruned_rating = find_conflicting(df, compas_rating)
        if verbose:
            print('Finish preprocessing data..')

        self.X = pruned_df
        self.y = pruned_rating.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return (self.X.iloc[idx].values.astype(float), self.y[idx])


def load_compas(data_path="datasets/data/compas_new.csv", train_percent=0.8, batch_size=200, num_workers=4,
                valid_size=0.1,
                **kwargs):
    """Return compas dataloaders.

    If compas data can not be found, will download preprocessed compas data: `propublica_data_for_fairml.csv`
    from fairml github repo.

    Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'

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
    compas_url = 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
    if not os.path.isfile(data_path):
        download_file(data_path, compas_url)
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


def find_conflicting(df, labels, consensus_delta=0.2):
    """
    Find examples with same exact feature vector but different label.

    Finds pairs of examples in dataframe that differ only in a few feature values.

    From SENN authors' code.

    Parameters
    ----------
    df : pd.Dataframe
        Containing compas data.
    labels : iterable
        Containing ground truth labels
    consensus_delta : float
        Decision rule parameter.

    Return
    ------
    pruned_df:
        dataframe with `inconsistent samples` removed.
    pruned_lab:
        pruned labels
    """

    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in (range(len(df))):
        if full_dups[i] and (i not in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5) < consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def download_file(store_path, url):
    """Download a file from `url` and write it to a file `store_path`.

    Parameters
    ----------
    store_path : str
        Data storage location.
    """
    # Download the file from `url` and save it locally under `file_name`
    with urllib.request.urlopen(url) as response, open(store_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.parameterizer = nn.Sequential(
            nn.Linear(11, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 11)
        )

    def forward(self, x):
        theta = self.parameterizer(x).unsqueeze(-1)
        aggregated = torch.bmm(theta.transpose(1, 2), x.unsqueeze(-1)).squeeze()
        return torch.sigmoid(aggregated)


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
"""
def accuracy(ys,labels):
    correct_pred = abs(labels - ys) < 0.5
    return float(sum(correct_pred))/len(correct_pred)
"""
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
