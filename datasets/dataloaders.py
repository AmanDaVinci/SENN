import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import urllib.request
import shutil


def get_dataloader(config):
    """Dispatcher that calls dataloader function depending on the config.

    Parameters
    ----------
    config : SimpleNameSpace
        Contains config values. Needs to at least have a `dataloader` field.
    
    Returns
    -------
    Corresponding dataloader.
    """
    if config.dataloader.lower() == 'mnist':
        return load_mnist(**config.__dict__)
    elif config.dataloader.lower() == 'compas':
        return load_compas(**config.__dict__)


def load_mnist(data_path, batch_size, num_workers=4, valid_size=0.1, **kwargs):
    """
    Load mnist data.

    Loads mnist dataset and performs the following preprocessing operations:
        - converting to tensor
        - standard mnist normalization so that values are in (0, 1)

    Parameters
    ----------
    data_path: str
        Location of mnist data.
    batch_size: int
        Batch size.
    num_workers: int
        the number of  workers to be used by the Pytorch DataLoaders
    valid_size : float
        a float between 0.0 and 1.0 for the percent of samples to be used for validation

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_size = len(train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader


#  --------------- Compas Dataset  ---------------

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


def load_compas(compas_path, train_percent=0.8, batch_size=200, num_workers=4, valid_size=0.1, **kwargs):
    """Return compas dataloaders.

    Parameters
    ----------
    compas_path : str
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
    if not os.path.isfile(compas_path):
        download_compas_data(compas_path)
    dataset = CompasDataset(compas_path)

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
