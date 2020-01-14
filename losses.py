import torch
import torch.nn.functional as F

def weighted_mse(x, x_hat, sparsity):
    return sparsity * F.mse_loss(x,x_hat)

def mse_kl_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + F.kl_div(sparsity*torch.ones_like(concepts), concepts)

def mse_l1_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + sparsity * torch.abs(concepts).sum()

def robustness_loss(x, parameters, model):
    return torch.tensor(0)