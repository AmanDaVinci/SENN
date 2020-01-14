import torch
import torch.nn.functional as F
from utils.jacobian import jacobian

def get_robust_loss(x, relevances, num_concepts, num_classes, SENN):
    """Computes Robustness Loss given by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]


    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x concept_dim)
    num_concepts : int
                 Number of concepts where we assume the concept dimension is always 1
    num_classes  : int
                 Number of classes as the final prediction 
    SENN         : nn.Module
                 SENN containing a method called .conceptizer.encode() 

    Returns
    -------
    robust_loss  : torch.tensor
        Robustness loss is meaned across (batch_size x num_classes x num_features)
    """
    def y_SENN(x):
        y, _, _ = SENN(x)
        return y

    J_yx = jacobian(y_SENN, x, num_classes)
    J_hx = jacobian(SENN.conceptizer.encode, x, num_concepts)
    robust_loss = (J_yx - torch.bmm(relevances.permute(0,2,1), J_hx))
    
    return robust_loss.mean()

def weighted_mse(x, x_hat, sparsity):
    return sparsity * F.mse_loss(x,x_hat)

def mse_kl_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + F.kl_div(sparsity*torch.ones_like(concepts), concepts)

def mse_l1_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + sparsity * torch.abs(concepts).sum()

