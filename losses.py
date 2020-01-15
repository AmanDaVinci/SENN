import torch
import torch.nn.functional as F
from utils.jacobian import jacobian

def compas_robustness_loss(x, relevances, SENN):
    """Computes Robustness Loss for the Compas data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
    SENN         : nn.Module
                 SENN containing a method for .conceptizer.encode() 

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as (batch_size x num_classes x num_features)
        Frobenius norm across num_classes and num_features
        Averaged across batch_size
    """
    num_concepts = relevances.size()[1]
    num_classes = relevances.size()[2]

    def SENN_aggregator(x):
        aggregate, _, _ = SENN(x)
        # based on the design decision that concept_dim is always 1
        aggregate.reshape(-1, num_classes)
        return aggregate

    def compas_conceptizer(x):
        concepts = SENN.conceptizer.encode(x)
        # based on the design decision that concept_dim is always 1
        concepts.reshape(-1, num_concepts)
        return concepts

    J_yx = jacobian(SENN_aggregator, x, num_classes)
    J_hx = jacobian(compas_conceptizer, x, num_concepts)
    robustness_loss = (J_yx - torch.bmm(relevances.permute(0,2,1), J_hx))

    return robustness_loss.norm(dim=(1,2)).mean()

def weighted_mse(x, x_hat, sparsity):
    return sparsity * F.mse_loss(x,x_hat)

def mse_kl_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + F.kl_div(sparsity*torch.ones_like(concepts), concepts)

def mse_l1_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + sparsity * torch.abs(concepts).sum()

