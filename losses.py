import torch
import torch.nn.functional as F
from utils.jacobian import jacobian

def compas_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # add class dimension
    # TODO: this should be fixed upstream, not here
    aggregates = aggregates.unsqueeze(-1)
    
    x.requires_grad_(True)
    batch_size = x.size(0)
    num_features = x.size(1)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes) 
    grad_tensor.to(x.device)
    J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
     grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1) 

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances

    return robustness_loss.norm(p='fro')

def mnist_robustness_loss(x, aggregates, concepts, relevances):
    """Computes Robustness Loss for MNIST data
    
    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design

    Parameters
    ----------
    x            : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)
   
    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    # # add class dimension
    # # TODO: this should be fixed upstream, not here
    # aggregates = aggregates.unsqueeze(-1)
    
    # x.requires_grad_(True)
    # batch_size = x.size(0)
    # num_features = x.size(1)
    # num_classes = aggregates.size(1)

    # grad_tensor = torch.ones(batch_size, num_classes) 
    # grad_tensor.to(x.device)
    # J_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
    #  grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
    # # bs x num_features -> bs x num_features x num_classes
    # J_yx = J_yx.unsqueeze(-1) 

    # # J_hx = Identity Matrix; h(x) is identity function
    # robustness_loss = J_yx - relevances

    # return robustness_loss.norm(p='fro')
    return torch.tensor(0.0)

def weighted_mse(x, x_hat, sparsity):
    return sparsity * F.mse_loss(x,x_hat)

def mse_kl_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + F.kl_div(sparsity*torch.ones_like(concepts), concepts)

def mse_l1_sparsity(x, x_hat, sparsity, concepts):
    return F.mse_loss(x,x_hat) + sparsity * torch.abs(concepts).sum()

def robustness_loss(x, parameters, model):
    return torch.tensor(0)