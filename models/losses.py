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
    batch_size = x.size(0)
    num_features = x.size(1)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(x.device)
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
    # concept_dim is always 1
    concepts = concepts.squeeze(-1)
    aggregates = aggregates.squeeze(-1)

    batch_size = x.size(0)
    num_concepts = concepts.size(1)
    num_classes = aggregates.size(1)

    # Jacobian of aggregates wrt x
    jacobians = []
    for i in range(num_classes):    
        grad_tensor = torch.zeros(batch_size, num_classes).to(x.device)
        grad_tensor[:,i] = 1.
        j_yx = torch.autograd.grad(outputs=aggregates, inputs=x, \
        grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_yx.view(batch_size,-1).unsqueeze(-1))
    # bs x num_features x num_classes (bs x 784 x 10)
    J_yx = torch.cat(jacobians, dim=2)

    # Jacobian of concepts wrt x
    jacobians = []
    for i in range(num_concepts):    
        grad_tensor = torch.zeros(batch_size, num_concepts).to(x.device)
        grad_tensor[:,i] = 1.
        j_hx = torch.autograd.grad(outputs=concepts, inputs=x, \
        grad_outputs=grad_tensor, create_graph=True, only_inputs=True)[0]
        # bs x 1 x 28 x 28 -> bs x 784 x 1
        jacobians.append(j_hx.view(batch_size,-1).unsqueeze(-1))
    # bs x num_features x num_concepts
    J_hx = torch.cat(jacobians, dim=2)

    # bs x num_features x num_classes
    robustness_loss = J_yx - torch.bmm(J_hx, relevances)

    return robustness_loss.norm(p='fro')

def weighted_mse(x, x_hat, sparsity_reg):
    return sparsity_reg * F.mse_loss(x_hat,x)

def mse_kl_sparsity(x, x_hat, sparsity_reg, concepts):
    return F.mse_loss(x_hat,x) + F.kl_div(sparsity_reg*torch.ones_like(concepts), concepts)

def mse_l1_sparsity(x, x_hat, sparsity_reg, concepts):
    return F.mse_loss(x_hat,x) + sparsity_reg * torch.abs(concepts).sum()

def robustness_loss(x, parameters, model):
    return torch.tensor(0)