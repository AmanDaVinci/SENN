import torch


def jacobian(f, x, out_dim):
    """Computes Jacobian of a vector-valued function.

    Given f: batch_size x input_dim -> batch_size x output_dim
    Computes Jacobian Matrix J: batch_size x output_dim x input_dim
    By passing a gradient matrix to the backward call of the output tensor

    NOTE:
    This implementation only works for two dimensional f and x
    If not true, it proceeds with the assumption that the third dimension is 1

    Parameters
    ----------
    f : torch.nn.Module
        Function as a pytorch module. Jacobian will be computed against this.
    x : torch.tensor
        Input data tensor of shape (batch_size x input_dim).
        This object is not mutated and a clone is used for the forward pass.
    out_dim : int
        outer dimension of the function f

    Returns
    -------
    jacobian_fx : torch.tensor
        jacobian as a tensor of shape (batch_size x output_dim x input_dim)
    """
    model_input = x.clone().detach()
    bsize = model_input.size()[0]
    # (bs, in_dim) --repeated--> (bs, out_dim, in_dim)
    model_input = model_input.unsqueeze(1).repeat(1, out_dim, 1)
    model_input.requires_grad_(True)
    # can only compute Jacobian of inputs and outputs with 2 dimensions
    out = f(model_input).reshape(bsize, out_dim, out_dim)
    # for autograd of non-scalar outputs
    grad_matrix = torch.eye(out_dim).reshape(1, out_dim, out_dim).repeat(bsize, 1, 1)
    out.backward(grad_matrix)
    return model_input.grad.data
