def jacobian(f, x, out_dim):
    """Computes Jacobian of a vector-valued function.

    Given f: batch_size x input_dim -> batch_size x output_dim
    Computes Jacobian Matrix J: batch_size x output_dim x input_dim
    By passing a gradient matrix to the backward call of the output tensor

    Parameters
    ----------
    f : torch.nn.Module
        Function as a pytorch module. Jacobian will be computed against this. 
    x : torch.Tensor
        Input data tensor of shape (BATCH, *). Only restriction on the shape is that
        the first dimension should correspond to the batch size.
    out_dim : int
        outer dimension of the function f 

    Returns
    -------
    jacobian_fx : torch.Tensor
        jacobian as a tensor of shape (batch_size x output_dim x input_dim)
    """
    bsize = x.size()[0]
    # (bs, in_dim) --repeated--> (bs, out_dim, in_dim)
    x = x.unsqueeze(1).repeat(1, out_dim, 1)
    x.requires_grad_(True)
    y = f(x)
    # for autograd of non-scalar outputs
    grad_matrix = torch.eye(out_dim).reshape(1,out_dim, out_dim).repeat(bsize, 1, 1)
    y.backward(grad_matrix)
    return x.grad.data