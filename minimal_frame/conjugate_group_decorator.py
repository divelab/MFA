from minimal_frame.generalized_qr import *
from functools import wraps

def od_conjugate_equivariant_decorator(forward_func):
    """
        O(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        output = Q.T @ output @ Q
        return output

    return wrapper


def sod_conjugate_equivariant_decorator(forward_func):
    """
        SO(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        if int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        output = Q.T @ output @ Q
        return output

    return wrapper