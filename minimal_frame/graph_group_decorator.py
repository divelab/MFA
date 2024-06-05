"""
Group Decorator
============

Description:
------------
Equivariant/Invarirant decorator to random neural network models with undirected adjacency matrix as input

Author:
-------
Yuchao Lin

"""
import torch
from minimal_frame.canonical_labeling import *
from functools import wraps



def undirected_unweighted_sn_equivariant_decorator(forward_func):
    """
        S_n-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input adjacency matrix with shape $n\times n$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2 and torch.all(x == x.transpose(0, 1))
        n, _ = x.size()
        node_attr = x.diag()
        edge_index = (x != 0).nonzero(as_tuple=False)
        frames = generate_permutation_frames((node_attr.detach().cpu().numpy(), edge_index.detach().cpu().numpy(), None), weighted=False)
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame))).to(x.dtype).to(x.device) for frame in frames]
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Ps[i].T, *args, **kwargs) @ Ps[i] for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper

def undirected_unweighted_sn_invariant_decorator(forward_func):
    """
        S_n-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input adjacency matrix with shape $n\times n$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2 and torch.all(x == x.transpose(0, 1))
        n, _ = x.size()
        node_attr = x.diag()
        edge_index = (x != 0).nonzero(as_tuple=False)
        frames = generate_permutation_frames((node_attr.detach().cpu().numpy(), edge_index.detach().cpu().numpy(), None), weighted=False)
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame))).to(x.dtype).to(x.device) for frame in frames]
        output = sum(
            [forward_func(Ps[i] @ x @ Ps[i].T, *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper


def undirected_weighted_sn_equivariant_decorator(forward_func):
    """
        S_n-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input adjacency matrix with shape $n\times n$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2 and torch.all(x == x.transpose(0, 1))
        n, _ = x.size()
        node_attr = x.diag()
        edge_index = (x != 0).nonzero(as_tuple=False)
        edge_attr = x[x != 0]
        frames = generate_permutation_frames((node_attr.detach().cpu().numpy(), edge_index.detach().cpu().numpy(), edge_attr.detach().cpu().numpy()), maximum_node=n)
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame))).to(x.dtype).to(x.device) for frame in frames]
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Ps[i].T, *args, **kwargs) @ Ps[i] for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper

def undirected_weighted_sn_invariant_decorator(forward_func):
    """
        S_n-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input adjacency matrix with shape $n\times n$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2 and torch.all(x == x.transpose(0, 1))
        n, _ = x.size()
        node_attr = x.diag()
        edge_index = (x != 0).nonzero(as_tuple=False)
        edge_attr = x[x != 0]
        frames = generate_permutation_frames((node_attr.detach().cpu().numpy(), edge_index.detach().cpu().numpy(), edge_attr.detach().cpu().numpy()), maximum_node=n)
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame))).to(x.dtype).to(x.device) for frame in frames]
        output = sum(
            [forward_func(Ps[i] @ x @ Ps[i].T, *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper