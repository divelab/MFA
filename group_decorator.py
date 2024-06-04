"""
Group Decorator
============

Description:
------------
Equivariant/Invarirant decorator to random neural network models with input and output shape $n\times d$

Author:
-------
Yuchao Lin

"""
import random
from itertools import product, permutations

import torch

from generalized_qr import *
from canonical_labeling import *
from functools import wraps


def od_equivariant_decorator(forward_func):
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
        output = output @ Q.T
        return output

    return wrapper


def od_invariant_decorator(forward_func):
    """
        O(d)-invariant decorator

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
        return output

    return wrapper


def od_equivariant_puny_decorator(forward_func):
    """
        O(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        L, Q = torch.linalg.eigh((x.T @ x).to(torch.float64))
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        output = sum([forward_func(x @ m, *args, **kwargs) @ m.T for m in matrices]) / len(matrices)
        return output

    return wrapper


def od_invariant_puny_decorator(forward_func):
    """
        O(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        L, Q = torch.linalg.eigh((x.T @ x).to(torch.float64))
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        output = sum([forward_func(x @ m, *args, **kwargs) for m in matrices]) / len(matrices)
        return output

    return wrapper


def sum_algebraic_multiplicities(matrix):
    """
        Compute multiplicities (minus one) of degenerate nonzero eigenvalues

        Args:
            matrix (float32, float64): Degenerate data with shape $n\times d$
        """
    eigenvalues, _ = np.linalg.eig(matrix)
    unique, counts = np.unique(np.round(eigenvalues, decimals=5), return_counts=True)
    sum_multiplicities = np.sum(counts[(counts > 1) & (unique != 0)] - 1.0)
    return sum_multiplicities


def eigenvalue_perturbation(P, epsilon=0.1, epsilon_start=0.1, l_submax=10, tol=1e-5):
    """
        Eigenvalue first-order perturbation

        Args:
            P (float32, float64): Degenerate data with shape $n\times d$
            epsilon (float32, float64): Perturbation increment
            epsilon_start (float32, float64): Perturbation start
            l_submax (float32, float64): Maximum number of sub-iterations
            tol (float32, float64): Tolerance for determining the degenerate values

        """
    z = np.ones(P.shape[0])
    eigvals, eigvecs = np.linalg.eigh(P.T @ P)
    unique_eigvals, counts = np.unique(np.round(eigvals, decimals=5), return_counts=True)
    degenerate_indices = np.where(counts > 1)[0]
    k_set = set()
    for degenerate_index in degenerate_indices:
        degenerate_eigval = unique_eigvals[degenerate_index]
        if degenerate_eigval < 1e-5:
            continue
        indices = np.where(np.abs(eigvals - degenerate_eigval) < tol)[0]
        m = len(indices)
        if m == 1:
            continue
        V = eigvecs[:, indices]
        orthogonal_vectors = \
            modified_gram_schmidt(find_independent_vectors_cuda(torch.DoubleTensor(P @ V @ V.T)).transpose(0, 1),
                                  torch.eye(P.shape[1]).to(torch.float64))[0].numpy()
        P_proj = P @ orthogonal_vectors[:, :m]

        for l in range(m - 1):
            for k in range(len(z)):
                zl = np.zeros_like(z)
                if k in k_set:
                    continue

                u = P_proj[k]
                u_norm = np.sum(u ** 2)

                if abs(u_norm) < 1e-8:
                    continue
                else:
                    k_set.add(k)

                al = sum_algebraic_multiplicities(P.T @ P)
                if al == 0.0:
                    break
                eps = epsilon_start
                for _ in range(l_submax):
                    zl[k] = eps * degenerate_eigval / u_norm
                    if sum_algebraic_multiplicities(P.T @ np.diag(1.0 + zl) @ P) < al:
                        break
                    eps += epsilon
                P = np.diag(np.sqrt(1 + zl)) @ P
                z *= (1 + zl)
                break

    z -= 1
    return z


def od_equivariant_puny_improve_decorator(forward_func):
    """
        O(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        P = x.detach().cpu().numpy().astype(np.float64)
        A = P.T @ P
        if sum_algebraic_multiplicities(A) > 0:
            z = eigenvalue_perturbation(P)
            A = P.T @ np.diag(1.0 + z) @ P
        A = torch.DoubleTensor(A).to(x.device)
        L, Q = torch.linalg.eigh(A)
        Q[:, L < 1e-5] = 0.0
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        output = sum([forward_func(x @ m, *args, **kwargs) @ m.T for m in matrices]) / len(matrices)
        return output

    return wrapper


def od_invariant_puny_improve_decorator(forward_func):
    """
        O(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        P = x.detach().cpu().numpy().astype(np.float64)
        A = P.T @ P
        if sum_algebraic_multiplicities(A) > 0:
            z = eigenvalue_perturbation(P)
            A = P.T @ np.diag(1.0 + z) @ P
        A = torch.DoubleTensor(A).to(x.device)
        L, Q = torch.linalg.eigh(A)
        Q[:, L < 1e-5] = 0.0
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        output = sum([forward_func(x @ m, *args, **kwargs) for m in matrices]) / len(matrices)
        return output

    return wrapper


def od_equivariant_sfa_decorator(forward_func):
    """
        O(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        L, Q = torch.linalg.eigh((x.T @ x).to(torch.float64))
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        m = random.choice(matrices)
        output = forward_func(x @ m, *args, **kwargs) @ m.T
        return output

    return wrapper


def od_invariant_sfa_decorator(forward_func):
    """
        O(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        L, Q = torch.linalg.eigh((x.T @ x).to(torch.float64))
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d))).to(torch.float64)
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix.to(x.dtype))
        m = random.choice(matrices)
        output = forward_func(x @ m, *args, **kwargs)
        return output

    return wrapper


def sod_equivariant_decorator(forward_func):
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
        output = output @ Q.T
        return output

    return wrapper


def sod_invariant_decorator(forward_func):
    """
        SO(d)-invariant decorator

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
        return output

    return wrapper


def ed_equivariant_decorator(forward_func):
    """
        E(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        centroid = x.mean(0, keepdim=True)
        x = x - centroid
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        output = output @ Q.T + centroid
        return output

    return wrapper


def ed_invariant_decorator(forward_func):
    """
        E(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        centroid = x.mean(0, keepdim=True)
        x = x - centroid
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        return output

    return wrapper


def sed_equivariant_decorator(forward_func):
    """
        SE(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        centroid = x.mean(0, keepdim=True)
        x = x - centroid
        if int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        output = output @ Q.T + centroid
        return output

    return wrapper


def sed_invariant_decorator(forward_func):
    """
        SE(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        centroid = x.mean(0, keepdim=True)
        x = x - centroid
        if int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x.to(torch.float64), torch.eye(x.size(1)).to(torch.float64).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q = Q.to(x.dtype)
        output = forward_func(x @ Q, *args, **kwargs)
        return output

    return wrapper


def o1d_equivariant_decorator(forward_func):
    """
        O(1,d-1)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(torch.float64).to(x.device)
        non_null_indices = torch.abs(torch.diag(x.to(torch.float64) @ eta @ x.T.to(torch.float64))) > 1e-2
        Q, _ = generalized_qr_decomposition(x[non_null_indices].to(torch.float64), eta)
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv.to(x.dtype), *args, **kwargs)
        output = output @ Q.T.to(x.dtype)
        return output

    return wrapper


def o1d_invariant_decorator(forward_func):
    """
        O(1,d-1)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(torch.float64).to(x.device)
        non_null_indices = torch.abs(torch.diag(x.to(torch.float64) @ eta @ x.T.to(torch.float64))) > 1e-2
        Q, _ = generalized_qr_decomposition(x[non_null_indices].to(torch.float64), eta)
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv.to(x.dtype), *args, **kwargs)
        return output

    return wrapper


def so1d_equivariant_decorator(forward_func):
    """
        SO(1,d-1)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(torch.float64).to(x.device)
        non_null_indices = torch.abs(torch.diag(x.to(torch.float64) @ eta @ x.T.to(torch.float64))) > 1e-2
        non_null_x = x[non_null_indices]
        if int(torch.linalg.matrix_rank(non_null_x.to(torch.float32))) == non_null_x.size(1) - 1:
            while int(torch.linalg.matrix_rank(non_null_x.to(torch.float32))) == non_null_x.size(1) - 1:
                non_null_x = torch.cat([non_null_x, torch.rand(non_null_x.size(1)).unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(non_null_x.to(torch.float64), eta)
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv.to(x.dtype), *args, **kwargs)
        output = output @ Q.T.to(x.dtype)
        return output

    return wrapper


def so1d_invariant_decorator(forward_func):
    """
        SO(1,d-1)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(torch.float64).to(x.device)
        non_null_indices = torch.abs(torch.diag(x.to(torch.float64) @ eta @ x.T.to(torch.float64))) > 1e-2
        non_null_x = x[non_null_indices]
        if int(torch.linalg.matrix_rank(non_null_x.to(torch.float32))) == non_null_x.size(1) - 1:
            while int(torch.linalg.matrix_rank(non_null_x.to(torch.float32))) == non_null_x.size(1) - 1:
                non_null_x = torch.cat([non_null_x, torch.rand(non_null_x.size(1)).unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(non_null_x.to(torch.float64), eta)
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv.to(x.dtype), *args, **kwargs)
        return output

    return wrapper


def ud_equivariant_decorator(forward_func):
    """
        U(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (complex64, complex128): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        Q, _ = qr_decomposition_complex(x.to(torch.complex128))
        Q = Q.to(x.dtype)
        r = int(torch.linalg.matrix_rank(x.to(torch.complex64)))
        if r < x.size(1):
            Q[:, :r] = 0.0 + 0.0j
        output = forward_func(x @ Q, *args, **kwargs)
        output = output @ Q.H
        return output

    return wrapper


def ud_invariant_decorator(forward_func):
    """
        U(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (complex64, complex128): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        Q, _ = qr_decomposition_complex(x.to(torch.complex128))
        Q = Q.to(x.dtype)
        r = int(torch.linalg.matrix_rank(x.to(torch.complex64)))
        if r < x.size(1):
            Q[:, :r] = 0.0 + 0.0j
        output = forward_func(x @ Q, *args, **kwargs)
        return output

    return wrapper


def sud_equivariant_decorator(forward_func):
    """
        SU(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (complex64, complex128): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        if int(torch.linalg.matrix_rank(x.to(torch.complex64))) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x.to(torch.complex64))) == x.size(1) - 1:
                real_part = torch.randn(x.size(1))
                imaginary_part = torch.randn(x.size(1))
                complex_vector = real_part + 1j * imaginary_part
                x = torch.cat([x, complex_vector.unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = qr_decomposition_complex(x.to(torch.complex128))
        det = np.linalg.det(Q)
        Q = Q / (det ** (1 / d))
        Q = Q.to(x.dtype)
        r = int(torch.linalg.matrix_rank(x.to(torch.complex64)))
        if r < x.size(1):
            Q[:, :r] = 0.0 + 0.0j
        output = forward_func(x @ Q, *args, **kwargs)
        output = output @ Q.H
        return output

    return wrapper


def sud_invariant_decorator(forward_func):
    """
        SU(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (complex64, complex128): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        if int(torch.linalg.matrix_rank(x.to(torch.complex64))) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x.to(torch.complex64))) == x.size(1) - 1:
                real_part = torch.randn(x.size(1))
                imaginary_part = torch.randn(x.size(1))
                complex_vector = real_part + 1j * imaginary_part
                x = torch.cat([x, complex_vector.unsqueeze(0).to(x.dtype).to(x.device)], -1)
        Q, _ = qr_decomposition_complex(x.to(torch.complex128))
        det = np.linalg.det(Q)
        Q = Q / (det ** (1 / d))
        Q = Q.to(x.dtype)
        r = int(torch.linalg.matrix_rank(x.to(torch.complex64)))
        if r < x.size(1):
            Q[:, :r] = 0.0 + 0.0j
        output = forward_func(x @ Q, *args, **kwargs)
        return output

    return wrapper


def gld_equivariant_decorator(forward_func):
    """
        GL(d, \mathbb{R})-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        assert int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-5, "Input requires full column rank"
        output = forward_func(torch.linalg.solve(phi_A.to(torch.float64).T, x.to(torch.float64).T).T.to(x.dtype), *args, **kwargs)
        output = output @ phi_A
        return output

    return wrapper


def gld_invariant_decorator(forward_func):
    """
        GL(d, \mathbb{R})-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        assert int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-5, "Input requires full column rank"
        output = forward_func(torch.linalg.solve(phi_A.to(torch.float64).T, x.to(torch.float64).T).T.to(x.dtype), *args, **kwargs)
        return output

    return wrapper


def sld_equivariant_decorator(forward_func):
    """
        SL(d, \mathbb{R})-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        assert int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1)
        phi_A = find_independent_vectors_cuda(x).to(torch.float64)
        assert torch.abs(torch.det(phi_A)) > 1e-5, "Input requires full column rank"
        sign = torch.sign(torch.det(phi_A))
        coeff = sign * torch.pow(torch.abs(torch.det(phi_A)), 1.0 / phi_A.size(1))
        output = forward_func((coeff * torch.linalg.solve(phi_A.to(torch.float64).T, x.to(torch.float64).T).T).to(x.dtype), *args, **kwargs)
        output = output @ (phi_A * 1.0 / coeff).to(x.dtype)
        return output

    return wrapper


def sld_invariant_decorator(forward_func):
    """
        SL(d, \mathbb{R})-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        assert int(torch.linalg.matrix_rank(x.to(torch.float32))) == x.size(1)
        phi_A = find_independent_vectors_cuda(x).to(torch.float64)
        assert torch.abs(torch.det(phi_A)) > 1e-5, "Input requires full column rank"
        sign = torch.sign(torch.det(phi_A))
        coeff = sign * torch.pow(torch.abs(torch.det(phi_A)), 1.0 / phi_A.size(1))
        output = forward_func((coeff * torch.linalg.solve(phi_A.to(torch.float64).T, x.to(torch.float64).T).T).to(x.dtype), *args, **kwargs)
        return output

    return wrapper


def sn_equivariant_decorator(forward_func):
    """
        S_n-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        sorted_indices = torch.argsort(x, dim=0)[:, 0]
        sorted_matrix = x[sorted_indices]
        unique_rows, inverse_indices = torch.unique(sorted_matrix, return_inverse=True, dim=0)
        grouped_indices = [[] for _ in range(unique_rows.size(0))]
        for idx, group in enumerate(inverse_indices):
            grouped_indices[group].append(sorted_indices[idx].item())

        group_permutations = []
        for group in grouped_indices:
            if len(group) > 1:
                group_permutations.append(list(permutations(group)))
            else:
                group_permutations.append([tuple(group)])

        Ps = []
        for perm_combination in product(*group_permutations):
            perm = [idx for group_perm in perm_combination for idx in group_perm]
            Ps.append(torch.FloatTensor(permutation_array_to_matrix(perm)).to(x.dtype).to(x.device))
        output = sum([P.T @ forward_func(P @ x, *args, **kwargs) for P in Ps]) / len(Ps)
        return output

    return wrapper


def sn_invariant_decorator(forward_func):
    """
        S_n-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        sorted_indices = torch.argsort(x, dim=0)[:, 0]
        sorted_matrix = x[sorted_indices]
        unique_rows, inverse_indices = torch.unique(sorted_matrix, return_inverse=True, dim=0)
        grouped_indices = [[] for _ in range(unique_rows.size(0))]
        for idx, group in enumerate(inverse_indices):
            grouped_indices[group].append(sorted_indices[idx].item())

        group_permutations = []
        for group in grouped_indices:
            if len(group) > 1:
                group_permutations.append(list(permutations(group)))
            else:
                group_permutations.append([tuple(group)])

        Ps = []
        for perm_combination in product(*group_permutations):
            perm = [idx for group_perm in perm_combination for idx in group_perm]
            Ps.append(torch.FloatTensor(permutation_array_to_matrix(perm)).to(x.dtype).to(x.device))
        output = sum([forward_func(P @ x, *args, **kwargs) for P in Ps]) / len(Ps)
        return output

    return wrapper


def sn_od_equivariant_decorator(forward_func):
    """
        S_n \times O(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=3)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.dtype).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        eta = torch.eye(x.size(1)).to(torch.float64).to(x.device)
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta)[0].to(x.dtype) for transformed_x in transformed_xs]
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Qs[i], *args, **kwargs) @ Qs[i].T for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper


def sn_od_invariant_decorator(forward_func):
    """
        S_n \times O(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=3)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.dtype).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        eta = torch.eye(x.size(1)).to(torch.float64).to(x.device)
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta)[0].to(x.dtype) for transformed_x in transformed_xs]
        output = sum([forward_func(Ps[i] @ x @ Qs[i], *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper


def sn_sod_equivariant_decorator(forward_func):
    """
        S_n \times SO(d)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=3)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        for i, transformed_x in enumerate(transformed_xs):
            if int(torch.linalg.matrix_rank(transformed_x.to(torch.float32))) == transformed_x.size(1) - 1:
                while int(torch.linalg.matrix_rank(transformed_x.to(torch.float32))) == transformed_x.size(1) - 1:
                    transformed_xs[i] = torch.cat(
                        [transformed_x, torch.rand(transformed_x.size(1)).unsqueeze(0).to(transformed_x.device)], -1)

        eta = torch.eye(x.size(1)).to(torch.float64).to(x.device)
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta)[0].to(x.dtype) for transformed_x in transformed_xs]
        for i, Q in enumerate(Qs):
            if torch.linalg.det(Q) < 0:
                Q[:, -1] *= -1
                Qs[i] = Q
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Qs[i], *args, **kwargs) @ Qs[i].T for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper


def sn_sod_invariant_decorator(forward_func):
    """
        S_n \times SO(d)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=3)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        for i, transformed_x in enumerate(transformed_xs):
            if int(torch.linalg.matrix_rank(transformed_x.to(torch.float32))) == transformed_x.size(1) - 1:
                while int(torch.linalg.matrix_rank(transformed_x.to(torch.float32))) == transformed_x.size(1) - 1:
                    transformed_xs[i] = torch.cat(
                        [transformed_x, torch.rand(transformed_x.size(1)).unsqueeze(0).to(transformed_x.device)], -1)

        eta = torch.eye(x.size(1)).to(torch.float64).to(x.device)
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta)[0].to(x.dtype) for transformed_x in transformed_xs]
        for i, Q in enumerate(Qs):
            if torch.linalg.det(Q) < 0:
                Q[:, -1] *= -1
                Qs[i] = Q
        output = sum([forward_func(Ps[i] @ x @ Qs[i], *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper


def sn_o1d_equivariant_decorator(forward_func):
    """
        S_n \times O(1,d-1)-equivariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(x.device)
        phi = (x @ eta @ x.T).detach().cpu().numpy()
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        transformed_xs = [transformed_x[torch.abs(torch.diag(transformed_x @ eta @ transformed_x.T)) > 1e-2] for
                          transformed_x in transformed_xs]
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta.to(torch.float64))[0] for transformed_x in transformed_xs]
        Q_invs = [eta.to(torch.float64) @ Q @ eta.to(torch.float64) for Q in Qs]
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Q_invs[i].to(x.dtype), *args, **kwargs) @ Qs[i].to(x.dtype).T for i in range(len(Ps))]) / len(
            Ps)
        return output

    return wrapper


def sn_o1d_invariant_decorator(forward_func):
    """
        S_n \times O(1,d-1)-invariant decorator

        Args (wrapping forward function in PyTorch nn.Module):
            x (float32, float64): The input data with shape $n\times d$
            *args, **kwargs: Other arguments in wrapped model's forward function

        """

    @wraps(forward_func)
    def wrapper(x, *args, **kwargs):
        assert x.dim() == 2
        n, d = x.size()
        eta = -torch.eye(x.size(1))
        eta[0, 0] = 1.0
        eta = eta.to(x.device)
        phi = (x @ eta @ x.T).detach().cpu().numpy()
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        transformed_xs = [transformed_x[torch.abs(torch.diag(transformed_x @ eta @ transformed_x.T)) > 1e-2] for
                          transformed_x in transformed_xs]
        Qs = [generalized_qr_decomposition(transformed_x.to(torch.float64), eta.to(torch.float64))[0] for transformed_x in transformed_xs]
        Q_invs = [eta.to(torch.float64) @ Q @ eta.to(torch.float64) for Q in Qs]
        output = sum([forward_func(Ps[i] @ x @ Q_invs[i].to(x.dtype), *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper
