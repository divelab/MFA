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
from itertools import product
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
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
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
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
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
        L, Q = torch.linalg.eigh(x.T @ x)
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d)))
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix)
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
        L, Q = torch.linalg.eigh(x.T @ x)
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d)))
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix)
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
        if degenerate_eigval == 0:
            continue
        indices = np.where(np.abs(eigvals - degenerate_eigval) < tol)[0]
        m = len(indices)
        if m == 1:
            continue
        V = eigvecs[:, indices]
        orthogonal_vectors = \
            modified_gram_schmidt(find_independent_vectors_cuda(torch.FloatTensor(P @ V @ V.T)).transpose(0, 1),
                                  torch.eye(P.shape[1]).astype(P.dtype))[0].numpy()
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
        P = x.detach().cpu().numpy()
        A = P.T @ P
        if sum_algebraic_multiplicities(A) > 0:
            z = eigenvalue_perturbation(P)
            A = P.T @ np.diag(1.0 + z) @ P
        A = torch.FloatTensor(A).to(x.device)
        L, Q = torch.linalg.eigh(A)
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d)))
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix)
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
        P = x.detach().cpu().numpy()
        A = P.T @ P
        if sum_algebraic_multiplicities(A) > 0:
            z = eigenvalue_perturbation(P)
            A = P.T @ np.diag(1.0 + z) @ P
        A = torch.FloatTensor(A).to(x.device)
        L, Q = torch.linalg.eigh(A)
        sign_combinations = torch.tensor(list(product([-1, 1], repeat=d)))
        matrices = []
        for signs in sign_combinations:
            new_matrix = Q * signs.unsqueeze(0)
            matrices.append(new_matrix)
        output = sum([forward_func(x @ m, *args, **kwargs) for m in matrices]) / len(matrices)
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
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
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
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
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                x = torch.cat([x, torch.rand(x.size(1)).unsqueeze(0).to(x.device)], -1)
        Q, _ = generalized_qr_decomposition(x, torch.eye(x.size(1)).to(x.device))
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
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
        eta = eta.to(x.device)
        non_null_indices = torch.abs(torch.diag(x @ eta @ x.T)) > 1e-3
        Q, _ = generalized_qr_decomposition(x[non_null_indices], eta)
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv, *args, **kwargs)
        output = output @ Q.T
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
        eta = eta.to(x.device)
        non_null_indices = torch.abs(torch.diag(x @ eta @ x.T)) > 1e-3
        Q, _ = generalized_qr_decomposition(x[non_null_indices], eta)
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv, *args, **kwargs)
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
        eta = eta.to(x.device)
        non_null_indices = torch.abs(torch.diag(x @ eta @ x.T)) > 1e-3
        non_null_x = x[non_null_indices]
        if int(torch.linalg.matrix_rank(non_null_x)) == non_null_x.size(1) - 1:
            while int(torch.linalg.matrix_rank(non_null_x)) == non_null_x.size(1) - 1:
                non_null_x = torch.cat([non_null_x, torch.rand(non_null_x.size(1)).unsqueeze(0).to(non_null_x.device)],
                                       -1)
        Q, _ = generalized_qr_decomposition(non_null_x, eta)
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv, *args, **kwargs)
        output = output @ Q.T
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
        eta = eta.to(x.device)
        non_null_indices = torch.abs(torch.diag(x @ eta @ x.T)) > 1e-3
        non_null_x = x[non_null_indices]
        if int(torch.linalg.matrix_rank(non_null_x)) == non_null_x.size(1) - 1:
            while int(torch.linalg.matrix_rank(non_null_x)) == non_null_x.size(1) - 1:
                non_null_x = torch.cat([non_null_x, torch.rand(non_null_x.size(1)).unsqueeze(0).to(non_null_x.device)],
                                       -1)
        Q, _ = generalized_qr_decomposition(non_null_x, eta)
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
        Q_inv = eta @ Q @ eta
        output = forward_func(x @ Q_inv, *args, **kwargs)
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
        Q, _ = qr_decomposition_complex(x)
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
        Q, _ = qr_decomposition_complex(x)
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                real_part = torch.randn(x.size(1))
                imaginary_part = torch.randn(x.size(1))
                complex_vector = real_part + 1j * imaginary_part
                x = torch.cat([x, complex_vector.unsqueeze(0).to(x.device)], -1)
        Q, _ = qr_decomposition_complex(x)
        det = np.linalg.det(Q)
        Q = Q / (det ** (1 / d))
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
        if int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
            while int(torch.linalg.matrix_rank(x)) == x.size(1) - 1:
                real_part = torch.randn(x.size(1))
                imaginary_part = torch.randn(x.size(1))
                complex_vector = real_part + 1j * imaginary_part
                x = torch.cat([x, complex_vector.unsqueeze(0).to(x.device)], -1)
        Q, _ = qr_decomposition_complex(x)
        det = np.linalg.det(Q)
        Q = Q / (det ** (1 / d))
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
        assert int(torch.linalg.matrix_rank(x)) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-4
        output = forward_func(x @ torch.inverse(phi_A), *args, **kwargs)
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
        assert int(torch.linalg.matrix_rank(x)) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-4
        output = forward_func(x @ torch.inverse(phi_A), *args, **kwargs)
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
        assert int(torch.linalg.matrix_rank(x)) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-4
        sign = torch.sign(torch.det(phi_A))
        frame_inv = sign * torch.pow(torch.abs(torch.det(phi_A)), 1.0 / phi_A.size(1)) * torch.inverse(phi_A)
        output = forward_func(x @ frame_inv, *args, **kwargs)
        output = output @ torch.inverse(frame_inv)
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
        assert int(torch.linalg.matrix_rank(x)) == x.size(1)
        phi_A = find_independent_vectors_cuda(x)
        assert torch.abs(torch.det(phi_A)) > 1e-4
        sign = torch.sign(torch.det(phi_A))
        frame_inv = sign * torch.pow(torch.abs(torch.det(phi_A)), 1.0 / phi_A.size(1)) * torch.inverse(phi_A)
        output = forward_func(x @ frame_inv, *args, **kwargs)
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
        n, d = x.size()
        phi = (x @ x.T).detach().cpu().numpy()
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
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
        n, d = x.size()
        phi = (x @ x.T).detach().cpu().numpy()
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
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
        eta = torch.eye(x.size(1)).to(x.device)
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        Qs = [generalized_qr_decomposition(transformed_x, eta)[0] for transformed_x in transformed_xs]
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
        eta = torch.eye(x.size(1)).to(x.device)
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        Qs = [generalized_qr_decomposition(transformed_x, eta)[0] for transformed_x in transformed_xs]
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
        eta = torch.eye(x.size(1)).to(x.device)
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        for i, transformed_x in enumerate(transformed_xs):
            if int(torch.linalg.matrix_rank(transformed_x)) == transformed_x.size(1) - 1:
                while int(torch.linalg.matrix_rank(transformed_x)) == transformed_x.size(1) - 1:
                    transformed_xs[i] = torch.cat(
                        [transformed_x, torch.rand(transformed_x.size(1)).unsqueeze(0).to(transformed_x.device)], -1)
        Qs = [generalized_qr_decomposition(transformed_x, eta)[0] for transformed_x in transformed_xs]
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
        eta = torch.eye(x.size(1)).to(x.device)
        phi = np.round((x @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        for i, transformed_x in enumerate(transformed_xs):
            if int(torch.linalg.matrix_rank(transformed_x)) == transformed_x.size(1) - 1:
                while int(torch.linalg.matrix_rank(transformed_x)) == transformed_x.size(1) - 1:
                    transformed_xs[i] = torch.cat(
                        [transformed_x, torch.rand(transformed_x.size(1)).unsqueeze(0).to(transformed_x.device)], -1)
        Qs = [generalized_qr_decomposition(transformed_x, eta)[0] for transformed_x in transformed_xs]
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
        phi = np.round((x @ eta @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        transformed_xs = [transformed_x[torch.abs(torch.diag(transformed_x @ eta @ transformed_x.T)) > 1e-3] for
                          transformed_x in transformed_xs]
        Qs = [generalized_qr_decomposition(transformed_x, eta)[0] for transformed_x in transformed_xs]
        Q_invs = [eta @ Q @ eta for Q in Qs]
        output = sum(
            [Ps[i].T @ forward_func(Ps[i] @ x @ Q_invs[i], *args, **kwargs) @ Qs[i].T for i in range(len(Ps))]) / len(
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
        phi = np.round((x @ eta @ x.T).detach().cpu().numpy(), decimals=2)
        iu = np.triu_indices(x.size(0), 1)
        edge_index = np.vstack(iu).T
        edge_attr = phi[iu]
        frames = generate_permutation_frames((np.zeros((n, 1)), edge_index, edge_attr))
        Ps = [torch.FloatTensor(permutation_array_to_matrix(list(frame)[:n])).to(x.device) for frame in frames]
        transformed_xs = [P @ x for P in Ps]
        transformed_xs = [transformed_x[torch.abs(torch.diag(transformed_x @ eta @ transformed_x.T)) > 1e-3] for
                          transformed_x in transformed_xs]
        Qs = [generalized_qr_decomposition(transformed_x, torch.eye(transformed_x.size(1)).to(transformed_x.device))[0]
              for transformed_x in transformed_xs]
        Q_invs = [eta @ Q @ eta for Q in Qs]
        output = sum([forward_func(Ps[i] @ x @ Q_invs[i], *args, **kwargs) for i in range(len(Ps))]) / len(Ps)
        return output

    return wrapper