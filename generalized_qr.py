"""
Generalized QR Decomposition in PyTorchr
============

Description:
------------

Author:
-------
Yuchao Lin

"""
import torch


@torch.jit.script
def find_independent_vectors_cuda(P):
    n = P.size(0)
    r = int(torch.linalg.matrix_rank(P))

    indices = torch.arange(r)
    done = False
    while not done:
        subset = P[indices, :]
        if torch.linalg.matrix_rank(subset) == torch.linalg.matrix_rank(P):
            return subset
        done = True
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                done = False
                break

    return None

@torch.jit.script
def qr_decomposition(coords):
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.transpose(0, 1), mode='complete')
    for j in range(R.size(1)):
        if R[j, j] < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def qr_decomposition_complex(coords):
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.H, mode='complete')
    for j in range(R.size(1)):
        if R[j, j].real < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def inner_product(u, v, eta):
    return torch.dot(u, torch.mv(eta, v))


@torch.jit.script
def project(u, v, eta):
    norm_sq = inner_product(u, u, eta)
    assert norm_sq != 0.0
    coeff = inner_product(u, v, eta) / norm_sq
    return coeff * u


@torch.jit.script
def gram_schmidt(A, eta):
    m, n = A.size()
    metric_length = eta.size(0)
    Q = torch.zeros((m, metric_length), dtype=A.dtype).to(A.device)
    R = torch.zeros((metric_length, n), dtype=A.dtype).to(A.device)
    eta_c = torch.zeros(metric_length, dtype=A.dtype).to(A.device)

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            v -= project(Q[:, i], A[:, j], eta)
        norm_sq = inner_product(v, v, eta)
        norm_sq = torch.sqrt(torch.abs(norm_sq))
        assert norm_sq != 0.0
        Q[:, j] = v / norm_sq
        Rjj = inner_product(Q[:, j], A[:, j], eta)
        if Rjj < 0:
            Q[:, j] = -Q[:, j]
            Rjj = -Rjj
        R[j, j] = Rjj
        eta_c[j] = torch.sign(inner_product(Q[:, j], Q[:, j], eta))
        for i in range(j):
            R[i, j] = inner_product(Q[:, i], A[:, j], eta)

    return Q, eta_c, R


@torch.jit.script
def modified_gram_schmidt(A, eta):
    m, n = A.size()
    metric_length = eta.size(0)
    Q = torch.zeros((m, metric_length), dtype=A.dtype).to(A.device)
    R = torch.zeros((metric_length, n), dtype=A.dtype).to(A.device)
    eta_c = torch.zeros(metric_length, dtype=A.dtype).to(A.device)

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            v = v - project(Q[:, i], v, eta)

        norm_sq = inner_product(v, v, eta)
        norm_sq = torch.sqrt(torch.abs(norm_sq))
        assert norm_sq != 0.0
        Q[:, j] = v / norm_sq
        R[j, j] = inner_product(Q[:, j], A[:, j], eta)
        if R[j, j] < 0:
            Q[:, j] = -Q[:, j]
            R[j, j] = -R[j, j]
        eta_c[j] = torch.sign(inner_product(Q[:, j], Q[:, j], eta))
    return Q, eta_c, R


@torch.jit.script
def generate_permutation(eta_c, eta, vecs):
    n, d = vecs.size()
    S = torch.eye(d).to(vecs.dtype)
    a = eta_c
    b = torch.diag(eta)

    for i in range(d):
        if a[i] != 0 and a[i] != b[i]:
            for j in range(i + 1, d):
                if a[j] != b[j] and a[j] == b[i]:
                    S_prime = torch.eye(d).to(vecs.dtype)
                    S_prime[i, i] = 0
                    S_prime[j, j] = 0
                    S_prime[i, j] = 1
                    S_prime[j, i] = 1
                    a = torch.matmul(a, S_prime)
                    S = torch.matmul(S, S_prime)
                    break
    return S.T


@torch.jit.script
def generate_minkowski_permutation_matrix(Q, eta):
    diag_elements = torch.diag(torch.matmul(Q.T, torch.matmul(eta, Q)))
    swap_index = int(torch.argmax(diag_elements).item())
    P = torch.eye(len(diag_elements)).to(Q.dtype).to(Q.device)
    P[0, 0] = 0.0
    P[swap_index, swap_index] = 0.0
    P[0, swap_index] = 1.0
    P[swap_index, 0] = 1.0
    return P


@torch.jit.script
def generalized_qr_decomposition(coords, eta):
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, eta_c, R = modified_gram_schmidt(vecs.transpose(0, 1), eta)
    # P = generate_minkowski_permutation_matrix(Q, eta)
    P = generate_permutation(eta_c, eta, vecs)
    Q = Q @ P
    R = P.T @ torch.diag(eta_c) @ R
    return Q, R
