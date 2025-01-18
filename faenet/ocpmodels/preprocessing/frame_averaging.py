import random
from copy import deepcopy
from itertools import product
from ocpmodels.common.graph_transforms import RandomRotate
import torch


@torch.jit.script
def find_independent_vectors_cuda(P):
    """ Find rank(P) linearly independent vectors from P """
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
def frame_cuda(coords):
    """ QR decomposition on induced SO(d) set """
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.transpose(0, 1), mode='complete')
    for j in range(R.size(1)):
        if R[j, j] < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    if R.size(0) == R.size(1):
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
    elif R.size(0) == R.size(1) + 1:
        if torch.linalg.det(Q) < 0:
            Q[:, -1] *= -1
    else:
        Q[:, R.size(1):] = 0.0
    return Q


def frame_averaging_3D(pos, cell=None, fa_method="random", check=False):
    """Computes new positions for the graph atoms using minimal frame averaging

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        fa_method (str): FA method used (NOT USED HERE)
        check (bool): apply random reflection or not. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used
    """
    pos = pos - pos.mean(dim=0, keepdim=True)

    rot = frame_cuda(pos)
    if check:
        if random.randint(0, 1) == 0:
            # Additional rotation augmentation by pi
            rot[:, 0] *= -1
            rot[:, 1] *= -1

    fa_pos = pos @ rot
    fa_cell = cell @ rot
    return [fa_pos], [fa_cell], [rot]


def frame_averaging_2D(pos, cell=None, fa_method="random", check=False):
    """Computes new positions for the graph atoms using minimal frame averaging

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph
        fa_method (str): FA method used (NOT USED HERE)
        check (bool): apply random reflection or not. Default: False.

    Returns:
        tensor: updated atom positions
        tensor: updated unit cell
        tensor: the rotation matrix used
    """

    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)

    rot = frame_cuda(pos_2D)
    # if check:
    #     if random.randint(0, 1) == 0:
    #         # Additional rotation augmentation by pi
    #         rot[:, 0] *= -1
    #         rot[:, 1] *= -1

    fa_pos = pos_2D @ rot
    fa_rot = torch.eye(3)
    fa_pos = torch.cat((fa_pos, pos[:, 2].unsqueeze(1)), dim=1)
    fa_rot[:2, :2] = rot
    fa_cell = cell @ fa_rot
    return [fa_pos], [fa_cell], [fa_rot]


def data_augmentation(g, d=3, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated
