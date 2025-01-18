import os
import matplotlib
import numpy as np

matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from scipy.stats import ortho_group


def create_folders(args):
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_recon')
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_gen')
    except OSError:
        pass


def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def normalize_res(res, keys=[]):
    for key in keys:
        if key != 'counter':
            res[key] = res[key] / res['counter']
    del res['counter']
    return res


def plot_coords(coords_mu, path, coords_logvar=None):
    if coords_mu is None:
        return 0
    if coords_logvar is not None:
        coords_std = torch.sqrt(torch.exp(coords_logvar))
    else:
        coords_std = torch.zeros(coords_mu.size())
    coords_size = (coords_std ** 2) * 1

    plt.scatter(coords_mu[:, 0], coords_mu[:, 1], alpha=0.6, s=100)

    # plt.errorbar(coords_mu[:, 0], coords_mu[:, 1], xerr=coords_size[:, 0], yerr=coords_size[:, 1], linestyle="None", alpha=0.5)

    plt.savefig(path)
    plt.clf()


def augment_random_matrices(device, seed):
    return torch.FloatTensor(ortho_group.rvs(dim=3, random_state=seed)).to(device)


def augment_rotation_matrices(device, seed=None):
    A = augment_random_matrices(device, seed)
    I = torch.eye(3).float().to(device)
    Rx = torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to(device)
    Ry = torch.FloatTensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).to(device)
    Rz = torch.FloatTensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(device)
    Rx_neg = torch.FloatTensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to(device)
    Ry_neg = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).to(device)
    return I @ A, Rx @ A, Ry @ A, Rz @ A, Rx_neg @ A, Ry_neg @ A


def filter_nodes(dataset, n_nodes):
    new_graphs = []
    for i in range(len(dataset.graphs)):
        if len(dataset.graphs[i].nodes) == n_nodes:
            new_graphs.append(dataset.graphs[i])
    dataset.graphs = new_graphs
    dataset.n_nodes = n_nodes
    return dataset


def adjust_learning_rate(optimizer, epoch, lr_0, factor=0.5, epochs_decay=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (factor ** (epoch // epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
