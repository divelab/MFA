import argparse
import json
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from n_body_system.dataset_nbody import NBodyDataset
import os
from torch import nn, optim

from n_body_system.model import GNN
import torch.nn.functional as F

from utils import augment_rotation_matrices

parser = argparse.ArgumentParser(description='n-Body Experiments')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=43, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=2, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=60, metavar='N',
                    help='learning rate')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=5e-6, metavar='N',
                    help='timing experiment')
parser.add_argument('--checkpoint', type=str, default=None, metavar='N', help='checkpoint path')
parser.add_argument('--inference', action='store_true', default=False, help='inference or not')

torch.set_num_threads(1)
time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mae = nn.L1Loss()

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# Setting seed for the reproducible results
SEED = 52


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.jit.script
def batch_frame_cuda(vecs):
    """ QR decomposition / Frame calculation on induced O(3) set """
    assert vecs is not None
    frame, R = torch.linalg.qr(vecs.transpose(1, 2), mode='complete')
    neg_diag = torch.diagonal(R, dim1=1, dim2=2) < 0
    for j in range(3):
        frame[neg_diag[:, j], :, j] *= -1
    return frame


def main():
    # Setting seed for the reproducible results
    set_seed(SEED)
    dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset,
                                 max_samples=args.max_training_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, act_fn=nn.SiLU()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    results = {'epochs': [], 'train_losess': [], 'val_losess': [], 'test_losess': [], 'best_val': 1e10,
               'best_test': 1e10, 'best_epoch': 0}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    early_stop = 0

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.inference:
        test_loss = train(model, optimizer, 0, loader_test, backprop=False)
        print("*** Best Test Loss: %.5f" % test_loss)
        return test_loss

    # Setting seed for the reproducible results
    rng = np.random.default_rng(SEED)
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train, seed=rng)
        early_stop += 1
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['train_losess'].append(train_loss)
            results['val_losess'].append(val_loss)
            results['test_losess'].append(test_loss)
            if val_loss < best_val_loss:
                early_stop = 0
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                results['best_val'] = val_loss
                results['best_test'] = test_loss
                results['best_epoch'] = epoch
                torch.save(model.state_dict(), args.outf + "/" + args.exp_name + "/model.pth")
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (
                best_val_loss, best_test_loss, best_epoch))
        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)
        if early_stop > 3000:
            break
    print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (
        best_val_loss, best_test_loss, best_epoch))
    return best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True, seed=None):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'mse_loss': 0}
    for data in tqdm(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]

        loc, vel, edge_attr, charges, loc_end = data
        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        size = loc.size()

        center = loc.mean(dim=1, keepdim=True).expand_as(loc)
        loc_mean = loc - center
        # Calculating O(3) frame
        frame = batch_frame_cuda(loc_mean[:, :3, :])

        loc = loc.view(-1, loc.size(-1))
        edge_attr = edge_attr.view(-1, edge_attr.size(-1))
        charges = charges.view(-1, charges.size(-1))
        loc_end = loc_end.view(-1, loc_end.size(-1))
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, dim=1, keepdim=True)
        edge_attr = torch.cat([charges[rows] * charges[cols], loc_dist ** 2], -1)  # frame will not change this feature

        # Applying O(3) frame
        loc_mean = torch.bmm(loc_mean, frame)
        vel = torch.bmm(vel, frame)
        loc_mean = loc_mean.view(-1, loc.size(-1))
        center = center.reshape(-1, loc.size(-1))
        vel = vel.view(-1, vel.size(-1))

        if backprop:
            # Training
            optimizer.zero_grad()
            loss = 0
            # Alleviate discontinuity of canonicalization in Appendix I.2. FA-GNN is retrained under the same circumstance.
            # See https://arxiv.org/abs/2402.16077.
            for R in augment_rotation_matrices(device, seed=seed):
                nodes = torch.cat([loc_mean @ R + center, vel @ R], dim=-1)
                loc_pred = model(nodes, edges, edge_attr)
                loc_pred = torch.mm(loc_pred, R.transpose(0, 1))
                # Applying O(3) frame
                loc_pred = torch.bmm(loc_pred.view(*size), frame.transpose(1, 2)).view_as(loc) + loc
                loc_pred = loc_pred.view(-1, loc_pred.size(-1))
                loss += loss_mae(loc_pred, loc_end)

            loss = loss / 6.0
            loss.backward()
            optimizer.step()
        else:
            # Inference
            nodes = torch.cat([loc_mean + center, vel], dim=-1)
            loc_pred = model(nodes, edges, edge_attr)
            # Applying O(3) frame
            loc_pred = torch.bmm(loc_pred.view(*size), frame.transpose(1, 2)).view_as(loc) + loc
            loc_pred = loc_pred.view(-1, loc_pred.size(-1))
            loss = loss_mae(loc_pred, loc_end)

        res['loss'] += loss.item()
        res['mse_loss'] += F.mse_loss(loc_pred, loc_end).item()
        res['counter'] += 1

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix + loader.dataset.partition, epoch, res['mse_loss'] / res['counter']))

    return res['mse_loss'] / res['counter']


if __name__ == "__main__":
    main()
