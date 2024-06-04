import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, GCNConv, radius_graph
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu


class MLP_Complex(nn.Module):
    def __init__(self, d):
        super(MLP_Complex, self).__init__()
        self.conv1 = ComplexLinear(d, 64)
        self.conv2 = ComplexLinear(64, 64)
        self.conv3 = ComplexLinear(64, 64)
        self.fc1 = ComplexLinear(64, 16)
        self.fc2 = ComplexLinear(16, d)

    def forward(self, x):
        x = complex_relu(self.conv1(x))
        x = complex_relu(self.conv2(x))
        x = complex_relu(self.conv3(x))
        x = complex_relu(self.fc1(x))
        return self.fc2(x)


class MLP_BN_Mish_Complex(nn.Module):
    def __init__(self, d):
        super(MLP_BN_Mish_Complex, self).__init__()
        self.conv1 = ComplexLinear(d, 64)
        self.conv2 = ComplexLinear(64, 64)
        self.conv3 = ComplexLinear(64, 64)
        self.fc1 = ComplexLinear(64, 16)
        self.fc2 = ComplexLinear(16, d)
        self.bn1 = ComplexBatchNorm1d(64)
        self.bn2 = ComplexBatchNorm1d(64)
        self.bn3 = ComplexBatchNorm1d(64)

    def forward(self, x):
        x = self.bn1(complex_relu(self.conv1(x)))
        x = self.bn2(complex_relu(self.conv2(x)))
        x = self.bn3(complex_relu(self.conv3(x)))
        x = complex_relu(self.fc1(x))
        return self.fc2(x)

class MLP_S(nn.Module):
    def __init__(self, d, n=128):
        super(MLP_S, self).__init__()
        self.conv1 = torch.nn.Linear(d, 32)
        self.conv2 = torch.nn.Linear(n, n)
        self.conv3 = torch.nn.Linear(32, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, d)

        nn.init.orthogonal(self.fc1.weight)


    def forward(self, x):
        x = F.relu(self.conv1(x)).T
        x = F.relu(self.conv2(x)).T
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLP_S_BN_Mish(nn.Module):
    def __init__(self, d, n=128):
        super(MLP_S_BN_Mish, self).__init__()
        self.conv1 = torch.nn.Linear(d, 32)
        self.conv2 = torch.nn.Linear(n, n)
        self.conv3 = torch.nn.Linear(32, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.fc2 = torch.nn.Linear(16, d)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(n)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.bn1(F.mish(self.conv1(x))).T
        x = self.bn2(F.mish(self.conv2(x))).T
        x = self.bn3(F.mish(self.conv3(x)))
        x = F.mish(self.fc1(x))
        return self.fc2(x)



class MLP(nn.Module):
    def __init__(self, d):
        super(MLP, self).__init__()
        self.conv1 = torch.nn.Linear(d, 64)
        self.conv2 = torch.nn.Linear(64, 64)
        self.conv3 = torch.nn.Linear(64, 64)
        self.fc1 = torch.nn.Linear(64, 16)
        self.fc2 = torch.nn.Linear(16, d)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLP_BN_Mish(nn.Module):
    def __init__(self, d):
        super(MLP_BN_Mish, self).__init__()
        self.conv1 = torch.nn.Linear(d, 64)
        self.conv2 = torch.nn.Linear(64, 64)
        self.conv3 = torch.nn.Linear(64, 64)
        self.fc1 = torch.nn.Linear(64, 16)
        self.fc2 = torch.nn.Linear(16, d)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.bn1(F.mish(self.conv1(x)))
        x = self.bn2(F.mish(self.conv2(x)))
        x = self.bn3(F.mish(self.conv3(x)))
        x = F.mish(self.fc1(x))
        return self.fc2(x)


class GIN(nn.Module):
    def __init__(self, d):
        super(GIN, self).__init__()
        neuron = 64
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        r3 = np.random.uniform()

        nn1 = nn.Linear(d, neuron)
        self.conv1 = GINConv(nn1, eps=r1, train_eps=True)

        nn2 = nn.Linear(neuron, neuron)
        self.conv2 = GINConv(nn2, eps=r2, train_eps=True)

        nn3 = nn.Linear(neuron, neuron)
        self.conv3 = GINConv(nn3, eps=r3, train_eps=True)

        self.fc1 = torch.nn.Linear(neuron, 16)
        self.fc2 = torch.nn.Linear(16, d)
        nn.init.normal(self.fc1.weight, std=1e-6)

    def forward(self, x):
        edge_index = radius_graph(x, 1.0)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, torch.zeros(x.size(0)).long())
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GCN(nn.Module):
    def __init__(self, d):
        super(GCN, self).__init__()
        neuron = 64
        self.conv1 = GCNConv(d, neuron)
        self.conv2 = GCNConv(neuron, neuron)
        self.conv3 = GCNConv(neuron, neuron)

        self.fc1 = torch.nn.Linear(neuron, 16)
        self.fc2 = torch.nn.Linear(16, d)

    def forward(self, x):
        edge_index = radius_graph(x, 1.0)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, torch.zeros(x.size(0)).long())
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class NonlinearSine(nn.Module):
    def __init__(self, d):
        super(NonlinearSine, self).__init__()
        pass

    def forward(self, x):
        return torch.sin(x) - F.normalize(x, dim=-1)


class NonlinearReLU(nn.Module):
    def __init__(self, d):
        super(NonlinearReLU, self).__init__()
        pass

    def forward(self, x):
        return x + torch.relu(x)