"""
Canonical Labeling Utils
============

Author:
-------
Yuchao Lin

"""
import numpy as np
from sympy.combinatorics import Permutation

from minimal_frame.nauty import Nauty


def permutation_array_to_matrix(permutation_array):
    """ Converting permutation array to permutation matrix """
    n = len(permutation_array)
    permutation_matrix = np.zeros((n, n), dtype=int)
    for i, p in enumerate(permutation_array):
        permutation_matrix[i, p] = 1
    return permutation_matrix


def coordinates_to_graph(coords):
    """ Converting coordinates to undirected weighted graph with inner products as edge attributes """
    d, n = coords.shape
    iu = np.triu_indices(n, 1)
    edge_index = np.vstack(iu).T
    edge_attr = (coords.T @ coords)[iu]
    return edge_index, edge_attr


def relabel_undirected_graph(node_attr, edge_index, edge_attr):
    """ Converting undirected weighted graph to undirected unweighted graph """
    n = node_attr.shape[0]

    unique_node_attrs, node_colors = np.unique(node_attr, axis=0, return_inverse=True)
    unique_edge_attrs, edge_colors = np.unique(edge_attr, axis=0, return_inverse=True)
    edge_colors += n

    adj_matrix_size = n + len(edge_index)
    adj_matrix = np.zeros((adj_matrix_size, adj_matrix_size), dtype=np.bool_)

    for idx, (src, tgt) in enumerate(edge_index):
        edge_vertex = n + idx
        adj_matrix[src][edge_vertex] = True
        adj_matrix[edge_vertex][src] = True
        adj_matrix[tgt][edge_vertex] = True
        adj_matrix[edge_vertex][tgt] = True

    weights = np.concatenate((node_colors, edge_colors))

    return adj_matrix, weights


def create_lab_ptn_from_weights(weights):
    """ Create input to nauty algorithm from node colors (or weights) """
    inds = np.arange(len(weights))

    indices = np.lexsort((inds, weights))
    sorted_weights = np.array(weights)[indices]

    ptn = np.ones_like(weights, dtype=np.int32)
    ptn[-1] = 0

    for i in range(len(sorted_weights) - 1):
        if sorted_weights[i] != sorted_weights[i + 1]:
            ptn[i] = 0

    return indices, ptn


def generate_permutation_frames(graph_data, weighted=True, maximum_node=None):
    """ Create S_n frames """
    node_attr, edge_index, edge_attr = graph_data
    if weighted:
        adj_matrix, weights = relabel_undirected_graph(node_attr, edge_index, edge_attr)
    else:
        n = node_attr.shape[0]
        adj_matrix = np.zeros((n, n), dtype=np.bool_)
        for edge in edge_index:
            adj_matrix[edge[0], edge[1]] = True
            adj_matrix[edge[1], edge[0]] = True
        weights = node_attr

    lab, ptn = create_lab_ptn_from_weights(weights)
    N_py = Nauty(adj_matrix.shape[0], adj_matrix, lab, ptn, defaultptn=False)
    if maximum_node is not None:
        canon = Permutation(N_py.canonlab[:maximum_node].tolist())
    else:
        canon = Permutation(N_py.canonlab.tolist())
    autos = N_py.generate_full_group(maximum_node)
    if autos is not None:
        return [canon * auto for auto in autos]
    else:
        return [canon]
