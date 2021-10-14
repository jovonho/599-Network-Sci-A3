import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

"""
    Adapted from the original
    https://github.com/tkipf/gcn/blob/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81/gcn/utils.py
"""
def load_data(dataset_str, add_val_to_test=False):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/gcn/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/gcn/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)


    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        # tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # tx_extended[test_idx_range-min(test_idx_range), :] = tx
        # tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    G = nx.from_dict_of_lists(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # There are 15 nodes with no category in citeseer
    # We will arbitrarily assign them to category 0 
    # since its easier than modifying the graph object to remove them
    if dataset_str == 'citeseer':
        labels_all = []

        for line in labels:
            category = np.where(line == 1)
            if len(category[0]) > 0:
                category = int(category[0][0])
            else:
                category = 0
            labels_all.append(category)
        
        labels = labels_all
    else:
        labels = list(labels.nonzero()[1])

    # Add the train labels to the graph
    for i in range(len(G.nodes)):
        if i in idx_train:
            G.nodes[i]['value'] = labels[i]
        elif i in idx_val and add_val_to_test:
            G.nodes[i]['value'] = labels[i]

    # Optionally add the labels of the validation set to the graph as well
    # Effectively adding the validation set to the training set
    if add_val_to_test:
        test_size = len(labels) - (len(idx_train) + len(idx_val))
    else:
        test_size = len(labels) - len(idx_train)
    
    # print(f"{dataset_str} train: {len(idx_train)} val:{len(idx_val)} test:{len(idx_test)}")

    return G, labels, idx_test, test_size


def load_data_fully_labeled(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/gcn/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/gcn/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)


    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        # tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # tx_extended[test_idx_range-min(test_idx_range), :] = tx
        # tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    G = nx.from_dict_of_lists(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # There are 15 nodes with no category in citeseer
    # We will arbitrarily assign them to category 0 
    # since its easier than modifying the graph object to remove them
    if dataset_str == 'citeseer':
        labels_all = []

        for line in labels:
            category = np.where(line == 1)
            if len(category[0]) > 0:
                category = int(category[0][0])
            else:
                category = 0
            labels_all.append(category)
        
        labels = labels_all
    else:
        labels = list(labels.nonzero()[1])

    # Label all nodes
    for i in range(len(G.nodes)):
        G.nodes[i]['value'] = labels[i] 

    return G, np.array(labels)



if __name__=='__main__':

    G, labels, idx_test, _ = load_data("citeseer")

    print(G.nodes(data=True))

    print(idx_test)

    
    