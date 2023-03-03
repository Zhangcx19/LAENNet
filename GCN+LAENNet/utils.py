import numpy as np
import copy
import random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.kl import kl_divergence
import logging
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder


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
    return np.array(mask, dtype=np.bool)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset_str, noise_fea_dim=0, important_del_dim=0, feature_noise=0, adj_noise=0):
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
        with open("../data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # Labels.
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # Features.
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if important_del_dim != 0:
        with open(dataset_str + '.pkl', 'rb') as text:
            important_idx = pkl.load(text)
        a = random.sample(important_idx, important_del_dim)
        b = sorted(list(set([i for i in range(features.shape[1])]) - set(a)))
        features = features[:, b]

    elif noise_fea_dim != 0:
        c = sorted(random.sample([i for i in range(features.shape[1])], noise_fea_dim))
        features = features[:, c]

    elif feature_noise != 0:
        connected_nodes_file_path = '../data/' + dataset_str + '/connected_nodes.txt'
        connected_nodes = []
        with open(connected_nodes_file_path) as connected_node_file:
            for line in connected_node_file.readlines():
                l = line.strip('\n')
                connected_nodes.append(int(l))
        d = features.todense().A
        idxes = [(i, j) for i in connected_nodes for j in range(d.shape[1])]
        noised_idx = random.sample(idxes, int(len(connected_nodes) * d.shape[1] * feature_noise))
        for idx in noised_idx:
            random_noise = np.random.uniform(0, 1)
            d[idx] = random_noise
        features = sp.lil_matrix(d)

    features = preprocess_features(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # Adj.
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, labels


def load_npz(dataset_str, important_del_dim=0, feature_noise=0):
    file_map = {'coauthor-cs': 'ms_academic_cs.npz', 'coauthor-phy': 'ms_academic_phy.npz'}
    file_name = file_map[dataset_str]

    # with np.load('../data/' + file_name, allow_pickle=True) as f:
    f = np.load('../data/' + file_name, allow_pickle=True)
    f = dict(f)
    features = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']), shape=f['attr_shape']).tolil()
    features = features.astype(np.float64)
    labels = f['labels'].reshape(-1, 1)
    labels = OneHotEncoder(sparse=False).fit_transform(labels)
    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']), shape=f['adj_shape'])
    adj = torch.FloatTensor(adj.todense().A)

    if important_del_dim != 0:
        with open('../data/' + dataset_str + '.pkl', 'rb') as text:
            important_idx = pkl.load(text)
        a = random.sample(important_idx, important_del_dim)
        b = sorted(list(set([i for i in range(features.shape[1])]) - set(a)))
        features = features[:, b]

    elif feature_noise != 0:
        connected_nodes_file_path = '../data/' + dataset_str + '_connected_nodes.txt'
        connected_nodes = []
        with open(connected_nodes_file_path) as connected_node_file:
            for line in connected_node_file.readlines():
                l = line.strip('\n')
                connected_nodes.append(int(l))
        d = features.todense().A
        rand_noise = np.random.random(d.shape)
        indicies = np.random.choice(np.arange(rand_noise.size), replace=False,
                                    size=int(rand_noise.size * (1 - feature_noise)))
        rand_noise[np.unravel_index(indicies, rand_noise.shape)] = 0.
        d += rand_noise
        features = sp.lil_matrix(d)

    features = preprocess_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    return adj, features, labels


def set_diag_zero(adj):
    for i in range(adj.shape[0]):
        adj[i][i] = 0
    return adj


def sym_adj(adj):
    sym_adj = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            temp = adj[i][j]
            if temp != 0:
                sym_adj[i][j] = temp
                sym_adj[j][i] = temp
    return sym_adj


def dense_to_onehot(labels_dense, num_classes=5):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot


def preprocess_features(features):
    """Ro9w-normalize feature matrix."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def data_partion(dataset, labels, train_percentage, eval_percentage):
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        connected_nodes_file_path = '../data/' + dataset + '/connected_nodes.txt'
    else:
        connected_nodes_file_path = '../data/' + dataset + '_connected_nodes.txt'
    connected_nodes = []
    with open(connected_nodes_file_path) as connected_node_file:
        for line in connected_node_file.readlines():
            l = line.strip('\n')
            connected_nodes.append(int(l))
    disconnected_nodes = [i for i in range(labels.shape[0]) if i not in connected_nodes]

    nodes_class = np.argmax(labels, axis=1)
    classes_node = []
    for cla in range(labels.shape[1]):
        class_node = [j for j, x in enumerate(nodes_class) if x == cla]
        temp = [k for k in class_node if k not in disconnected_nodes]
        classes_node.append(temp)

    test_percentage = 1.0 - train_percentage - eval_percentage
    train_indexes = []
    eval_indexes = []
    test_indexes = []
    train_eval_classes_node = []
    for category in classes_node:
        class_samples = int(len(category) * test_percentage)
        test_temp = random.sample(category, class_samples)
        test_indexes = test_indexes + test_temp
        temp = [p for p in category if p not in test_temp]
        train_eval_classes_node.append(temp)
    for category in train_eval_classes_node:
        train_class_samples = int(len(category) * (train_percentage / (train_percentage + eval_percentage)))
        train_temp = random.sample(category, train_class_samples)
        train_indexes = train_indexes + train_temp
        eval_temp = [q for q in category if q not in train_temp]
        eval_indexes = eval_indexes + eval_temp

    train_mask = sample_mask(train_indexes, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_train = torch.FloatTensor(y_train)
    labels = torch.LongTensor(np.argmax(labels, axis=-1))
    eval_test_indexes = eval_indexes + test_indexes

    return labels, train_indexes, eval_indexes, test_indexes, y_train, connected_nodes, eval_test_indexes


def data_partion_gcn_style(dataset, labels, nodes_per_class, label_nr=0):
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        connected_nodes_file_path = '../data/' + dataset + '/connected_nodes.txt'
    else:
        connected_nodes_file_path = '../data/' + dataset + '_connected_nodes.txt'
    connected_nodes = []
    with open(connected_nodes_file_path) as connected_node_file:
        for line in connected_node_file.readlines():
            l = line.strip('\n')
            connected_nodes.append(int(l))
    disconnected_nodes = [i for i in range(labels.shape[0]) if i not in connected_nodes]

    nodes_class = np.argmax(labels, axis=1)
    classes_node = []
    for cla in range(labels.shape[1]):
        class_node = [j for j, x in enumerate(nodes_class) if x == cla]
        temp = [k for k in class_node if k not in disconnected_nodes]
        classes_node.append(temp)

    train_indexes = []
    eval_test_indexes = []
    eval_test_classes_node = []
    for category in classes_node:
        train_temp = random.sample(category, nodes_per_class)
        train_indexes = train_indexes + train_temp
        temp = [p for p in category if p not in train_temp]
        eval_test_classes_node.append(temp)
    for category in eval_test_classes_node:
        eval_test_indexes = eval_test_indexes + category
    eval_indexes = random.sample(eval_test_indexes, 500)
    test_select_indexes = [q for q in eval_test_indexes if q not in eval_indexes]
    test_indexes = random.sample(test_select_indexes, 1000)

    if label_nr != 0:
        new_labels = copy.deepcopy(labels)
        for idx in train_indexes:
            real_label = labels[idx]
            noise_random = random.random()
            if noise_random <= label_nr:
                non_zero = np.where(real_label == 0)[0]
                ran = random.sample(list(non_zero), 1)
                noisy_label = np.zeros(labels.shape[1], dtype=np.int32)
                noisy_label[ran[0]] = 1
                new_labels[idx] = noisy_label
        labels = copy.deepcopy(new_labels)

    train_mask = sample_mask(train_indexes, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_train = torch.FloatTensor(y_train)
    labels = torch.LongTensor(np.argmax(labels, axis=-1))

    return labels, train_indexes, eval_indexes, test_indexes, y_train, connected_nodes, eval_test_indexes


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def del_list(all, delist):
    all_copy = copy.deepcopy(all)
    for item in delist:
        all_copy.remove(item)
    return all_copy


def obtain_adj(adj, zero_indexes):
    adj_copy = copy.deepcopy(adj)
    for ind in zero_indexes:
        adj_copy[:, ind] = 0
    return adj_copy


def obtain_no_real_label_node_labeled_neighbors(no_real_label_nodes, adj, all_labeled_node_indexes):
    node_and_labeled_neighbors = dict.fromkeys(no_real_label_nodes, [])
    for node in no_real_label_nodes:
        node_neighbors = [int(nei[0].cpu().numpy()) for nei in torch.nonzero(adj[node])]
        labeled_neighbors = []
        for neighbor in node_neighbors:
            if neighbor in all_labeled_node_indexes:
                labeled_neighbors.append(neighbor)
        if len(labeled_neighbors) > 0:
            node_and_labeled_neighbors[node] = labeled_neighbors
        else:
            node_and_labeled_neighbors.pop(node)
    return node_and_labeled_neighbors


def hc_obtain_predicted_label(pred, confidence_rate, no_real_label_node_index):
    max_value_indexes = torch.argmax(pred, dim=1)
    hc_indexes = []
    for ind in no_real_label_node_index:
        if pred[ind][max_value_indexes[ind]] > confidence_rate:
            hc_indexes.append(ind)
    return hc_indexes


def update_y_train(y_train, predicted_labels, label_predicted_nrl_nodes):
    for ind in label_predicted_nrl_nodes:
        y_train[ind] = predicted_labels[ind]
    return y_train


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=logFilename,
        filemode='w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)