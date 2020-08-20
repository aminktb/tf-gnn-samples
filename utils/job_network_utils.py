import numpy as np
from collections import defaultdict
from .citation_network_utils import sample_mask


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(directory: str):
    idx_features_labels = np.genfromtxt(
        "{}/{}.content".format(directory, "jobs"), dtype=np.dtype(str)
    )
    features = idx_features_labels[:, 1:-1].astype(float)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}/{}.graph".format(directory, "jobs"), dtype=np.int32
    )
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    graph = defaultdict(list)
    for node in range(len(idx)):
        graph[node] = []
    for edge in edges:
        graph[edge[0]].append(edge[1])

    idx_test = range(4000, 4500)
    idx_train = range(3500)
    idx_val = range(3500, 4000)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return graph, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
