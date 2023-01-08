#!/usr/bin/env python
"""
Created on 05 16, 2022

model: utils.py

@Author: Stormzudi
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from config import pair_names, index, index_map
from config import data_path


def bulid_graph(path):
    datasets = ['{}_association.csv'.format(name) for name in pair_names]

    if not os.path.exists(os.path.join(path, datasets[0])) or not os.path.exists(os.path.join(path, datasets[1])) or \
        not os.path.exists(os.path.join(path, datasets[1])):
        raise NameError('No full datasets in dir: {}'.format(path))

    with open(os.path.join(path, "lncRNA_disease_association.csv"), encoding="utf-8") as f:
        data = pd.read_csv(f)
        idx = np.unique(data.values.flatten())
        idx_map = {j: i for i, j in enumerate(idx)}
        index_map.update({"lncRNA-disease": idx_map})

        edges = np.array(list(map(idx_map.get, data.values.flatten())),
                         dtype=np.int32).reshape(data.shape)

        edges_index = torch.tensor(edges.T, dtype=torch.long)
        index_lncRNA = {i: j for i, j in idx_map.items() if i in data['lncRNA'].values}
        index_disease = {i: j for i, j in idx_map.items() if i in data['disease'].values}

        index.update({"index_lncRNA": index_lncRNA})
        index.update({"index_disease": index_disease})
        index.update({"edges": edges})

    with open(os.path.join(path, "miRNA_disease_association.csv"), encoding="utf-8") as f:
        data = pd.read_csv(f)
        # index_mRNA
        idx_miRNA = np.unique(data['miRNA'].values)
        index_miRNA = {j: i for i, j in enumerate(idx_miRNA)}
        index.update({"index_miRNA": index_miRNA})

        # get disease embedding
        node_disease_row = np.array(list(map(index["index_disease"].get, data['disease'].values)))
        node_miRNA_col = np.array(list(map(index_miRNA.get, data['miRNA'].values)))

        sparse_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_disease_row, node_miRNA_col)),
                                  shape=(len(idx), len(idx_miRNA)),
                                  dtype=np.float32)
        disease_embedding = sparse_mx.toarray()

    with open(os.path.join(path, "lncRNA_miRNA_association.csv"), encoding="utf-8") as f:
        data = pd.read_csv(f)
        # get lncRNA embedding
        node_lncRNA_row = np.array(list(map(index["index_lncRNA"].get, data['lncRNA'].values)))
        node_miRNA_col = np.array(list(map(index["index_miRNA"].get, data['miRNA'].values)))

        sparse_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_lncRNA_row, node_miRNA_col)),
                                  shape=(len(idx), len(idx_miRNA)),
                                  dtype=np.float32)
        lncRNA_embedding = sparse_mx.toarray()

    # build the node embedding (node : lncRNA、disease、 the dim of embedding : n(miRNA))
    node_matrix = torch.zeros(len(idx_map), len(index_miRNA), dtype=torch.float32)
    labels = list()
    for i,v in idx_map.items():
        if i in index_lncRNA:
            node_matrix[v, :] = torch.from_numpy(lncRNA_embedding[v,:])
            labels.append(1)
        else:
            node_matrix[v, :] = torch.from_numpy(disease_embedding[v,:])
            labels.append(0)

    return index, edges_index, node_matrix, torch.tensor(labels, dtype=torch.int64)


def buildDataLoder(node_attr, edges_index, y):
    data = Data(x=node_attr
                , edge_index=edges_index
                , edge_attr=None
                , y=y)
    # Data(edge_index=[2, 18789], x=[828, 233]) 一个图有18789条边；828个节点，每个节点有233个特征

    # link predict must build train_mask, val_mask, and tesk_mask
    data.train_mask = data.val_mask = data.test_mask = data.y = None  # 不再有用

    transform = RandomLinkSplit(num_val=0.1,
                                num_test=0.1,
                                add_negative_train_samples=False,
                                is_undirected=True)
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
            (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx




if __name__ == '__main__':

    path = os.path.join(data_path, "lnc-dis/traindata")
    index, edges_index, node_attr, label = bulid_graph(path)
    train_data, val_data, test_data = buildDataLoder(node_attr, edges_index, y=label)
    print("train_data: ", train_data)
    print("val_data: ", val_data)
    print("test_data: ", test_data)