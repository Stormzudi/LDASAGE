#!/usr/bin/env python
"""
Created on 05 16, 2022

model: utils.py

@Author: Stormzudi
"""


import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import torch.nn as nn

"""
def:
    1. bulid_graph
    2. buildDataLoder
    3. create_alias_table
    4. alias_sample
    5. partition_num
    6. preprocess_nxgraph
"""


class BuildDataset():
    def __init__(self):
        self.index = dict()
        self.len_l = None  # 350
        self.len_m = None  # 233
        self.len_d = None  # 478

    def getAMatrix(self, idx_map, name_row, name_col, data):
        """
        @param idx_map:  Dict for all node including(lncRNA, miRNA, disease)
        @param name_row: lncRNA, miRNA, disease
        @param name_col: lncRNA, miRNA, disease
        @param data: .csv
        @return:
        """
        row, col = 0, 0
        data_mx = None
        node_len = self.len_l + self.len_m + self.len_d
        if name_row == "lncRNA":
            node_lncRNA = np.array(list(map(self.index["index_lncRNA"].get, data['lncRNA'].values)))
            if name_col == "disease":
                # define A_LD
                node_disease = np.array(list(map(self.index["index_disease"].get, data['disease'].values)))
                data_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_lncRNA, node_disease)),
                                     shape=(node_len, node_len),
                                     dtype=np.float32).toarray()
                row, col = self.len_l, self.len_d
            elif name_col == "miRNA":
                # define A_LM
                node_miRNA = np.array(list(map(self.index["index_miRNA"].get, data['miRNA'].values)))
                data_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_lncRNA, node_miRNA)),
                                     shape=(node_len, node_len),
                                     dtype=np.float32).toarray()
                row, col = self.len_l, self.len_m
                pass
        else:
            # define A_MD
            node_miRNA = np.array(list(map(self.index["index_miRNA"].get, data['miRNA'].values)))
            node_disease = np.array(list(map(self.index["index_disease"].get, data['disease'].values)))
            data_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_miRNA, node_disease)),
                                    shape=(node_len, node_len),
                                    dtype=np.float32).toarray()
            row, col = self.len_m, self.len_d
            pass

        Mat = torch.zeros(row, col, dtype=torch.float32)
        num = 0
        for i, v in idx_map.items():  # A1BG-AS1, 0
            if i in self.index["index_{}".format(name_row)]:
                line = torch.from_numpy(data_mx[v, :])
                # get miRNA from line
                Mat[num, :] = line[list(self.index["index_{}".format(name_col)].values())]
                num += 1
        return Mat

    def getSMetrix(self,row,col,min,max):
        X = np.random.randint(min,max,(row, col))
        X = np.triu(X)
        X += X.T - np.diag(X.diagonal())
        row, col = np.diag_indices_from(X)
        X[row, col] = 1

        return X

    def getData(self, path):
        names = ['lncRNA_disease', 'miRNA_disease', 'lncRNA_miRNA']
        datasets = ['{}_association.csv'.format(name) for name in names]
        data_ld, data_md, data_lm = None, None, None
        for name in datasets:
            with open("{}{}".format(path, name), encoding="utf-8") as f:
                data = pd.read_csv(f)
            if name == "lncRNA_disease_association.csv":
                data_ld = data
            if name == "miRNA_disease_association.csv":
                data_md = data
            if name == "lncRNA_miRNA_association.csv":
                data_lm = data

        # idx: 1061
        idx = np.unique(np.concatenate(
            [data_ld['lncRNA'].values, data_lm['miRNA'].values, data_md['disease'].values]))
        idx_map = {j: i for i, j in enumerate(idx)}

        index = dict()
        index.update({"index_lncRNA": {i: j for i, j in idx_map.items() if i in data_ld['lncRNA'].values}})
        index.update({"index_disease": {i: j for i, j in idx_map.items() if i in data_ld['disease'].values}})
        index.update({"index_miRNA": {i: j for i, j in idx_map.items() if i in data_md['miRNA'].values}})
        index.update({"index_all":idx_map})

        self.len_l, self.len_m, self.len_d = len(np.unique(data_ld['lncRNA'].values)), len(np.unique(data_lm['miRNA'].values)), len(np.unique(data_md['disease'].values))

        # 1. Given similarity matrix S(intra_L, intra_M, intra_D)
        S_ll = self.getSMetrix(self.len_l, self.len_l, 0, 2)
        S_mm = self.getSMetrix(self.len_m, self.len_m, 0, 2)
        S_dd = self.getSMetrix(self.len_d, self.len_d, 0, 2)

        # 2. Given inter-association matrix A
        self.index["index_lncRNA"] = {i: j for i, j in idx_map.items() if i in data_ld['lncRNA'].values}
        self.index["index_disease"] = {i: j for i, j in idx_map.items() if i in data_md['disease'].values}
        self.index["index_miRNA"] = {i: j for i, j in idx_map.items() if i in data_lm['miRNA'].values}

        ### 2.1 A_LD???A_LM???A_MD
        A_LD = self.getAMatrix(idx_map=idx_map, name_row="lncRNA", name_col="disease", data=data_ld)
        A_LM = self.getAMatrix(idx_map=idx_map, name_row="lncRNA", name_col="miRNA", data=data_lm)
        A_MD = self.getAMatrix(idx_map=idx_map, name_row="miRNA", name_col="disease", data=data_md)

        ### 2.2 Adj-matrix M-complex
        # Feature matrix of inter-graph
        M_interLD = np.vstack((np.hstack((S_ll,A_LD)),
                               np.hstack((A_LD.T, S_dd))))
        M_interLM = np.vstack((np.hstack((S_ll,A_LM)),
                               np.hstack((A_LM.T, S_mm))))
        M_interMD = np.vstack((np.hstack((S_mm,A_MD)),
                               np.hstack((A_MD.T, S_dd))))

        # 3. Adj_c
        adj_c = np.vstack((np.hstack((S_ll, A_LD, A_LM)),
                           np.hstack((A_LD.T, S_dd, A_MD.T)),
                           np.hstack((A_LM.T, A_MD, S_mm)))
                          )

        return index, adj_c, S_ll, S_mm, S_dd, M_interLD, M_interLM, M_interMD
        # return adj_c, adj_l, adj_m, adj_d, adj_ld, adj_lm, adj_md


def buildDataLoder(node_attr, edges_index, y):
    data = Data(x=node_attr
                , edge_index=edges_index
                , edge_attr =None
                , y=y)
    # Data(edge_index=[2, 18789], x=[828, 233]) ????????????18789?????????828???????????????????????????233?????????

    # link predict must build train_mask, val_mask, and tesk_mask
    data.train_mask = data.val_mask = data.test_mask = data.y = None  # ????????????

    transform = RandomLinkSplit(num_val=0.1,
                                num_test=0.1,
                                add_negative_train_samples=False,
                                is_undirected=True)
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


def getFeature(adj, min=0, max=5):
    row, col = adj.shape
    return np.random.randint(min, max, (row, col))

def bulid_graph(path):
    """
    return: data (Type: Data)
    :param path: the dataset path
    """
    global idx_map

    names = ['lncRNA_disease', 'miRNA_disease', 'lncRNA_miRNA']
    datasets = ['{}_association.csv'.format(name) for name in names]

    # index_lncRNA = {}
    # index_mRNA = {}
    # index_disease = {}
    index = {}

    for name in datasets:
        with open("{}{}".format(path, name), encoding="utf-8") as f:
            data = pd.read_csv(f)
            # data = np.loadtxt(f, str, delimiter=",", skiprows=1)
        # build Adj
        if name == "lncRNA_disease_association.csv":
            idx = np.unique(data.values.flatten())
            idx_map = {j: i for i, j in enumerate(idx)}

            edges = np.array(list(map(idx_map.get, data.values.flatten())),
                             dtype=np.int32).reshape(data.shape)

            edges_index = torch.tensor(edges.T, dtype=torch.long)
            index_lncRNA = {i: j for i, j in idx_map.items() if i in data['lncRNA'].values}
            index_disease = {i: j for i, j in idx_map.items() if i in data['disease'].values}
            index["index_lncRNA"] = index_lncRNA
            index["index_disease"] = index_disease
            index["edges"] = edges


        elif name == "miRNA_disease_association.csv":
            # index_mRNA
            idx_miRNA = np.unique(data['miRNA'].values)
            index_miRNA = {j: i for i, j in enumerate(idx_miRNA)}
            index["index_miRNA"] = index_miRNA

            # get disease embedding
            node_disease_row = np.array(list(map(index["index_disease"].get, data['disease'].values)))
            node_miRNA_col = np.array(list(map(index["index_miRNA"].get, data['miRNA'].values)))

            sparse_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_disease_row, node_miRNA_col)),
                                      shape=(len(idx), len(idx_miRNA)),
                                      dtype=np.float32)
            disease_embedding = sparse_mx.toarray()


        elif name == "lncRNA_miRNA_association.csv":

            # get lncRNA embedding
            node_lncRNA_row = np.array(list(map(index["index_lncRNA"].get, data['lncRNA'].values)))
            node_miRNA_col = np.array(list(map(index["index_miRNA"].get, data['miRNA'].values)))

            sparse_mx = sp.coo_matrix((np.ones(data.shape[0]), (node_lncRNA_row, node_miRNA_col)),
                                      shape=(len(idx), len(idx_miRNA)),
                                      dtype=np.float32)
            lncRNA_embedding = sparse_mx.toarray()

        else:
            pass

    # build the node embedding (node : lncRNA???disease??? the dim of embedding : n(miRNA))
    node_attr = torch.zeros(len(idx_map), len(index_miRNA), dtype=torch.float32)
    labels = list()
    for i,v in idx_map.items():
        if i in index_lncRNA:
            node_attr[v, :] = torch.from_numpy(lncRNA_embedding[v,:])
            labels.append(1)
        else:
            node_attr[v, :] = torch.from_numpy(disease_embedding[v,:])
            labels.append(0)

    return index, edges_index, node_attr, torch.tensor(labels, dtype=torch.int64)



class SimpleMSELoss(nn.Module):
    def __init__(self):
        super(SimpleMSELoss, self).__init__()

    def forward(self, y, y_hat):
        y = torch.mm(y, y.transpose(0,1))
        return torch.sum(torch.abs(y - y_hat))

if __name__ == '__main__':

    path = "../../dataset/valdata/"
    bd = BuildDataset()
    index, adj_c, adj_l, adj_m, adj_d, adj_ld, adj_lm, adj_md = bd.getData(path)
    a = 1