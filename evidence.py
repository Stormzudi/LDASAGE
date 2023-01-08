#!/usr/bin/env python
"""
Created on 08 12, 2022

model: evidence.py

@Author: Stormzudi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
import networkx as nx
import os
from config import save_path, data_path, lncRNA_miRNA, DATA
import matplotlib.pyplot as plt

class ModelEvidence():
    """
        Evidence the result of `LDASAGE` model
    """
    def __init__(self):
        self.result_path = save_path
        self.lnc_Dis_path = None
        self.hidden_emb = None
        self.adj_orig = None
        self.nodeList = None
        self.nodeAll = None

    def load_data(self, sample_data_path):
        df = pd.read_csv(sample_data_path, sep=",")
        g = nx.from_pandas_edgelist(df, "lncRNA", "disease")
        # g = nx.read_edgelist(sample_data_path)
        adj = nx.adjacency_matrix(g)

        nodeIndex = dict()
        nodeAll = dict()
        # 得到node 和 node的 序列

        data = df.values

        nodeLncRNA = dict()
        nodeDisease = dict()
        g_nodes = list(g.nodes)

        for i in range(len(g_nodes)):
            nodeAll[g_nodes[i]] = i
            if g_nodes[i] in data[:, 0]:
                nodeLncRNA[g_nodes[i]] = i
            if g_nodes[i] in data[:, 1]:
                nodeDisease[g_nodes[i]] = i

        nodeIndex["index_lncRNA"] = nodeLncRNA
        nodeIndex["index_disease"] = nodeDisease

        return adj, nodeIndex, nodeAll

    def load_model(self):

        # 1. load the result embedding
        embeds_path = os.listdir(self.result_path)
        embeds_path = list(filter(lambda x: "lncRNA_disease" in x, embeds_path))
        path = os.path.join(self.result_path, embeds_path[-1])
        self.hidden_emb = np.load(path)

        # 2. load adj 原始邻接矩阵，除去对角线元素
        path_ = os.path.join(data_path, DATA.get('database'), DATA.get("dataset").get("train"))
        self.lnc_Dis_path = os.path.join(path_, "lncRNA_disease_association.csv")
        adj, self.nodeList, self.nodeAll = self.load_data(self.lnc_Dis_path)
        self.adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        self.adj_orig.eliminate_zeros()

    def topK(self, node_type, node, k=10):
        """
        @param type: disease | lncRNA
        @param node: node-name
        @param k: top-K
        @return: dict()
        """
        node_id = self.nodeList["index_{}".format(node_type)][node]

        if node_type == 'disease':
            name = "index_{}".format('lncRNA')
            node_to_list = np.array(list(self.nodeList[name].values()))
        else:
            name = "index_{}".format('disease')
            node_to_list = np.array(list(self.nodeList[name].values()))

        params = tf.convert_to_tensor(self.hidden_emb)
        lookup_embeding_params = tf.nn.embedding_lookup(
            params=params,
            ids=node_to_list,
            max_norm=None,
            name=None)

        node_emb = tf.nn.embedding_lookup(params=params, ids=node_id, max_norm=None, name=None)
        scores = tf.math.sigmoid(tf.matmul(lookup_embeding_params, tf.reshape(node_emb, (-1, 1))))

        top_k = tf.nn.top_k(tf.reshape(scores, (1, -1))[0], k, sorted=True)

        res = node_to_list[top_k[1].numpy()]
        topKindex = {}
        items = list(self.nodeAll.items())
        for i in range(len(res)):
            key = items[res[i]][0]
            topKindex[key] = top_k[0][i].numpy()

        return topKindex

    def predict(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # 内积
        adj_rec = np.dot(self.hidden_emb, self.hidden_emb.T)
        adj_rec = sigmoid(adj_rec)
        return self.adj_orig, adj_rec


    def match(self, k):

        # 1. get all disease to lncRNA:list
        df = pd.read_csv(self.lnc_Dis_path, sep=",")
        lnc_dis = df.groupby('disease')['lncRNA'].apply(list).to_dict()

        # 2. get pre lncRNA
        res = dict()
        for name in lnc_dis.keys():
            res[name] = self.topK('disease', name, k)

        res_df = dict()
        for i in lnc_dis.keys():
            cell = [0, 0]
            v1 = lnc_dis[i]
            v2 = res[i].keys()

            cell[0] = len(v1)
            cell[1] = len(set(v1) & set(v2))
            res_df[i] = cell

        # 3. plot
        df = pd.DataFrame(res_df).T
        df.columns = ["num","k/K"]
        df = df.rename_axis('index').reset_index()

        df.plot(kind='bar', x="index", y= ["num","k/K"], color=["orange", "red"])
        plt.title("The num of Disease match for lncRNA", fontsize="16", fontweight="bold")
        plt.xlabel("Disease", fontweight="bold")
        plt.ylabel("Amount", fontweight="bold")
        plt.xticks([])

        f = plt.gcf()
        f.subplots_adjust(left=0.2, bottom=0.2)
        plt.show()
        return res_df


if __name__ == '__main__':
    m = ModelEvidence()
    m.load_model()
    m.match(k=10)

    node_name = "PCAT6"
    predlist = m.topK('lncRNA', node_name, 10)
    print("Node:{}".format(node_name), "\n", predlist)