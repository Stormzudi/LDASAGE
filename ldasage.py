#!/usr/bin/env python
"""
Created on 07 09, 2022

model: ldasage.py

@Author: Stormzudi
"""

# load dataset
import os
import torch
from model.pre_model import Node2Vec, DeepWalk
from model.gcn_model import NET, DBNET
from model.down_model import downRF, downLightgbm, downSVM, downXgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config import lncRNA_miRNA, miRNA_disease
from config import index


class LDASage(object):
    def __init__(self, num_features, hidden_channels, out_channels, dropout, weight,
                 pre_model=None, down_model=None):
        self.pre_model = pre_model
        self.down_model = down_model
        self.path_lncRNA_miRNA = lncRNA_miRNA
        self.path_miRNA_disease = miRNA_disease

        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.weight = weight

    def model(self):
        """
        :return:
        """
        if not self.pre_model:
            model = NET(in_channels=self.num_features,
                        hidden_channels=self.hidden_channels,
                        out_channels=self.out_channels,
                        dropout=self.dropout)
        else:
            model = DBNET(in_channels=self.num_features,
                          hidden_channels=self.hidden_channels,
                          out_channels=self.out_channels,
                          weight=self.weight,
                          dropout=self.dropout)

        return model

    def preModel(self, path, node_attr):
        """
        :param path:
        :param node_attr:
        :return:
        """
        if not self.pre_model:
            return node_attr

        # step1.1 get lncRNA embedding
        df_lm = pd.read_csv(os.path.join(path, self.path_lncRNA_miRNA))
        G1 = nx.from_pandas_edgelist(df_lm, "miRNA", "lncRNA")

        # step2.1 get disease embedding
        df_md = pd.read_csv(os.path.join(path, self.path_miRNA_disease))
        G2 = nx.from_pandas_edgelist(df_md, "miRNA", "disease")

        print("The pre_model is {}".format(self.pre_model))
        if self.pre_model == "deepwalk":
            model_l = DeepWalk(G1, walk_length=64, num_walks=80, emb_name="lncRNA", workers=6)
            model_d = DeepWalk(G2, walk_length=64, num_walks=80, emb_name="disease", workers=6)
        elif self.pre_model == "node2vec":
            model_l = Node2Vec(G1, walk_length=64, num_walks=80, emb_name="lncRNA",
                               p=0.25, q=4, workers=6, use_rejection_sampling=0)
            model_d = Node2Vec(G2, walk_length=64, num_walks=80, emb_name="disease",
                               p=0.25, q=4, workers=6, use_rejection_sampling=0)
        else:
            raise NameError('No define optimization function name!')
        # step 1.2
        model_l.train(window_size=6, iter=3)
        embedding_l = model_l.get_embeddings(nodes=index['index_lncRNA'])

        # step 2.2
        model_d.train(window_size=6, iter=3)
        embedding_d = model_d.get_embeddings(nodes=index['index_disease'])

        # step3 aggregate to be a new node_feature
        embed = torch.cat((torch.tensor(embedding_l, dtype=torch.float32),
                           torch.tensor(embedding_d, dtype=torch.float32)), 0)

        # step 4 concat
        node_attr = torch.cat((node_attr, embed), dim=1)

        return node_attr

    def downModel(self):
        if not self.down_model:
            return

        print("The down model is: {}".format(self.down_model))
        if self.down_model == "RF":
            model = downRF()

        elif self.down_model == "lgb":
            model = downLightgbm()

        elif self.down_model == "xgb":
            model = downXgb(booster='gbtree', eta=0.03, colsample_bytree=0.7,
                            learning_rate=0.1, n_estimators=50, max_depth=8, min_child_weight=5, gamma=0.8, seed=1024)

        elif self.down_model == "svm":
            model = downSVM()
        else:
            raise NameError('No define optimization function name!')

        return model
