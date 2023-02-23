#!/usr/bin/env python
"""
Created on 02 17, 2023

model: main.py

@Author: Stormzudi
"""

from Code.MGATE.utils import BuildDataset, buildDataLoder, getFeature, SimpleMSELoss
from Code.MGATE.mgate import MGATE
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import datetime

# ========================== Args ========================
args = {
    "num_edges": 18789
    , "num_features": 233
    , "num_nodes": 828
    , "hidden_channels": [256]
    , "out_channels": 128
    , "n_class_size": 64
    , "lncRNA-miRNA": "../../dataset/lncRNA_miRNA_association.csv"
    , "disease-miRNA": "../../dataset/miRNA_disease_association.csv"
    , "weight": [0.7, 0.3] # [miRNA, node2vec]
    , "dropout": 0.0
    , "epoch": 10
}

path = "../../dataset/valdata/"
bd = BuildDataset()
adj_c, adj_l, adj_m, adj_d, adj_ld, adj_lm, adj_md = bd.getData(path)
adj_c, adj_l, adj_m, adj_d, adj_ld, adj_lm, adj_md = \
    torch.from_numpy(adj_c).float(), \
    torch.from_numpy(adj_l).float(), \
    torch.from_numpy(adj_m).float(),\
    torch.from_numpy(adj_d).float(),\
    torch.from_numpy(adj_ld).float(),\
    torch.from_numpy(adj_lm).float(),\
    torch.from_numpy(adj_md).float()

features_c = torch.from_numpy(getFeature(adj_c)).float()
features_l = torch.from_numpy(getFeature(adj_l)).float()
features_m = torch.from_numpy(getFeature(adj_m)).float()
features_d = torch.from_numpy(getFeature(adj_d)).float()
features_ld = torch.from_numpy(getFeature(adj_ld)).float()
features_lm = torch.from_numpy(getFeature(adj_lm)).float()
features_md = torch.from_numpy(getFeature(adj_md)).float()

hidden_c = hidden_intra = hidden_inter = 700
n_class_c = n_class_intra = n_class_inter = 128
input_l, input_m, input_d = 350, 233, 478
input_c = input_l + input_m + input_d
input_ld = input_l + input_d
input_lm = input_l + input_m
input_md = input_m + input_d
input_att, output_att, decoder_size = 3, input_c, 128


model = MGATE(
    input_c, hidden_c, n_class_c, input_l, input_m, input_d, hidden_intra, n_class_intra,
    input_ld, input_lm, input_md, hidden_inter, n_class_inter, input_att, output_att, decoder_size
)

de_out_c, de_out_l, de_out_m, de_out_d, de_out_ld, de_out_lm, de_out_md, de_out_em, out_cross_fusion = \
    model.forward(adj_c, features_c, adj_l, features_l, adj_m, features_m, adj_d, features_d,
                adj_ld, features_ld, adj_lm, features_lm, adj_md, features_md)


# 3 define optimization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()


# 4 train
def train(adj_c):
    model.train()
    criteria = SimpleMSELoss()

    optimizer.zero_grad()
    de_out_c, de_out_l, de_out_m, de_out_d, de_out_ld, de_out_lm, de_out_md, de_out_em, out_cross_fusion = \
        model.forward(adj_c, features_c, adj_l, features_l, adj_m, features_m, adj_d, features_d,
                      adj_ld, features_ld, adj_lm, features_lm, adj_md, features_md)

    loss = criteria(out_cross_fusion, adj_c)
    loss.backward()
    optimizer.step()

    return loss



for epoch in range(1, args["epoch"]):
    loss = train(adj_c)

    print(f'Epoch: {epoch:03d}, '
          f'Loss: {loss:.4f}, ')


a = 1
