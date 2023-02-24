#!/usr/bin/env python
"""
Created on 02 17, 2023

model: main.py

@Author: Stormzudi
"""

from Code.MGATE.utils import BuildDataset, buildDataLoder, \
    getFeature, SimpleMSELoss, bulid_graph
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
index, adj_c, adj_l, adj_m, adj_d, adj_ld, adj_lm, adj_md = bd.getData(path)
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


# 5 下游通过 cross 来实现分类
def downModel(z, path):

    index, edges_index, node_attr, label = bulid_graph(path)

    # 1 load dataset
    train_data, val_data, test_data = buildDataLoder(node_attr, edges_index, y=label)
    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1))

    # add edges (train_edge, neg_edge)
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0).numpy()

    out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1).detach().numpy()

    train_data_d, test_data_d, train_labels_d, test_labels_d = train_test_split(out, edge_label,
                                                                                test_size=0.1, random_state=50)

    # fit and predict
    import lightgbm as lgb
    lgb_train = lgb.Dataset(train_data_d, train_labels_d)
    params = {'num_leaves': 48,
              'min_child_samples': 20,
              'min_data_in_leaf': 0,
              'objective': 'binary',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_sum_hessian_in_leaf": 15,
              "boosting": "gbdt",
              "feature_fraction": 0.3,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 1,
              "lambda_l1": 0.3,  # l1
              'lambda_l2': 0.01,  # l2
              "verbosity": -1,
              "nthread": -1,
              'metric': {'binary_logloss', 'auc'},
              "random_state": 1,
              "n_estimators": 200,
              "max_bin": 425,

              }
    gbm = lgb.train(params, lgb_train)
    pred = gbm.predict(test_data_d)

    # save auc score
    data_ = pd.DataFrame({
        'pred': pred,
        'labels': test_labels_d
    })
    date = datetime.date.today()
    data_.to_csv("{}_MGATE_0.91.csv".format(date))


a = 1
index_ld = index["index_lncRNA"].copy()
index_ld.update(index["index_disease"])

z = out_cross_fusion[list(index_ld.values())]

downModel(z, path)





