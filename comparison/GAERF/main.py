#!/usr/bin/env python
"""
Created on 02 15, 2023

model: main.py

@Author: Stormzudi
"""
from Code.GAMCLDA.utils import bulid_graph, buildDataLoder
from Code.GAMCLDA.model import GAMCLDA
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import datetime
from sklearn.ensemble import RandomForestClassifier

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
index, edges_index, node_attr, label = bulid_graph(path)


# 1 load dataset
train_data, val_data, test_data = buildDataLoder(node_attr, edges_index, y=label)

# 2 define model
gamclda = GAMCLDA(
    num_features=args['num_features'],hidden_channels=args['out_channels'],num_classes=args['n_class_size'])
model = gamclda.model()

# 3 define optimization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

log = dict()

# 4 train
def train():
    model.train()

    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1))

    # add edges (train_edge, neg_edge)
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )

    # label
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)

    loss = criterion(out, edge_label)
    train_auc = roc_auc_score(edge_label.numpy(), out.detach().numpy())
    loss.backward()
    optimizer.step()

    log.update({"z":z,
                "edge_label_index":edge_label_index,
                "edge_label":edge_label})

    return loss, train_auc

@torch.no_grad()
def test(data):
    model.eval()
    z = model(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

# ======================
# ==== Train  =======================
best_val_auc = final_test_auc = 0
for epoch in range(1, args["epoch"]):
    loss, train_auc = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc

    print(f'Epoch: {epoch:03d}, '
          f'Loss: {loss:.4f}, '
          f'train: {train_auc:.4f}, '
          f'Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')


z = log["z"]
edge_label_index = log["edge_label_index"]
edge_label = log["edge_label"]
out = model.decode(z, edge_label_index, model="RF")
train_data_d, test_data_d, train_labels_d, test_labels_d = train_test_split(out, edge_label,
                                                                            test_size=0.1, random_state=50)
# 下游使用 random forest
downModel = RandomForestClassifier(n_estimators=5,
                                   max_features="auto",random_state=5)
downModel.fit(train_data_d, train_labels_d)
pred = downModel.predict_proba(test_data_d)[:, 1]
train_auc = roc_auc_score(test_labels_d, pred)

# save auc score
data_ = pd.DataFrame({
    'pred': pred,
    'labels': test_labels_d
})
date = datetime.date.today()
data_.to_csv("{}_GAERF_{:.4f}.csv".format(date, train_auc))
