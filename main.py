#!/usr/bin/env python
"""
Created on 05 16, 2022

model: train.py

@Author: Stormzudi
"""
# load dataset
import argparse
import datetime
import os
import torch
import numpy as np
import pandas as pd
from utils import buildDataLoder, bulid_graph
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from config import data_path, DATA, save_path
from ldasage import LDASage
from log_handler import logs
date = datetime.date.today()
parser = argparse.ArgumentParser(description='Link Prediction with LDASAG')

# setups in pre&down the training set
parser.add_argument('-pre_model', type=str, default=None,
                    help='pre-model for the dataset: deepwalk, node2vec')
parser.add_argument('-down_model', type=str, default=None,
                    help='down-model for the dataset: lgb, xgb, dnn, lr, cos')

parser.add_argument('-num_edges', type=int, default=1180,
                    help='the num of edges')
parser.add_argument('-num_features', default=233, nargs='+', type=int,
                    help='the num of features')
parser.add_argument('-num_nodes', type=int, default=267,
                    help='the num of nodes')
parser.add_argument('-hidden_channels', type=int, default=[256, 128],
                    help='the hidden-channels of model.')
parser.add_argument('-out_channels', type=int, default=64,
                    help='the out-channels of output.')

# pathway of miRNA, disease
parser.add_argument('-path_dataset', type=str, default=data_path,
                    help='dir of dataset')
parser.add_argument('-dataset', type=str, default=DATA.get('database'),
                    help='name of dataset')
parser.add_argument('-train', type=str, default=DATA.get("dataset").get("train"), choices=["traindata", "valdata"],
                    help='train or val')
parser.add_argument('-lncRNA_miRNA', type=str, default='lncRNA_miRNA_association.csv',
                    help='lncRNA-miRNA dataset')
parser.add_argument('-miRNA_disease', type=str, default='miRNA_disease_association.csv',
                    help='disease-miRNA dataset')

# Model and Training
parser.add_argument('-epoch', type=int, default=100, help='epoch of train')
parser.add_argument('-dropout', type=int, default=0.0)
parser.add_argument('-weight', type=float, default=[0.7, 0.3], nargs='+',
                    help='the weight of pre_model and GCN: [miRNA, node2vec]')
args = parser.parse_args()

logs.info(args)
# ========================== Load dataset  =======================
path = os.path.join(args.path_dataset, args.dataset, args.train)
index, edges_index, node_attr, label = bulid_graph(path)

# ========================== Model  =======================
LDAModel = LDASage(num_features=args.num_features, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                   dropout=args.dropout, weight=args.weight, pre_model=args.pre_model, down_model=args.down_model)
node_attr = LDAModel.preModel(path, node_attr)  # 更新node的embedding
model = LDAModel.model()  # 返回GCN模型
downModel = LDAModel.downModel()  # 返回下游模型

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# ========================== train and val data  =======================
train_data, val_data, test_data = buildDataLoder(node_attr, edges_index, y=label)


# ========================== Train & Test  =======================
def train():
    model.train()

    optimizer.zero_grad()
    if not args.pre_model:
        z = model(train_data.x, train_data.edge_index)
    else:
        z = model([train_data.x[:, :233], train_data.x[:, 233:]], train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_index.size(1), method='sparse')

    # add edges (train_edge, neg_edge)
    edge_label_index = torch.cat(
        [train_data.edge_index, neg_edge_index],
        dim=-1,
    )

    edge_label = torch.cat([
        torch.ones(train_data.edge_index.size(1), dtype=torch.float32),
        torch.zeros(train_data.edge_index.size(1), dtype=torch.float32)
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)

    loss = criterion(out, edge_label)
    train_auc = roc_auc_score(edge_label.numpy(), out.detach().numpy())
    loss.backward()
    optimizer.step()

    return loss, train_auc


@torch.no_grad()
def test(data):
    model.eval()
    if not args.pre_model:
        z = model(data.x, data.edge_index)
    else:
        z = model([data.x[:, :233], data.x[:, 233:]], data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


# ========================== Train  =======================
best_val_auc = final_test_auc = 0
for epoch in range(1, args.epoch):
    loss, train_auc = train()
    val_auc = test(val_data)
    test_auc = test(test_data)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc

    logs.info("Epoch:{:03d}, Loss:{:.4f}, train:{:.4f}, val:{:.4f}, Test:{:.4f}" \
              .format(epoch, loss, train_auc, val_auc, test_auc))
    # print(f'Epoch: {epoch:03d}, '
    #       f'Loss: {loss:.4f}, '
    #       f'train: {train_auc:.4f}, '
    #       f'Val: {val_auc:.4f}, '
    #       f'Test: {test_auc:.4f}')

logs.info("Final Val:{:.4f}".format(best_val_auc))
logs.info("Final Test:{:.4f}".format(final_test_auc))
logs.info("Final Weight:{}".format(args.weight))
# print(f'Final Val: {best_val_auc:.4f}')
# print(f'Final Test: {final_test_auc:.4f}')
# print(f'Final Weight: {args.weight}')


# ========================== Down Model  =======================
if args.down_model:
    logs.info("the downModel: {}".format(args.down_model))
    if not args.pre_model:
        z = model(train_data.x, train_data.edge_index)
    else:
        z = model([train_data.x[:, :233], train_data.x[:, 233:]], train_data.edge_index)

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
    ], dim=0)

    out = model.decode(z, edge_label_index, model=args.down_model)
    train_data_d, test_data_d, train_labels_d, test_labels_d = train_test_split(out, edge_label,
                                                                                test_size=0.1, random_state=50)
    # fit and predict
    downModel.fit(train_data_d, train_labels_d)
    pred = downModel.predict(test_data_d)

    # AUC scores
    scores = roc_auc_score(pred, test_labels_d)
    logs.info("Final Val:{:.4f}".format(scores))
    # save auc score
    data_ = pd.DataFrame({
        'pred': pred,
        'labels': test_labels_d
    })
    data_.to_csv(os.path.join(save_path, "test_auc_{}.csv".format(date)))



# ========================== Save GCN models  =======================
if not args.pre_model:
    z = model(train_data.x, train_data.edge_index)
else:
    z = model([train_data.x[:, :233], train_data.x[:, 233:]], train_data.edge_index)
hidden_emb = z.data.cpu().numpy()
np.save(os.path.join(save_path, "lncRNA_disease_{}.npy".format(date)), hidden_emb)