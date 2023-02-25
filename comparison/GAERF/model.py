#!/usr/bin/env python
"""
Created on 02 15, 2023

model: model.py

@Author: Stormzudi
"""

"""
Paper: Inferring LncRNA-disease associations based on graph autoencoder matrix completion
REF: https://github.com/sheng-n/MGATE/blob/main/MGATE.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index, model="Train"):
        out = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        if model != "Train":
            out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1).detach().numpy()
        return out



class GAMCLDA(object):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.):
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        pass

    def model(self, ):
        model = GCN(
            num_features=self.num_features
            , hidden_channels=self.hidden_channels
            , num_classes=self.num_classes
        )
        return model