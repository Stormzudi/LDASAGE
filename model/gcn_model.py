#!/usr/bin/env python
"""
Created on 05 16, 2022

model: gcn_model.py

@Author: Stormzudi
"""

import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Linear
import torch.nn.functional as F


class GCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize: bool = False):
        super(GCNConv, self).__init__(aggr='add', flow='source_to_target')
        # "Add" aggregation (Step 5).
        # flow='source_to_target' 表示消息从源节点传播到目标节点
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.normalize = normalize

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)
        # return self.propagate(edge_index, x=x, norm=norm, deg=deg.view((-1, 1)))

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.reshape(-1, 1) * x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        # print("self.aggr", self.aggr)
        # print("aggregate is called")
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x, norm):
        print('`message_and_aggregate` is called')


class DBNET(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 weight,
                 dropout=0.):
        super(DBNET, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))  # embedding of [miRNA, node2vec]
        self.dropout = dropout
        self.convs_1 = torch.nn.ModuleList()  # model_1
        self.convs_2 = torch.nn.ModuleList()  # model_2
        self.in_channels_1 = in_channels[0]
        self.in_channels_2 = in_channels[1]

        for i in range(len(hidden_channels)):
            self.convs_1.append(GCNConv(self.in_channels_1, hidden_channels[i]))
            self.convs_2.append(GCNConv(self.in_channels_2, hidden_channels[i]))
            self.in_channels_1 = self.in_channels_2 = hidden_channels[i]
        self.convout = GCNConv(hidden_channels[-1], out_channels)
        # self.result = Linear(out_channels*2, 128)

    def forward(self, x, edge_index):
        # embedding of miRNA
        x1, x2 = x[0], x[1]
        for i, l in enumerate(self.convs_1):
            x1 = self.convs_1[i](x1, edge_index)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # embedding of node2vec
        for i, l in enumerate(self.convs_2):
            x2 = self.convs_2[i](x2, edge_index)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # normalization and  attention
        w1 = torch.exp(self.weight[0]) / torch.sum(torch.exp(self.weight))
        w2 = torch.exp(self.weight[1]) / torch.sum(torch.exp(self.weight))

        # weight and concat
        concat = torch.cat((w1 * self.convout(x1, edge_index)
               , w2 * self.convout(x2, edge_index)), dim=1)

        # result
        # result = self.result(concat)
        # result = F.dropout(result, training=self.training, p=0.1)
        return concat

    def decode(self, z, edge_label_index, model="Cosine"):
        # modules： RF, LightGBM, GBDT, Cosine, SVM
        out = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        if model != "Cosine":
            out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1).detach().numpy()

        return out

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    

class NET(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.):
        super(NET, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(len(hidden_channels)):
            self.convs.append(GCNConv(in_channels, hidden_channels[i]))
            in_channels = hidden_channels[i]
        self.convout = GCNConv(hidden_channels[-1], out_channels)
        self.output = Linear(out_channels, 64)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # embedding of miRNA
        for i, l in enumerate(self.convs):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # convout layer
        out = self.convout(x, edge_index)

        # linear layer
        out = self.output(out)
        # out = F.dropout(out, training=self.training, p=0.2)
        return out

    # def decode(self, z, edge_label_index):
    #     return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    #

    def decode(self, z, edge_label_index, model="Cosine"):
        # modules： RF, LightGBM, GBDT, Cosine, SVM
        out = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        if model != "Cosine":
            out = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1).detach().numpy()

        return out

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
