#!/usr/bin/env python
"""
Created on 08 28, 2022

model: 03 ROC pre model.py

@Author: Stormzudi
"""

"""
ROC demo
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 组装（连接）
y = torch.cat((y0, y1), 0).type(torch.LongTensor)

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


net = Net(2, 10, 2)

optimizer = torch.optim.SGD(net.parameters(), lr=0.012)
for t in range(200):
    out = net(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (t + 1) % 10 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]  # 在第1维度取最大值并返回索引值
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accu=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


