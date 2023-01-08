#!/usr/bin/env python
"""
Created on 08 28, 2022

model: getAlldis2lnc.py

@Author: Stormzudi
"""
import numpy as np
import pandas as pd
import os
from config import save_path, data_path, lncRNA_miRNA, DATA
import matplotlib.pyplot as plt

# 1. load the result embedding
embeds_path = os.listdir(save_path)
path = os.path.join(save_path, embeds_path[-1])
hidden_emb = np.load(path)

# 2. load adj 原始邻接矩阵，除去对角线元素
path_ = os.path.join(data_path, DATA.get('database'), DATA.get("dataset").get("train"))
lnc_Dis_path = os.path.join(path_, "lncRNA_disease_association.csv")

# 3. 原始数据的关联表
df = pd.read_csv(lnc_Dis_path, sep=",")
data = df.groupby(['disease'])['lncRNA'].apply(list)
data = data.reset_index()
data.columns = ['disease', 'hist_lncRNA']

# 4. 模型预测
from evidence import ModelEvidence
m = ModelEvidence()
m.load_model()

result = data[['disease']].copy()
pre_data = []
for node_name in result['disease']:
    predlist = m.topK('disease', node_name, 50)
    pre_data.append(list(predlist.keys()))

result['pre_hist_lncRNA'] = pre_data


# 5. merge
data = pd.merge(data, result, on='disease')

def matchlnc2dis(list1, list2):
    temp = set(list1 + list2)
    return len(list1) + len(list2) - len(temp)

data['value'] = data.apply(lambda row:matchlnc2dis(row['hist_lncRNA'], row['pre_hist_lncRNA']), axis=1)
data['len_hist_disease'] = data.apply(lambda row:len(row['hist_lncRNA']), axis=1)
data['len_pre_hist_lncRNA'] = data.apply(lambda row:len(row['pre_hist_lncRNA']), axis=1)

data.to_csv(os.path.join(os.getcwd(), "lncRNAdiseaseall.csv"))
a = 1


