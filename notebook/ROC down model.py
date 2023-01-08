#!/usr/bin/env python
"""
Created on 09 18, 2022

model: ROC down model.py

@Author: Stormzudi
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

path_edge = '../result/test_auc_edge_2022-09-18.csv'
path_out = '../result/test_auc_out_2022-09-18.csv'
edge_label = pd.read_csv(path_edge)
out = pd.read_csv(path_out)


X_train, X_test, y_train, y_test = train_test_split(out, edge_label, test_size=0.1, random_state=50)

# 1. 使用不同的算法计算AUC
#set up plotting area
plt.figure(0).clf()

#fit logistic regression model and plot ROC curve
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

#调用模型包
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, max_depth=3, min_samples_split=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random Forest, AUC="+str(auc))


import lightgbm as lgb
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=200,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=150, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="LightGBM, AUC="+str(auc))






#add legend
plt.legend()
plt.show()