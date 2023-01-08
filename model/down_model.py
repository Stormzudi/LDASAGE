#!/usr/bin/env python
"""
Created on 07 09, 2022

model: down_model.py

@Author: Stormzudi
"""

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.svm import SVC
import xgboost as xgb


def downXgb(booster='gbtree', eta=0.03, colsample_bytree=0.7,
            learning_rate=0.1, n_estimators=50, max_depth=8, min_child_weight=5, gamma=0.1, seed=1024):

    params = {
        'booster': booster,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'gamma': gamma,
        'max_depth': max_depth,
        'alpha': 0,
        'lambda': 0,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': min_child_weight,
        'silent': 0,
        'eta': eta,
        'nthread': -1,
        'seed': seed,
    }

    model = xgb.XGBClassifier(**params)
    return model


def downLightgbm():
    model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=200,
                               max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                               min_child_weight=5, min_child_samples=150, subsample=0.8, subsample_freq=1,
                               colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
    return model


def downRF():
    model = RandomForestClassifier()
    return model


def downSVM():
    model = SVC(C=5, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
    return model
