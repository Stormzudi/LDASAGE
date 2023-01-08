#!/usr/bin/env python
"""
Created on 05 16, 2022

model: utils.py

@Author: Stormzudi
"""

import os
import sys
from pathlib import Path


CURPATH = os.path.abspath(__file__)
DIRPATH = os.path.dirname(CURPATH)

data_path = os.path.join(DIRPATH, 'dataset/')
save_path = os.path.join(DIRPATH, 'result/')
model_path = os.path.join(save_path, 'model_save/')

cur_file_lab = Path(__file__).absolute()
parent_cur_dir = Path(cur_file_lab).parent
sys.path.append(str(parent_cur_dir))

# print(str(parent_cur_dir))
pair_names = ['lncRNA_disease', 'miRNA_disease', 'lncRNA_miRNA']
lncRNA_miRNA="lncRNA_miRNA_association.csv"
miRNA_disease="miRNA_disease_association.csv"
log_save_days = 100

DATA = {}


# index
index = {}
index_map = {}


# dataset
DATA.update({"database": "lnc-dis"})
DATA.update({"dataset": {"train": "traindata", "val": "valdata"}})