#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot.py
# @Author: Stormzudi
# @Date  : 2022/4/16 14:00

import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
from configparser import ConfigParser
import matplotlib.pyplot as plt

def draw_network():
    '''
    加载邻接矩阵，这里利用networkx读取文件，生成图和邻接矩阵
    生成的节点的编号是根据节点在文件中出现的顺序进行编号
    :param sample_data_path:
    :return:
    '''
    # load config file
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    config = ConfigParser()
    config.read(config_path)
    section = config.sections()[0]

    # data catalog path
    data_catalog = config.get(section, "data_catalog")

    # train file path
    train_file_name = config.get(section, "train_file_name")

    sample_data_path = os.path.join(data_catalog, train_file_name)
    g = nx.read_edgelist(sample_data_path)
    nx.draw(g)
    plt.show()


if __name__ == '__main__':
    draw_network()