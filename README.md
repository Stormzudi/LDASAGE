## **LDASAGE**：基于图卷积网络lncRNA-disease链路预测



**目录：**

1. 数据集
2. LDASAGE模型框架
3. 代码



### 1 数据集

使用了 四个公共数据集 中的lncRNA ,miRNA, disease相关数据。通过整合后得到了训练集和两个验证集。

| 数据集名称       | 描述                                               | 网站                                                         |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| LncRNADisease    | 实验验证支持的人类Disease与lncRNA的关联数据        | http://www.cuilab.cn/lncrnadisease                           |
| MNDR             | 实验验证支持的人类Disease与lncRNA、miRNA的关联数据 | http://www.rna-society.org/mndr/download.html                |
| LncRNASNP2       | 实验验证支持的人类miRNA与lncRNA的关联数据          | [http://bioinfo.life.hust.edu.cn/lncRNASNP/#!/](http://bioinfo.life.hust.edu.cn/lncRNASNP/) |
| Disease Ontology | 提供一系列语义描述疾病和疾病的功能                 | http://disease-ontology.org/                                 |



训练集和验证集的数据情况：

|                         | Database1 | Database2  (外部数据集) | Database3  (外部数据集) |
| ----------------------- | --------- | ----------------------- | ----------------------- |
| lncRNA                  | 350       | 95                      | 304                     |
| miRNA                   | 233       | 257                     | 263                     |
| disease                 | 478       | 172                     | 465                     |
| Pair(lncRNA  - disease) | 18789     | 1180                    | 4852                    |
| Pair(lncRNA  - miRNA)   | 2796      | 1352                    | 3176                    |
| Pair(miRNA  - disease)  | 2971      | 11255                   | 22013                   |





### 2 LDASAGE模型框架



（1）图卷积SAGE网络

​     LDASAGE是两阶段模型。第一阶段：使用预训练图卷积模型（Deepwalk、node2vec）、和与miRNA的邻接关系得到了表征lncRNA的向量L1、L2, 表征disease的向量D1、D2。第二阶段：使用了SAGE框架，实现merge 和 aggregation 两个步骤。最后通过机器学习模型包括xgboost, lightgbm等实现二分类。

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/ldasage.jpg)

优点：

- 基于直推式预训练算法得到节点表征L_1和D_1；
- 基于miRNA关联数据得到节点表征L_2和D_2；



（2）节点嵌入向量拼接

​     初始化构建节点向量L_1和L_2以及D_1和D_2的权重系数w ，以实现节点外部向量和预训练向量进行拼接。

在模型训练时，网络图中节点的两种向量表征都会随模型迭代训练，并达到最优权重系数w^∗ 。

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/ldasage2.jpg)



（3）网络层结构

​      在训练深度图卷积网络SAGE时，直接对整个网络进行梯度下降会导致输出层的误差在多层反向传播中逐渐减小，即产生梯度消失现象。为了解决这个问题，可以在每个阶段的输出上计算损失，并对这些损失值进行加权求和后反向传播，以确保底层参数得到正常更新。这种方法中，这些网络层中计算的损失也被称为辅助损失。具体而言，在SAGE的前向传播网络图中，如图3-5所示。

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/SAGE.jpg)



（2）模型的算法步骤

首先，根据外部数据miRNA-lncRNA，miRNA-疾病矩阵，计算lncRNA和disease的特征矩阵L_1和D_1。基于直推式算法，通过在建立的lncRNA-疾病网络拓扑图上进行有偏的随机游走，得到了节点的遍历序列，经过word2vec得到了节点的embedding，计算lncRNA-疾病节点矩阵L_2和D_2。然后，构建融合L1和  L2以及 D1 和 D2 的权重系数 w 。初始化权重系数w  ，实现节点外部向量和预训练向量进行拼接，在模型训练阶段，参数  w会跟随SAGE模型框架一起迭代训练，并得到最优的参数权重 w* 。然后，计算lncRNA-疾病网络中节点被采样权重  di 。通过节点有偏采样避免度高的节点被频繁采样，从而导致无法学习度小的节点向量等问题。最后，基于提出的图卷积网络算法SAGE提取lncRNA和疾病的主要特征矩阵得到  L和  D。然后，基于机器学习模型拟合SAGE网络全连接最后一层网络M。添加下游机器学习分类器，有效的增加了模型分类的准确性。最后，计算lncRNA和疾病的相关系数  r。通过比较传统的机器学习模型包括LightGBM, XGBoost, SVM, 随机森林等模型计算最优的lncRNA和疾病相关系数，并通过计算AUC比较模型的精确率。使用10折交叉验证来评估各个模型的预测效果。

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/steps.jpg)



（3）模型对比结果

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/result.png)



### 3 代码

一些脚本

```tex
1. main.py 主函数入口
2. ldasage.py 模型框架
3. model/gcn_model.py 实现SAGE模型框架
4. model/pre_model.py 实现预训练的两种方法
5. model/down_model.py 实现下游机器学习的方法
6. cofig.py 一些关于路径、存储、参数的配置
7. evidence.py 模型结果评估，实现输入lncRNA或者Disease后得到topK的相关Disease和lncRNA.
8. log_handler.py 日志记录
9. utils.py 一些函数的定义
```



不同实现方法

- python main.py 
- python main.py -down_model "xgb" 
- python main.py -pre_model "deepwalk" -num_features 233 128 -weight 0.7 0.3 
- python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3 
- python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3 -down_model "xgb" 
- python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3 -down_model "lgb" 
- python main.py -down_model "xgb" -epoch 10








Refer:

1. https://blog.csdn.net/weixin_43872709/article/details/123679423
2. https://github.com/datawhalechina/team-learning-nlp/tree/master/GNN/Markdown%E7%89%88%E6%9C%AC
3. modules fuse: https://blog.csdn.net/TDD_Master/article/details/110401964

