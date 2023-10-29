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



（1）LDASAGE是两阶段模型。第一阶段：使用预训练图卷积模型（Deepwalk、node2vec）、和与miRNA的邻接关系得到了表征lncRNA的向量L1、L2, 表征disease的向量D1、D2。第二阶段：使用了SAGE框架，实现merge 和 aggregation 两个步骤。最后通过机器学习模型包括xgboost, lightgbm等实现二分类。

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/ldasage.jpg)



（2）模型的算法步骤

![image](https://github.com/Stormzudi/LDASAGE/blob/master/images/steps.jpg)





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

