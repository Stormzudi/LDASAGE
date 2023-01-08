```python
"""
Refer:
1. https://blog.csdn.net/weixin_43872709/article/details/123679423
2. https://github.com/datawhalechina/team-learning-nlp/tree/master/GNN/Markdown%E7%89%88%E6%9C%AC
3. modules fuse: https://blog.csdn.net/TDD_Master/article/details/110401964

"""
```





```python
"""
    1. bulid_graph
    2. buildDataLoder
    3. create_alias_table
    4. alias_sample
    5. partition_num
    6. preprocess_nxgraph
"""
```


> python main.py 
> python main.py -down_model "xgb"
python main.py -pre_model "deepwalk" -num_features 233 128 -weight 0.7 0.3  
python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3  
python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3 -down_model "xgb"  
python main.py -pre_model "node2vec" -num_features 233 128 -weight 0.7 0.3 -down_model "lgb"  


> python main.py -down_model "xgb" -epoch 10
