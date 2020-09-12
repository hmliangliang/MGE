# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\12\3 0003 09:25:25
# File:         Modularity.py
# Software:     PyCharm
#------------------------------------
import math
import networkx as nx
import numpy as np


def Modularity(data,label):#该函数计算社交网络的模块度
    '''
    graph: 所形成的图,有networkx生成
    data: 边的集合
    label: 各个节点聚类结果的类标签
    return: modularity返回网络的模块度
    计算方法见https://www.cnblogs.com/xiaofanke/p/7491824.html
    '''
    e = data.shape[0]#边的数目
    modularity = 0
    num_nodes = label.shape[0]
    A = np.zeros((num_nodes,num_nodes))
    for i in range(data.shape[0]):
        A[int(data[i,0]),int(data[i,1])] = 1
        A[int(data[i,1]),int(data[i,0])] = 1
    for i in range(data.shape[0]):
        if label[int(data[i,1])] == label[int(data[i,0])]:
            modularity = modularity + 1/(2*e)*(1 - sum(A[int(data[i,0]),:])*sum(A[int(data[i,1]),:])/(2*e))
        else:
            modularity = modularity + 0
    return modularity

def Purity(label_real, label_pre):#计算聚类结果的纯度
    #label_real, label_pre均是一个行向量,形式如np.array([1, 2, 2, 4, 1, 2, 4, 1, 4, 2, 1])
    #https://zhuanlan.zhihu.com/p/53840697
    num_real = np.unique(label_real)#真实结果的标签种类
    num_pre = np.unique(label_pre)#聚类结果的标签种类
    P = 0#初始化纯度值
    for i in range(len(num_pre)):
        cluster_samples = set(np.argwhere(label_pre == num_pre[i])[:, 0])  # 寻找聚类标签中的每一类
        set_length = 0
        for j in range(len(num_real)):
            class_samples = set(np.argwhere(label_real == num_real[j])[:, 0])  # 寻找真实类标签中的每一类
            temp = len(cluster_samples.intersection(class_samples))#两者的交集
            if temp > set_length:#找到更大的公共序列
                set_length = temp
        P = P + set_length/len(label_real)
    return P