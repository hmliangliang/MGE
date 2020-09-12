# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\12\9 0009 10:36:36
# File:         MGE.py
# Software:     PyCharm
#------------------------------------
import numpy as np
import multiprocessing
import My_Metrics
import math
import copy
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,fowlkes_mallows_score
from sklearn.decomposition import PCA
import My_Metrics
import time

def matrix_multiply(data,t):
    for i in range(t):
        data = data.dot(data)
    data = 1/math.pow(2,t)*data
    return data

def compute_weight(data,M,tau,i,lam):
    '''
    data: 图的邻接矩阵,
    M: 图的高阶邻接矩阵
    tau: 阈值
    i: 节点的编号
    lam:对应于原文的lambda参数,用于调节
    return: 返回节点i的邻接节点,及其相应的权值
    '''
    result = dict(obj = [],weight = [])#obj保存为对象, weight保存为权值
    for j in  range(data.shape[1]):
        sim = np.dot(M[i,:],M[j,:])/math.sqrt(np.linalg.norm(M[i,:],2)*np.linalg.norm(M[i,:],2))#节点i与节点j的余弦夹角
        if sim >= tau and i!=j:
            result['obj'].append(j)#保存对象
            w = sim*(1/(1+lam*math.exp(-(M[j,i] + sum(data[j,:])))))
            result['weight'].append(w)  # 保存权值
        else:
            result['obj'].append(j)  # 保存对象
            w = 0
            result['weight'].append(w)#保存权值
    return [i,result]

def SVT(A,t):#执行的是Singular Value Thresholding
    '''参数说明
    A: 输入的数据矩阵
    t:Singular Value Thresholding的阈值
    '''
    U,Sigma,VT = np.linalg.svd(A,full_matrices=0)
    for i in range(len(Sigma)):
        if Sigma[i] > t:
            Sigma[i] = Sigma[i] - t
        else:
            Sigma[i] = 0
    return np.dot(U,np.dot(np.diag(Sigma),VT))


def update_E(A,alpha):#执行的更新参数E,即求解L2,1范数的优化问题
    '''
    A: 输入的矩阵,对应于原问题中的F-范右矩阵
    alpha: 阈值
    return: 返回
    '''
    W = np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[1]):#对于每一列
        num = np.linalg.norm(A[:,i],2)
        if num > alpha:
            if num < np.inf:
                W[:,i] = (num - alpha)/num*A[:,i]
            else:
                W[:, i] = A[:, i]
    return W


if __name__ == '__main__':
    start = time.time()
    thisdata = np.loadtxt('./data sets/wisconsin/wisconsin-data.txt',dtype=int)
    label = np.loadtxt('./data sets/wisconsin/wisconsin-labels.txt',dtype=int)
    label = label[:,1]
    t = 4
    lamda = 0.001
    tau = 0.5
    N = max(max(thisdata[:,0]),max(thisdata[:,1]))+1#样本点的数目
    data = np.zeros((N,N))#获得邻接矩阵
    D = copy.deepcopy(data)#度矩阵
    for i in range(thisdata.shape[0]):
        data[thisdata[i,0],thisdata[i,1]] = 1
        data[thisdata[i,1],thisdata[i,0]] = 1
    del thisdata
    for i in range(N):
        D[i,i] = sum(data[:,i])
    A = np.linalg.pinv(D).dot(data)
    pool = multiprocessing.Pool()
    res = []
    for i in range(t):#分别计算每一阶
        res.append(pool.apply_async(matrix_multiply,(A,i+1)))
    pool.close()
    pool.join()
    M = 0#
    for matrix_data in res:#计算矩阵的各阶之和
        M = M + matrix_data.get()
    del pool
    del res
    for i in range(M.shape[0]):#对M值进行归一化
        value = sum(M[i,:])
        for j in range(M.shape[1]):
            M[i,j] = M[i,j]/(value+0.0001)
    pool = multiprocessing.Pool()
    result = []
    for i in range(N):#对于每一个节点,并行计算相应的节点权值
        result.append(pool.apply_async(compute_weight,(data,M,tau,i,lamda)))
    pool.close()
    pool.join()
    W = np.zeros((N,N))#保存邻域的权值
    for res in result:
        ress = res.get()
        i = ress[0]#获取当前的顶点
        for j in range(len(ress[1]['obj'])):
            W[i,ress[1]['obj'][j]] = ress[1]['weight'][j]
    #计算非对称的拉普拉斯矩阵
    D = np.zeros((N, N)) #获得度矩阵
    for i in range(N):
        D[i,i] = sum(W[:,i])
    L = W - D
    '''
    采用优化方法来优化求解最终的矩阵,采用增广拉格朗日法算法和对偶梯度上升法求解
    min tr(Y'LY)+alpha/2||X||_{F}^{2}+beta||Y||*+gamma||E||_{2,1}
    s.t. M=XY+E, Y=J
    '''
    #初始化参数值
    u = 0.000001
    umax = 100000
    alpha = 1.8
    beta = 0.8
    gamma = 5
    p = 1.1
    e = 0.00000001
    d = 64#降维后的数据
    #初始化矩阵
    Y = 1.1*np.random.rand(d,N)#初始化节点的低维嵌入向量
    J = 0.8*np.random.rand(d,N)
    X = 1.1*np.random.rand(N,d)#上下文嵌入向量
    E = 0.0001*np.random.rand(N,N)
    V = np.random.rand(N,d)
    eta = np.random.rand(N,N)
    while np.linalg.norm(M-np.dot(X,Y)-E,np.inf)>e and np.linalg.norm(Y-J,np.inf)>e:
        #固定其他参数更新J
        J = SVT(Y+V.transpose()/u,beta/u)
        #固定其他参数更新X
        X = np.dot(np.dot(eta.transpose(),Y.transpose()) + u*np.dot(M,Y.transpose()) - u*np.dot(E,Y.transpose()),np.linalg.inv(alpha*np.eye(d,d)-u*np.dot(Y,Y.transpose())))
        #固定其他参数更新Y
        Y = np.dot(np.linalg.inv(u*np.dot(X.transpose(),X)+u*np.eye(d,d)),np.dot(X.transpose(),eta.transpose()) + V.transpose() - np.dot(Y,L) - np.dot(Y,L.transpose()) + u*J + u*np.dot(X.transpose(),M) - u*np.dot(X.transpose(),E))
        # 固定其他参数更新E
        E = update_E(eta.transpose()/u+M-np.dot(X,Y),gamma/u)
        #固定其他参数更新乘子
        eta = eta + u*(M-np.dot(X,Y)-E).transpose()
        V = V + u*(Y-J).transpose()
        #更新参数u
        u = min(p*u,umax)
    #可以将Y\in R^{d*n}看低维嵌入,X\in R^{n*d}属于上下文向矩阵
    Y = Y.transpose()#将Y转化成n*d的数据,每一行代表一个样本
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X.transpose())
    X = X.transpose()
    Y = scaler.fit_transform(Y.transpose())
    Y = Y.transpose()
    W = np.zeros((N,N))
    #传播网络节点的结构
    for _ in range(10):
        for i in range(N):
            for j in range(N):
                W[i, j] = 1 / (1 + math.exp(-np.dot(X[i, :], Y[j, :])))
        for i in range(N):  # 归一化
            dsum = sum(W[i, :])
            for j in range(N):
                W[i, j] = W[i, j] / dsum
        for i in range(N):
            Y_m = np.zeros((1,d))[0]
            for j in range(N):
                Y_m = Y_m + W[i,j]*X[j,:]
            X[i,:] = (X[i,:]+Y_m)/2
    X = (X + 0.8*np.dot(M,np.random.rand(N,d)))/1.8
    k = len(np.unique(label))
    print('聚类的类簇数目k=',k)
    model = KMeans(n_clusters=k)
    model.fit(X)
    cluster_label = model.labels_ #获取聚类的结果
    #调整类簇的聚类结果
    for i in range(k):
        temp = []#记录当前节点的标签改变后的模块度值
        temp_label = copy.deepcopy(cluster_label)
        modularity1 = My_Metrics.Modularity(data,cluster_label)
        for j in range(k):
            temp_label[np.argwhere(cluster_label==i)[:,0]] = j
            modularity2 = My_Metrics.Modularity(data, temp_label)#计算当前节点的标签改变为j的模块度值
            temp.append(modularity2)#保存模块度
        if max(temp) > modularity1:#模块度增大
            cluster_label = copy.deepcopy(temp_label)
    print('ARI=', adjusted_rand_score(label, cluster_label))
    print('NMI=', normalized_mutual_info_score(label, cluster_label))
    print('FM=', fowlkes_mallows_score(label, cluster_label))
    print('Purity=', My_Metrics.Purity(label, cluster_label))
    print('modularity=', My_Metrics.Modularity(data, cluster_label))
    end = time.time()
    print('运行的时间消耗为', end - start)

'''
计算方法参考文献：
1. Liu, Guangcan, et al. "Robust recovery of subspace structures by low-rank representation." IEEE transactions on pattern analysis and machine intelligence 35.1 (2012): 171-184.
2. Liu, Guangcan, Zhouchen Lin, and Yong Yu. "Robust subspace segmentation by low-rank representation." 2010 ICML.
'''