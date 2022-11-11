"""
    author: Qinhsiu
    date: 2022/11/09
"""

import numpy as np
from scipy.spatial import distance


# 欧氏距离
def Euclidean_distance(a,b):
    # 调包
    # print(distance.euclidean(a,b))
    return np.sqrt(np.sum((a-b)**2))

# 曼哈顿距离
def Manhattan_distance(a,b):
    # 调包
    # print(distance.cityblock(a,b))
    return np.sum(np.abs(a-b))

# 切比雪夫距离
def Chebyshev_distance(a,b):
    # 调包
    # print(distance.chebyshev(a,b))
    return np.max(np.abs(a-b))

# 闵可夫斯基距离
def Minkowski_distance(a,b,p):
    # 调包
    # print(distance.minkowski(a,b,p))
    return (np.sum(np.abs(a-b)**p))**(1/p)

# 标准化欧氏距离
def S_Euclidean_distance(a,b):
    c=np.array([a[0],b[0]])
    return np.sqrt(np.sum((a-b)**2/np.var(c,axis=0)))

# 马氏距离
def Marginal_distance(a,b):
    X=np.vstack([a[0],b[0]])
    XT=X.T
    SI=np.linalg.inv(np.cov(X)) # 协方差矩阵的逆矩阵
    n=XT.shape[0]
    d1=[]
    # 两两组合，假若有10个样本，故有(10)*(10-1)/2=45个结果
    for i in range(0,n):
        for j in range(i+1,n):
            delta=XT[i]-XT[j]
            d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))
            d1.append(d)
    return d1

# 夹角余弦
def Cosin(a,b):
    # 调包
    # print(distance.cosine(a,b))
    return 1-np.sum(a*b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))

# 汉明距离
def Hanming_distance(a,b):
    # 调包
    # print(distance.hamming(a,b))
    return 1-np.sum(np.logical_xor(a,b))

# 杰卡德距离
def Jacquard_distance(a,b):
    # 调包
    # print(distance.jaccard(a,b))
    # 杰卡德相似系数
    j_=np.sum(np.where(a==b))/len(np.union1d(a,b))
    return 1-j_

# 相关距离
def Correlation_distance(a,b):
    # 相关系数
    r_=np.corrcoef(a,b)
    return 1-r_

# 信息熵
def Entropy_distance(a):
    return np.sum(-1*a*np.log(a))

# KL散度
def KL_diatcne(a,b):
    return np.sum(a*(np.log(a)-np.log(b)))

# JL散度
def JL_distance(a,b):
    m=(a+b)/2
    return 1/2*(KL_diatcne(a,m)+KL_diatcne(b,m))

# Wasserstein距离
from scipy.stats import wasserstein_distance
def Wasserstein(a,b):
    return wasserstein_distance(a,b)


# Dot product距离
def Dot_distance(a,b):
    return np.dot(a,b)

# 半正矢距离
from sklearn.metrics.pairwise import haversine_distances
def Haversine_distance(a,b):
    return haversine_distances([a,b])

# Sorensen-Dice指数
def Sorensen_distacne(a,b):
    # 调包
    # print(distance.dice(a,b))
    return distance.dice(a,b)

if __name__ == '__main__':
    a_=np.random.random((1,10))
    b_=np.random.random((1,10))

    print("a=",a_)
    print("b=",b_)

    ou=Euclidean_distance(a_,b_)
    print("欧式距离=",ou)

    man=Manhattan_distance(a_,b_)
    print("曼哈顿距离=",man)

    che=Chebyshev_distance(a_,b_)
    print("切比雪夫距离=",che)

    mink=Minkowski_distance(a_,b_,1)
    print("闵可夫斯基距离=",mink)

    s_ou=S_Euclidean_distance(a_,b_)
    print("标准化欧氏距离=",s_ou)

    ma=Marginal_distance(a_,b_)
    print("马氏距离=",ma)

    cosin=Cosin(a_,b_)
    print("余弦距离=",cosin)

    hanm=Hanming_distance(a_,b_)
    print("汉明距离=",hanm)

    corr=Correlation_distance(a_[0],b_[0])
    print("相关距离=",corr)

    dot=Dot_distance(a_[0],b_[0])
    print("点积距离=",dot)

    entropy=Entropy_distance(a_)
    print("信息熵=",entropy)

    kl=KL_diatcne(a_,b_)
    print("KL散度=",kl)

    jl=JL_distance(a_,b_)
    print("JL散度=",jl)

    jaccard=Jacquard_distance(a_,b_)
    print("Jaccard距离=",jaccard)

    wasserstein=wasserstein_distance(a_[0],b_[0])
    print("Wasserstein距离=",wasserstein)

    haversine=Haversine_distance(a_[0][:2],b_[0][:2])
    print("Haversine距离=",haversine)

    sorensen=Sorensen_distacne(a_[0],b_[0])
    print("Sorensen距离=",sorensen)












