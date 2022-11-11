"""
    author: Qinhsiu
    date: 2022/11/09
"""
import numpy as np
import torch
import torch.nn as nn


# L1 Loss
def l1(pred,y):
    # 直接调包
    # print(nn.L1Loss()(torch.tensor(pred),torch.tensor(y)))
    return np.sum(np.abs(y-pred))/y.shape[0]


# L2 Loss
def l2(pred,y):
    # 直接调包
    # print(nn.MSELoss()(torch.tensor(pred),torch.tensor(y)))
    return np.sum((y-pred)**2)/y.shape[0]


# NLL Loss
def NLL(pred,y):
    # 直接调包
    # print(nn.NLLLoss()(torch.tensor(pred),torch.tensor(y,dtype=torch.long)))
    temp=-1*(pred[0,y[0]]+pred[1,y[1]]+pred[2,y[2]])
    return temp/3

# Cross-Entropy Loss
def CE(pred,y):
    # 直接调包
    # print(nn.CrossEntropyLoss()(torch.tensor(pred), torch.tensor(y, dtype=torch.long)))
    pred_=torch.log(torch.softmax(torch.tensor(pred),dim=-1))
    return NLL(pred_,y)

# Hing Embedding Loss
def HE(a,b):
    # 计算相似程度
    a=torch.randn((10,128))
    b=torch.randn((10,128))
    x=1-torch.cosine_similarity(a,b)
    y=2*torch.empty(10).random_(2)-1
    # 调包
    output=nn.HingeEmbeddingLoss()(x,y)
    return output


# Margin Ranking Loss
def MR(preds,y,margin):
    y=np.array([1 for i in range(10)])
    # 调包
    return nn.MarginRankingLoss()(torch.tensor(preds[0]),torch.tensor(preds[1]),torch.tensor(y))

# Triplet Margin Loss
def TM(a,p,n):
    # 直接调包
    return nn.TripletMarginLoss()(torch.tensor(a),torch.tensor(p),torch.tensor(n))

# KL Divergrnce Loss
def KL(pred,y):
    # 直接调包
    return nn.KLDivLoss(reduction="sum")(torch.tensor(pred),torch.tensor(y))


# BPR Loss
def BPR(p,n):
    # 直接调包
    p=torch.tensor(p)
    n=torch.tensor(n)
    return -1*torch.sum(torch.log(torch.sigmoid(p-n)))/p.shape[0]

# InfoNce Loss
def InfoNCE(p,k):
    p=torch.tensor(p)
    k=torch.tensor(k)
    return -1*torch.log(torch.exp(torch.matmul(p,k[0].t()))/torch.sum(torch.exp(torch.matmul(p,k.t()))))


if __name__ == '__main__':
    pred=np.random.random((10))
    pred_1 = np.random.random((10))
    y=np.random.random(10)

    l1_loss=l1(pred,y)
    print("L1 Loss=",l1_loss)

    l2_loss=l2(pred,y)
    print("L2 loss=",l2_loss)

    he_loss = HE(pred, y)
    print("HE Loss=", he_loss)

    mr_loss=MR([pred,pred_1],y,1)
    print("MR Loss=",mr_loss)

    triplet_loss=TM(pred,pred_1,y)
    print("TM Loss=",triplet_loss)

    kl_loss=KL(pred,y)
    print("KL Loss=",kl_loss)

    bpr_loss=BPR(pred,y)
    print("BPR Loss=",bpr_loss)

    p=np.random.random((1,10))
    k=np.random.random((10,10))
    info_loss=InfoNCE(p,k)
    print("INFONCE Loss=",info_loss)

    pred=np.random.random((3,5))
    y=np.array([0,1,2])

    nll_loss=NLL(pred,y)
    print("NLL Loss=",nll_loss)

    ce_loss=CE(pred,y)
    print("CE Loss=",ce_loss)












