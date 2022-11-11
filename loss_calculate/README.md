# 深度学习常用Loss

## 1. L1 Loss, 也叫均方绝对误差

## $$Loss(pred,y)=|y-pred|$$

## 2. L2 Loss, 也叫均方误差

## $$Loss(pred,y)=\sum(y-pred)^{2}$$

## 3. 负对数似然，NLL

## $$Loss(pred,y)=-\log(pred)$$

## 4. Corss-Entropy, 交叉熵

## $$Loss(pred,y)=-\sum(y\log(pred))=NLL(\log(softmax(pred)),y)$$

### 交叉熵损失=softmax+log+NLL

## 5. Hinge Embedding

## $$Loss(pred,y)=\max(0,1-y* pred),y \in (-1,1)$$

### 用于判断两个向量的相似程度，常用于非线性的词向量学习以及半监督学习

## 6. Margin Ranking Loss

## $$Loss(pred,y)=\max(0,-y*(pred1-pred2)+margin)$$

## 7. Triplet Margin Loss

## $$Loss(a,p,n)=\max(0,d(ai,pi)-d(ai,ni)+margin)$$

### a表示anchor，p表示正样本，n表示负样本

## 8. KL Divergence Loss

## $$Loss(pred,y)=y* (\log(y)-pred)$$

## 9. BPR Loss

## $$L(p,n)=-\frac{1}{N}\sum^{N}\limits_{j=1}\log\sigma(p-n_j)$$

### p表示正样本，N表示负样本数目，nj表示第j个负样本

## 10. InfoNCE

## $$L(p)=-\log\frac{\exp(sim(p,k_{+})/\tau}{\sum^{K}\limits_{i=0}\exp(sim(p,k_{i})/\tau)}$$

### sim()相似度计算，可以采用点积、余弦相似度等（更多相似性度量参看距离度量，[参看]([Trick/similarity_distance at main · QinHsiu/Trick (github.com)](https://github.com/QinHsiu/Trick/tree/main/similarity_distance))）,tau是温度系数