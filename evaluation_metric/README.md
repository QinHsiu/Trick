# 推荐场景中常用的评价指标

## 1. AUC

## $$AUC=\frac{\sum_{i \in positiveClass}rank_{ins_{i}}-\frac{M(1+M)}{2}}{M\times N}$$

## 2. GAUC

##  $$GAUC=\frac{\sum_{u_{i}}w_{u_{i}\times AUC_{u_{i}}}}{\sum_{w_{u_{i}}}}$$

## 3. LogLoss

## $$LogLoss=-\frac{1}{N}\sum^{N}_{i=1}((y_{i}\log p_{i})+(1-y_{i})\log(1-p_{i}))$$

## 4. NDCG

## $$DCG=\sum^{p}_{i=1}\frac{2^{rel_i}-1}{\log_2 (i+1)}$$

## $$IDCG=\sum^{|REL|}_{i=1}\frac{2^{rel_i}-1}{\log_2 (i+1)}$$

## $$NDCG=\frac{DCG}{IDCG}$$

## 5. HitRate

## $$HR=\frac{\sum^{N}\limits_{i=1}\frac{Hit_{i}}{Rel_{i}}}{N}$$

## 6. MRR

## $$\frac{1}{N}\sum^{N}\limits_{i=1}\frac{1}{{rank}_{i}}$$

## 7. Recall

## $$Recall=\frac{TP}{TP+FN}$$

## Code [传送门](https://mp.weixin.qq.com/s/8MElwWpskzw0Dj5CPj-Trw)
