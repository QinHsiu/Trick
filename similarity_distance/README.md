# 常用的相似性度量方法

## 1. 欧氏距离

$$d(a,b)=\sqrt{\sum^{I}\limits_{i=1}(x_i-y_i)^{2}}, a=(x_1,x_2,...,x_{I}),b=(y_1,y_2,...,y_{I})$$

## 2. 曼哈顿距离

$$d(a,b)=\sum^{I}\limits_{i=1}|x_i-y_i|,a=(x_1,x_2,...,x_{I}),b=(y_1,y_2,...,y_{I})$$

## 3. 切比雪夫距离

$$d(a,b)=\max(|x_1-y_1|,|x_2-y_2|,...,|x_{I}-y_{I}|),a=(x_1,x_2,...,x_{I}),b=(y_1,y_2,...,y_{I})$$

## 4. 闵可夫斯基距离

$$d(a,b)=\sqrt[p]{\sum^{I}\limits_{i=1}|x_{i}-y_{i}|^{p}},a=(x_{1},x_{2},...,x_{I}),b=(y_{1},y_{2},...,y_{I})$$

### 当p=1的时候，其值等于曼哈顿距离；当p=2的时候，其值等于欧氏距离；当p趋于正无穷时，其值等于切比雪夫距离

## 5.标准化欧式距离: 需要用到方差

$$d(a,b)=\sqrt{\sum^{I}\limits_{i=1}\frac{(x_{1i}-x_{2i})^{2}}{Var(x_{i})})},a=(x_{11},x_{12},...,x_{1i}),b=(x_{21},x_{22},...,x_{2i})$$

## 6. 马氏距离: 需要用到协方差

$$d(a)=\sqrt{(x-\mu)^{T}{\sum}^{-1}(x-\mu)},a=(x_1,x_2,...x_{I})$$

$$d(a,b)=\sqrt{(x-y)^{T}{\sum}^{-1}(x-y)},a=(x_1,x_2,...,x_{I}),y=(y_1,y_2,...,y_{I})$$

## 7. 夹角余弦

$$d(a,b)=1-\frac{x_1* x_2+y_1* y_2}{\sqrt{x_1^2+y_1^2}*\sqrt{x_2^2+y_2^2}},a=(x_1,y_1),b=(x_2,y_2)$$

$$d(a,b)=1-\frac{\sum^{I}\limits_{i=1}x_i*y_i}{\sqrt{\sum^{I}\limits_{i=1}x_i^2}*\sqrt{\sum^{I}\limits_{i=1}y_{i}^2}},a=(x_1,x_2,...,x_{I}),b=(y_1,y_2,...,y_{I})$$

## 8. 汉明距离: 使用了异或

$$d(a,b)=\sum^{I}\limits_{i=1} x_i\oplus y_i, a=(x_1,x_2,...,x_{I}),b=(y_1,y_2,...,y_{I})$$

## 9. 杰卡德距离&杰卡德相似系数

$$J(a,b)=\frac{|a \cap b|}{|a \cup b|}, 杰卡德系数$$

$$J_{d}(a,b)=1-J(a,b)=\frac{|a\cup b|-|a \cap b|}{a \cup b},杰卡德距离$$

## 10. 相关系数&相关距离

$$D(x_i,x_j)=\sqrt{(x_i-x_j)^{T}(x_i-x_j)}$$

$$r(a,b)=\frac{Cov(X,Y)}{\sqrt{D(X)D(Y)}}=\frac{E((X-EX)(Y-EY))}{\sqrt{D(X)\sqrt{D(Y)}}}, 相关系数$$

$$d(a,b)=1-r(a,b),相关距离$$

## 11. 信息熵

$$Entropy(a)=\sum^{I}\limits_{i=1}-p_{i}\log_{2}p_i,a=(p_1,p_2,...,p_{I}) $$

## 12. KL散度

$$d_{KL}(p||q)=\sum^{I}\limits_{i=1}p(x_i)(\log p(x_i)-\log q(x_i)), p,q表示两个分布$$

### 度量两个分布之间的相似性（距离）

## 13. JL散度

$$d_{JL}(p||q)=\frac{1}{2}KL(p||m)+\frac{1}{2}KL(q||m), m=\frac{1}{2}(p+q)$$

### 有一种MIX的思想，将两个分布混合得到一个新的分布，然后考虑原始两个分布与新的分布的相似性

## 14 Wasserstein距离

$$d_{w}(p,q)=\mathop{info}\limits_{\gamma\sim\amalg(p,q) }\mathbb{E}_{(x,y)\sim\gamma}[||x-y||], \amalg(p,q)表示两个分布组合起来的所有可能的联合分布的集合$$

#### 对于每一个可能的联合分布gamma，可以从中采样一部分得到样本x和y，并计算这对样本的距离||x-y||，所以可以计算该联合分布下，样本对距离的期望值。在所有可能的联合分布中能够对这个期望值取到下界，也即Wasserstein距离。

## 15. 半正矢距离，Haversine distance

$$d(\lambda,\phi)=2r \arcsin(\sqrt{{\sin}^{2}(\frac{\phi_{2}-\phi_{1}}{2})+\cos\phi_1\cos\phi_2{\sin}^{2}(\frac{\lambda_2-\lambda_1}{2})})$$

### 该距离衡量球面上两点之间的距离，其中r为球面半径，phi与lambda分别为经度和纬度

## 16. Sorensen-Dice指数

$$d(a,b)=\frac{2|a\cap b|}{|a|+|b|}$$

### 该指数类似于9中的Jaccard指数，可以衡量样本集的相似性和多样性，计算的是两个集合的重叠百分比

