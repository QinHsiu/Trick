# 机器学习常用标准化方法

## 1. 最小最大标准化

$$f(x)=\frac{x-\min(x)}{\max(x)-\min(x)}$$

## 2. 零均值规范化

$$f(x)=\frac{x-\overline{x}}{\sigma},\overline{x}表示均值,\sigma表示标准差$$

## 3. 比例标准化

$$f(x)=\frac{x}{\sum x}$$

## 3. 简单标准化

$$f(x)=\frac{x}{x_{max}}$$

$$f(x)=\frac{\log_{10}x}{\log_{10}x_{max}}$$

## 4. 反正切函数转换

$$f(x)=\frac{2*\arctan(x)}{\pi}$$

# 深度学习中常用标准化方法

## 1. BN=>Batch Normalization

$$f(x)=\gamma\hat{x}+\beta$$

$$\hat{x}=\frac{x-\overline{x}}{\sqrt{{\sigma}^{2}+\epsilon}}$$

$${\sigma}^{2}=\frac{1}{m}\sum^{m}\limits_{i=1}(x_{i}-\mu)^{2}$$

$$\mu=\frac{1}{m}\sum^{m}\limits_{i=1}x_{i}$$

### 按照每次训练的小批量数据进行标准化，并且是针对特征维度进行的，例如输入的是一个三维向量（B,L,H），则是针对L*H做的标准化

## 2. LN=>Layer Normalization

$${\mu}^{l}=\frac{1}{H}\sum^{H}\limits_{i=1}a^{l}_{i},{\sigma}^{l}=\sqrt{\frac{1}{H}{\sum}^{H}\limits_{i=1}(a^{l}_{i}-{\mu}^{l}_{i})^{2}}, H表示隐藏单元的数目$$

## 3. WN=>Weight Normalization

$$y=\phi(w\cdot x+b),w表示线性函数的权重,b表示bias$$

$$w=\frac{g}{||v||}v,v是一个k维度向量,||v||表示欧几里得范数,g是一个张量$$

### 通过对网络参数w进行重参数化可以加速深度学习参数收敛速度，另外重写向量v是固定的，相较于BN而言会引入更少的噪音

## 4. IN=>Instance Normalization

$$y_{nijk}=\frac{x_{nijk}-\mu_{ni}}{\sqrt{{\sigma}^{2}_{ni}+\epsilon}}$$

$$\mu_{ni}=\frac{1}{HW}\sum^{W}\limits_{l=1}\sum^{H}\limits_{m=1}x_{nilm}$$

$${\sigma}^{2}_{ni}=\frac{1}{HW}{\sum}^{W}\limits_{i=1}\sum^{H}\limits_{m=1}(x_{nilm}-\mu_{ni})^{2}$$

$$N表示批量大小,x\in\mathbb{R}^{N\times C\times W\times H},x_{t}表示第t个元素,k和j为空间维度,i指特征通道,n批量数据中图片的位置$$

## 5. GN=>Group Normalization

$$\hat{x}_{i}=\frac{1}{\sigma_{i}}(x_i-\mu_{i})$$

$$\mu_{i}=\frac{1}{m}\sum\limits_{k \in S_{i}}x_{i},\sigma_{i}=\sqrt{\frac{1}{m}\sum\limits_{k \in S_{i}}(x_k-\mu_{i})^{2}+\epsilon_{i}}$$

$$S_{i}=\{k|k_{N}=i_{N},\left\lfloor\frac{k_{C}}{C/G}\right\rfloor=\left\lfloor\frac{i_{C}}{C/G}\right\rfloor\}$$

$$y_{i}=\gamma\hat{x}_{i}+\beta$$

$$\epsilon是一个很小的常量,S_{i}用来表示计算均值和方差构成的集合,m为该集合的大小,G表示组的数目,C/G表示每个组中通道的数目,\left\lfloor\cdot\right\rfloor表示向下取整$$

## 6. PN=>Position Normalization

$$\mu_{b,h,w}=\frac{1}{C}\sum^{C}\limits_{c=1}X_{b,c,h,w}$$

$$\sigma_{b,h,w}=\sqrt{\frac{1}{C}\sum^{C}\limits_{c=1}(X^{2}_{b,c,h,w}-\mu_{b,h,w})+\epsilon}$$

$$\epsilon表示一个很小的常量$$

## 7. CBN=>Conditional Batch Normalization

$$y=\frac{x-\mathbb{E}[x]}{\sqrt{Var[x]+\epsilon}}\cdot\gamma+\beta,一般的BN计算方式$$

$$y=\frac{x-\mathbb{E}[x]}{\sqrt{Var[x]+\epsilon}}\cdot\hat\gamma+\hat\beta,改进BN之后的CBN$$

$$\left\\{\begin{array}{**lr**}\hat\gamma=\gamma+\triangle\gamma\\\\ \hat\beta=\beta+\triangle\beta\end{array}\right.$$

## 8. CIN=>Conditional Instance Normalization

$$IN(x)=\gamma(\frac{x-\mu(x)}{\sigma(x)})+\beta$$

$$CIN(x)=\gamma^{s}(\frac{x-\mu(x)}{\sigma(x)})+\beta^{s}$$

## 9. AdaIN=>Adaptive Instance Normalization

$$AdaIN(x,y)=\sigma(y)(\frac{x-\mu(x)}{\sigma(x)})+\mu(y)$$

## 10. SPADE=>Spatially Adaptive Normalization

$$\gamma^{i}_{c,y,x}(m)=\frac{h^{i}_{n,c,y,x}-{\mu}^{i}_{c}}{{\sigma}^{i}_{c}}+{\beta}^{i}_{c,y,x}(m)$$

$${\mu}^{i}_{c}=\frac{1}{NH^{i}W^{i}}\sum\limits_{n,y,x}h^{i}_{n,c,y,x}$$

$${\sigma}^{i}_{c}=\sqrt{\frac{1}{NH^{i}W^{i}}\sum\limits_{n,y,x}((h^{i}_{n,c,y,x})^{2}-({\mu}^{i}_{c})^{2})}$$

## Reference

### [传送门](https://zhuanlan.zhihu.com/p/142866736)

