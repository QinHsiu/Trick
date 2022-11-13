# 深度学习常用激活函数

## 1.线性激活函数

$$f(x)=x$$

![avator](pic/linear.png)

## 2. 自然指数激活函数

$$f(x)=e^{x}$$

![avator](pic/exponent.png)

## 3. Sigmoid激活函数

$$f(x)=\frac{1}{1+e^{-x}}$$

![avator](pic/sigmoid.png)

## 4. Hard_sigmoid激活函数

$$f(x)=\left\{\begin{array}{**lr**}0 &x\textless -2.5\\0.2*x+0.5&-2.5 \leq x\leq2.5\\1&x\textgreater 2.5\\ \end{array}\right.$$

![avator](pic/hard_sigmoid.png)

## 5. 双曲正切激活函数

$$f(x)=\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$

![avator](pic/tanh.png)

## 6. Softsign激活函数

$$f(x)=softsign(x)=\frac{x}{1+|x|}$$

![avator](pic/softsign.png)

## 7. ReLU激活函数

$$f(x)=relu(x)=\max(0,x)$$

![avator](pic/relu.png)

## 8. GeLU激活函数

$$f(x)=x\sigma(1.702x)=0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^{3}))$$

![avator](pic/gelu.png)

## 9. Softplus激活函数

$$f(x)=softplus(x)=\log(e^{x}+1)$$

![avator](pic/softplus.png)

## 10. ThresholdedReLU激活函数

$$f(x)=ThresholdedReLU(x,\theta)=\max(x,\theta)=\left\{\begin{array}{**lr**}0&x\leq \theta\\x&x\textgreater\theta\end{array}\right.$$

![avator](pic/thresholdedrelu.png)

## 11. LeakyReLU激活函数

$$f(x)=LeakyReLU(x,\alpha)=\left\{\begin{array}{**lr**}\alpha*x&x\textless0\\x&0\leq x\end{array}\right.$$

![avator](pic/leakyrelu.png)

## 12.  ELU激活函数

$$f(x)=elu(x,\alpha)=\left\{\begin{array}{**lr**}\alpha(e^{x}-1)&x\textless0\\x&0\leq x\end{array}\right.$$

![avator](pic/elu.png)

## 13. Softmax激活函数

$$f(x)=softmax(x)=\frac{e^{x_i}}{\sum^{n}\limits_{j=0}e^{x_j}}$$

![avator](pic/softmax.png)

## 14. Swish激活函数

$$f(x)=x*sigmoid(x)$$

![avator](pic/swish.png)