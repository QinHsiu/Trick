"""
    author: Qinhsiu
    date: 2022/11/09
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def hard_sigmoid(x):
    res=[]
    for i in x:
        if i<-1*2.5:
            res.append(0)
        elif i>2.5:
            res.append(1)
        else:
            res.append(0.2*i+0.5)
    return np.array(res)

def tanh(x):
    return (np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))


def softsign(x):
    return x/(1+np.abs(x))

def relu(x):
    x_=copy.deepcopy(x)
    x_[x<0]=0
    return x_

def gelu(x):
    return x*sigmoid(1.702*x)

def softplus(x):
    return np.log(np.exp(x)+1)

def trelu(x,theta):
    x_=copy.deepcopy(x)
    x_[x<theta]=0
    return x_

def leaktrelu(x,alpha):
    x_=copy.deepcopy(x)
    x_[x<0]=alpha*x_[x<0]
    return x_


def elu(x,alpha):
    res=[]
    for i in x:
        if i<0:
            res.append(alpha*(np.exp(i)-1))
        else:
            res.append(i)
    return res


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def swish(x):
    return x*sigmoid(x)


def plot_act(x,names,i):
    plt.plot(range(len(x)), x,label=names[i])
    plt.legend()
    plt.show()
    # plt.savefig("./pic/%s.png" % names[i])


if __name__ == '__main__':
    x=np.array([(-1)**(i)*i for i in range(10)])
    print(x)
    names=["linear","exponent","sigmoid","hard_sigmoid","tanh","sfotsign","relu","gelu","soft_plus","thresholdedrelu","leakyrelu","elu","softmax","swish"]
    i=0
    theta=0.5
    alpha=0.5

    liear_x=x
    exp_x=np.exp(x)
    sigmoid_x=sigmoid(x)
    hard_s=hard_sigmoid(x)
    tanh_x=tanh(x)
    softsign_x=softsign(x)
    relu_x=relu(x)
    gelu_x=gelu(x)
    softplus_x=softplus(x)
    t_relu_x=trelu(x,theta)
    l_relu_x=leaktrelu(x,alpha)
    elu_x=elu(x,alpha)
    softmax_x=softmax(x)
    swish_x=swish(x)

    res=[liear_x,exp_x,sigmoid_x,hard_s,tanh_x,softsign_x,relu_x,gelu_x,softplus_x,t_relu_x,l_relu_x,elu_x,softmax_x,swish_x]

    for i in range(len(names)):
        plot_act(res[i],names,i)











