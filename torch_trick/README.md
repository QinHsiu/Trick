# Torch常用的trick

### 1. 训练过程中指定GPU编号

```python
import os
# 使用单块GPU
gpu_id=str(0)
os.envion["CUDA_VISIBLE_DEVICE"]=gpu_id
# 使用多块GPU
gpu_list=[0,1,2]
gpu_ids=",".join(map(str,gpu_list))
os.envion["CUDA_VISIBLE_DEVICE"]=gpu_ids
```

### 2. 查看模型每一层的输出情况

```python
from torchsummary import summary
summary(model,input_size=(channels,H,W))
```

### 3. 梯度裁剪

```python
import torch.nn as nn
output=model(data)
loss=loss_func(output,target)
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), # 模型参数
                         max_norm=20, # 梯度的最大范数
                         norm_type=2 # 规定范数类型，默认为L2
                        )
optimizer.step()
```

### 4. 张量升维

```python
import torch
a=torch.randn((1,2,3))
print(a.shape)

# 使用view
a=a.view(1,*a.size())

# 使用unsqueeze
a=a.unsqueeze(0)

# 使用squeeze
a=a.squeeze(0)

# 使用numpy 
import numpy as np
a=a[np.newaxis,:,:,:]
```

### 5. 独热编码

```python
# 按照类别，对原始数据的标签进行独热编码
import torch
class_num=8
batch_size=4
def one_hot(label):
	label=label.resize_(batch_size,1)
	m_zeros=torch.zeros(batch_size,class_num)
	one_hot=m_zeros.scatter_(1,label,1)
	return onehot.numpy()
label=torch.LongTensor(batch_size).random_()%class_num
```

### 6. 防止模型验证时爆显存

```python
with torch.no_grad():
	pass
```

### 7. 学习率衰减

```python
import torch.optim as optim
from torch.optim import lr_scheduler

# 初始化
optimizer=optim.Adam(model.parameters(),lr=0.001)
scheduler=lr_scheduler.StepLR(optimizer,10,0.1) # 每10个epoch，衰减一次

for n in iter_epoch:
	scheduler.step()
```

### 8. 冻结模型中的参数

```python
for param_name,value in model.named_parameters():
	value.requires_grad=False
```

### 9. 对不同模型使用不同的学习率

```python
optimizer=optim.Adam([{'model':model.parameters(),'lr':0.01},{'model1':model1.parameters(),'lr':0.1}])
```

### 10. 随机数字设置为42

```python
import random
import numpy as np
import torch
import os
seed=42
random.seed(seed)
os.environ["PYTHONHASHSEED"]=str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
```

### 11. 数据加载部分，使用多线程

```python
import torch
torch.utils.data.DataLoader(num_workers=3,pin_memory=True)
```

### 12 .使用自动混合精度

```python
import torch
scaler=torch.cuda.amp.GradScaler()

for data,label in data_iter:
	optimizer.zero_grad()
	with torch.cuda.amp.autocast():
		loss=model.loss_func(data)
	scaler.scale(loss).backward()
	
	scaler.step(optimizer)
	scaler.update()
```

### 13. 使用cudNN基准

```python
# 该基准适用于模型架构保持不变，输入大小不变的场景
torch.backends.cudnn.benchmark=True
```

### 14. 梯度积累（作用效果类似于增大batch_size）

```python
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulate
```

### 15. 设置梯度为None，而不是0

```python
model.zero_grad(set_to_none=True)
```

### 16. 使用as_tensor

```python
import torch
# 如果需要转换一个numpy数组，使用以下操作来避免复制tensor
torch.as_tensor()
torch.from_numpy()
```

### 17. 热启动
#### 在训练一定数量epoch之后，修改学习率（先使用较小的学习率进行训练，然后再使用大的学习率进行训练）

### 18. 根据batch size来设置学习效率
$$r=0.1*\frac{b}{256}$$

### 19. label smoothing
#### 人工标注数据中存在不同人标注的结果不一致的情况,因此需要模型降低一点对于标签的信任（过度依赖）
$$q_{i}=\left\\{\begin{array} 1\mbf{-} \epsilon& i=y \\\\ \frac{\epsilon}{K-1}& otherwise\end{array}\right.$$

### 20. 知识蒸馏
$$\mathcal{L}=\mathcal{L}(p,softmax(z))+T^{2}\mathcal{L}(softmax(\frac{r}{T}),softmax(\frac{z}{T}))$$

### 21. mixup 训练
$$\left\\{\begin{array}\hat{x}=\lambda x_{i}+(1-\lambda)x_{j} \\\\ \hat{y}= \lambda y_{i}+(1-\lambda)y_{j}\end{array}\right.$$

### 其他
#### (1) 度量学习相关函数: [传送门](https://mp.weixin.qq.com/s/NagauCb6zEJMeCEJx3A27w)
