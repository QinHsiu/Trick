# Tips for Tuning Parameters

##### 1.做新模型的时候，最开始不要加激活函数，不要加batchnorm，不要加dropout，先就纯模型。然后再一步一步的实验，不要过于信赖经典的模型结构(除非它是预训练的)，比如加了dropout一定会有效果，或者加了batchnorm一定会有提升所以先加上，首先你要确定你的模型处于什么阶段，到底是欠拟合还是过拟合，然后再确定解决手段。

##### 2.如果是欠拟合，直接模型宽度深度增加，一般2倍递增或者数据集增强，可以用大量[数据增强方式](https://github.com/QinHsiu/Trick/tree/main/data_augmentation)。另外考虑增加relu，swish等作为某些层的[激活函数](https://github.com/QinHsiu/Trick/tree/main/activate_function)不过在此做之前建议最好把每一层的参数打印出来，看是否符合正态分布，一般来说relu可能有奇效，but最后一层千万不要relu，因为relu可能把很多数值都干为0，所以使用relu需要慎重，如果效果严重下降，建议看看relu的参数分布。

##### 3.如果过拟合，首先是dropout，然后batchnorm，过拟合越严重dropout+bn加的地方就越多，有些直接对embedding层加，有奇效。另外因为引入了dropout导致存在模型的不确定性，可以在输出侧引入R-drop进行约束。

##### 4.对于数据量巨大的推荐系统的模型来说使用较少的epoch足矣，再多就会过拟合。

##### 5.不要盲目使用正则化（l2 normal等），可以比较加正则化与不加正则化的结果，[常用正则化技术](https://github.com/QinHsiu/Trick/tree/main/normalization)。

##### 6.特征不要一次性堆叠，可以逐步添加（有的特征是没有用的）。

##### 7.学习率最好是从高到底2倍速度递减一般从0.01开始。

##### 8.对于稀疏特征多的模型采用adagrad，稠密特征多的模型用adam。

##### 9.粗排用精排后topk做负样本，基本是有效果的。

##### 10.batch size对于推荐来说32-64-128-512测试效果再高一般也不会正向了，再低训练太慢了。

##### 11.对于负样本太多的数据集，测试集的loss降并不代表没有过拟合，试试看看f1或者auc有没有降低，因为有可能负样本学的爆好，所以loss降低，但是正样本凉了。[常用损失函数](https://github.com/QinHsiu/Trick/tree/main/loss_calculate)。

##### 12.对于长文本来说longformer的效果高于各种bert变体。

##### 13.对于图像和nlp，效果一直不提高，可以尝试自己标注一些模型经常分错的case，然后加入训练会有奇效。

##### 14. 对于推荐序列来说pooling和attention的效果差别真的不大，基本没有diff。

##### 15.对于推荐来说senet和lhuc，是深度学习领域为数不多的可以观察特征重要性的模型结构。

##### 16.一般不要尝试用强化学习优化，效果很一般。

##### 17.bert不要太过于相信cls的embedding能力，还是要看看它有没有做过相关任务，特别对于文本匹配场景。

##### 18.序列特征才用layernorm，正常用batchnorm。

##### 19. 推荐召回用softmax效果比sigmoid更好，意思就是召回更适合对比学习那种listwise学习方式。

##### 20.召回负采样负样本效果肯定高于曝光未点击负样本（也可以采用[相似度](https://github.com/QinHsiu/Trick/tree/main/similarity_distance)来进行负样本的筛选）。

##### 21. 参数初始化用xavier和truncated_normal可以加速收敛，但是，同样是tensorflow和pytorch用同样的初始化，pytorch可能存在多跑一段时间才开始收敛的情况，所以，如果出现loss不下降的情况，可以多跑几个epoch，当然可以用tensorflow实现一把，看看效果是不是一样。

##### 22.对于推荐系统的模型，可以使用热启动，一般要embedding+DNN全部热启动，然后仅对倒数第一层作扰动，或者类似wide&deep，新加一个结构对最后一层个性化扰动，因为，扰动太多又负向了。

##### 23. 对于nlp任务，采用embedding扰动的方式有奇效，比如Fast Gradient Method（FGM）和Projected Gradient Descent（PGD）。

##### 24.推荐多目标任务可以考虑采用Gradient Surgery的方式。

##### 25.Focal loss对于极大不平衡的数据集确实有奇效，其中gamma因子可以成10倍数衰减。

##### 26. textcnn, fasttext, dpcnn, textrnn在短文本的分类效果基本差不多，但是fasttext和textcnn明显比dpcnn快很多。

##### 27.对于embedding可视化的问题最好是先用pca降维，然后采用t-sne进行可视化，但是，对于类别很多的情况，个可以使用肉眼抽样观察，t-sne只能对于类目不多的情况的embedding进行可视化。

##### 28.. 显存不够用的时候，gradient checkpointing可以起到降低显存的效果。

##### 29. 对于机器阅读任务，在bert层后加bi-attention或者coattention有奇效。

##### 30.在推荐系统中，神经网络模型对于连续数值的处理是不如xgb的，所以，最好对连续数值分箱，一般等频分箱足矣，最好还是观察数据分布，把outlier单独一箱，如果还想完美一点可以，用IV值看看分箱的优劣。







