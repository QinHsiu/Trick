"""
    author: Qinhsiu
    date: 2022/11/09
"""

import numpy as np
import random
import copy

def Mask(seq,mask_ratio,n):
    """
    :param seq: 原始序列
    :param mask_ratio: 掩盖比例
    :param n: 原始序列长度
    :return: mask之后的序列
    """
    seq=copy.deepcopy(seq)
    mask_id=0
    mask_len=int(n*mask_ratio)
    if mask_len<1:
        return seq
    mask_idx=random.sample(range(n),mask_len)
    seq[mask_idx]=mask_id
    return seq

# 随机删除一部分item
def Crop_single(seq,crop_ratio,n):
    """
    :param seq: 原始序列
    :param crop_ratio: 剪切比例
    :param n: 原始序列长度
    :return: crop之后的序列
    """
    seq = copy.deepcopy(seq)
    crop_len=int(crop_ratio*n)
    if crop_len<1:
        return seq
    crop_idx=random.sample(range(n),crop_len)
    seq=np.delete(seq,crop_idx)
    return seq

# 随机删除一部分连续的item
def Crop_continue(seq,crop_ratio,n):
    """
   :param seq: 原始序列
    :param crop_ratio: 剪切比例
    :param n: 原始序列长度
    :return: crop之后的序列
    """
    seq = copy.deepcopy(seq)
    crop_len=int(crop_ratio*n)
    if crop_len<1:
        return seq
    start=random.randint(0,n-crop_len)
    if start+crop_len>=n:
        return seq[:start]
    elif start==0:
        return seq[start+crop_len:]
    else:
        sub_pre=seq[:start]
        sub_end=seq[start+crop_len:]
        return np.append(sub_pre,sub_end)


# 随机打乱一部分item顺序
def Reorder(seq,reorder_ratio,n):
    """
    :param seq: 原始序列
    :param reorder_ratio: 打乱比例
    :param n: 原始序列长度
    :return: reorder之后的序列
    """
    seq = copy.deepcopy(seq)
    reorder_len=int(reorder_ratio*n)
    if reorder_len<1:
        return seq
    start=random.randint(0,n-reorder_len)
    sub_seq=seq[start:start+reorder_len]
    random.shuffle(sub_seq)
    seq[start:start+reorder_len]=sub_seq
    return seq

# 随机替换一部分item
def Substitute(seq,sub_ratio,n,sim_dict):
    """
    :param seq: 原始序列
    :param sub_ratio: 替换比例
    :param n: 原始序列长度
    :param sim_dict: 一个字典，存储与每个item最相似的一些item
    :return: substitute之后的序列
    """
    seq = copy.deepcopy(seq)
    sub_len=int(sub_ratio*n)
    if sub_len<1:
        return seq
    sub_idx=random.sample(range(n),sub_len)
    for idx in sub_idx:
        seq[idx]=sim_dict[seq[idx]]
    return seq


# 在一些位置随机插入一部分item
def Insert(seq,insert_ratio,n,sim_dict):
    """
    :param seq: 原始序列
    :param insert_ratio: 插入比例
    :param n: 原始序列长度
    :param sim_dict: 一个字典，存储与每个item最相似的一些item
    :return: insert之后的序列
    """
    seq = copy.deepcopy(seq)
    insert_len=int(insert_ratio*n)
    if insert_len<1:
        return seq
    insert_idx=random.sample(range(n),insert_len)

    # 需要插入的item_id
    insert_pre=[]
    for idx in insert_idx:
        insert_pre.append(sim_dict[seq[idx]])
    seq=np.insert(seq,insert_idx,insert_pre)
    return seq



if __name__ == '__main__':
    # 序列长度
    n=10
    # 随机生成一个序列
    seq=np.random.randint(1,100,n)
    print("Ori: ",seq)

    mask_seq=Mask(seq,0.3,n)
    print("After mask: ",mask_seq)

    crop_seq=Crop_single(seq,0.3,n)
    print("After crop single: ",crop_seq)

    crop_seq=Crop_continue(seq,0.3,n)
    print("After crop continue: ",crop_seq)

    reorder_seq=Reorder(seq,0.5,n)
    print("After reorder: ",reorder_seq)

    # 这里为了方便，使用随机的方法构造相似性字典，一般采样itemCF、UserCF来根据item的共现信息计算两个item之间的相似性
    ori=list(range(1,101))
    sim=list(range(1,101))
    random.shuffle(sim)
    sim_dict=dict(zip(ori,sim))

    sub_seq=Substitute(seq,0.3,n,sim_dict)
    print("After substitute: ",sub_seq)

    insert_seq=Insert(seq,0.5,n,sim_dict)
    print("After insert: ",insert_seq)



