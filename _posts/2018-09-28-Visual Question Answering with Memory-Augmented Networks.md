---
layout: post
cover: 'https://bbs.dian.org.cn/assets/uploads/files/1538122133972-3b5518ed-8308-4f92-b6fc-3c076a887f14-image.png'
title: 'Visual Question Answering with Memory-Augmented Networks - 笔记'
subtitle: ''
date: 2018-09-28
categories: VQA
tags: VQA 机器学习 深度学习 CV
---

在一般的VQA问题中，我们使用梯度下降来更新模型，使用低频截断来减少答案分类数，这样会造成模型对低频答案得到的分数较低，难以得到正确答案。这篇文章介绍了一种新的Memory-Augmented方法来解决这一问题。

## 模型结构
 ![0_1538122132510_3b5518ed-8308-4f92-b6fc-3c076a887f14-image.png](https://bbs.dian.org.cn/assets/uploads/files/1538122133972-3b5518ed-8308-4f92-b6fc-3c076a887f14-image.png) 

模型主要有三大块
### 1. 问题及图像特征提取
这里使用了双向LSTM提取问题特征，而对于图像特征，这里是将图片输入CNN网络中，从它的一层池化层中输出，（如VGG-16从pool5层输出，得到矩阵为14 X 14 X 512），再reshape得到（196 X 512）的矩阵，这里的196就是CNN feature中的分区(region)数。
$$\lbrace v_1, ..., v_N \rbrace, N=196$$

### 2. Sequential Co-Attention 协同注意力

![0_1538123095867_ec71b638-cec8-4540-ae83-553836c01644-image.png](https://bbs.dian.org.cn/assets/uploads/files/1538123096862-ec71b638-cec8-4540-ae83-553836c01644-image.png) 

这里主要是为了做attention提取出question与image的相关的特征。

First, compute a base vector $m_0$:

$$m_0 = v_0 \odot q_0$$
$$where\ \  v_0 = tanh(\frac1N \sum_n V_n)$$
$$q_0 = \frac1T \sum_t q_t$$

Visual attention:

$$h_n = tanh(W_vv_n)\odot tanh(W_mm_0)$$  
$$\alpha_n =  softmax(W_hh_n)$$  
$$v^*=tanh(\sum_{n=1}^N \alpha_nv_n)$$  

where $W_v$, $W_m$, $W_h$ denote hidden states.

Question attention:

$$h_t = tanh(W_qq_t)\odot tanh(W_mm_0)$$  
$$\alpha_t=softmax(W_hh_t)$$  
$$q^*=\sum_{t=1}^T \alpha_t q_t$$  

### 3. Memory Augmented Network
这里用到了一个LSTM作为memory controller

$$h_t = LSTM(x_t, h_{t-1})$$  

将h_t与memory中的所有记忆单元计算余弦相似度，再过softmax得到概率，再与$M_t$相乘得到记忆向量$r_t$，将$h_t$与$r_t$ concatenate到一起输入到分类网络中。

$$D(h_t, M_t(i))=\frac{h_t\cdot M_t(i)}{||h_t||\ ||M_t(i)||}$$  
$$w_t^r(i) = softmax(D(h_t, M_t(i)))$$  
$$r_t = \sum_iw_t^r(i)M_i$$  

**Memory 的更新**

$$f(x) = 1\ \ if\ \ x\ \ is\ \ True\ \ else\ \ 0$$  
$$w_t^w=\sigma(\alpha)w_{t-1}^r+(1-\sigma (\alpha) f(w_{t-1}^u \leq m(w_{t-1}^u, n))) $$  
$$M_t^i = M_{t-1}(i)+w_t^w(i)h_t$$
