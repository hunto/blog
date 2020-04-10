---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/blog/master/assets/img/MemoryNetworks/1541406370117-493f70b3-1bb3-4e2c-bc90-388207ed854f-image.png'
title: 'Memory Network系列2 - End-To-End Memory Networks'
subtitle: ''
date: 2018-11-05
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References
1. [End-To-End Memory Networks](https://arxiv.org/abs/1503.08895) , 31 Mar 2015

---

# End-To-End Memory Networks

前一节中我们说到，Memory Networks虽然能够很好地处理长期依赖的问题，但由于其网络结构不是端到端的，导致训练时需要更多的标记，难以用于更多的任务中。于是Facebook紧接着在2015年提出了 End-To-End Memory Networks 解决了这一问题。

---

## 实现
与前一篇文章相同，模型的输入为离散的句子特征集合 $$x_1, ..., x_n$$，问题特征 $$q$$，输出为答案 $$a$$。
模型首先将所有的 $$x$$ 存储到一个固定大小的记忆中，再找到对 $$x$$ 和 $$q$$ 的一个连续表示，这一连续表示通过多个hop得到输出 $$a$$ 。这样我们可以通过反向传播将损失经过多个memory传回输入。

## Single Layer
我们先从单层记忆网络的端到端实现看起。

### Input Memory Representation
对于input set $$x_1, ..., x_2$$，我们首先将 $$x_i$$ 经过embedding层得到维度为 $$d$$ 的memory vector $$m_i$$。最简单的embedding实现方式是用一个大小为 $$d \times V$$ 的矩阵 $$A$$ 表示embedding。 $$q$$ 也通过同样的方式经过embedding得到其internal state $$u$$，接着，我们将 $$m_i$$ 与$$u^T$$ 做内积再经过softmax得到输入 $$x$$ 对于 $$q$$ 的概率表示 $$p$$：

$$p_i = Softmax(u^Tm_i)$$

### Output Memory Representation

首先，我们让每一个 $$x_i$$ 都有一个与其对应的表示 $$c_i$$ （最简单的方式就是再用一个embedding），最终output的输出就是input得到的概率矩阵 $$p$$ 与 $$c$$ 的乘积（element-wise）的和：

$$o = \sum_i p_ic_i$$

这样，我们就可以将o处的梯度往回传到input了。

### Generating the Final Prediction
这部分很简单，将记忆输出 $$o$$ 与问题表示 $$u$$ 相加再与一个权值矩阵 $$W$$ 相乘，最后过softmax即可：

$$\hat a = Softmax(W(o+u))$$

---

## Multiple Layers

这部分要实现的目的与上一篇文章的top-k相同，具体实现也与上一篇文章类似，将前一层的输出与 $$q$$ 一同作为下一层的输入。

* 第一层以后的输入 $$u^k$$ 为前一层的输出与 $$u$$ 的和：

$$u^{k+1} = o^k + u$$

* 每一层都有自己的embedding矩阵 $$A^k, C^k$$ 得到 $$m, c$$
* 最终的输出为：

$$\hat a = Softmax(W(o^k + u))$$

---

![0_1541406368567_493f70b3-1bb3-4e2c-bc90-388207ed854f-image.png](https://raw.githubusercontent.com/hunto/blog/master/assets/img/MemoryNetworks/1541406370117-493f70b3-1bb3-4e2c-bc90-388207ed854f-image.png)