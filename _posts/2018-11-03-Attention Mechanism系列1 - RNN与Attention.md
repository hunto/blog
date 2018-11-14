---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005449475-2ff99bf7-876e-41bd-a327-40df437d4248-image.png'
title: 'Attention Mechanism系列1 - RNN与Attention'
subtitle: 'RNN, GRU, LSTM, Seq2Seq, Attention'
date: 2018-11-03
categories: Attention
tags: Attention 机器学习 深度学习
---

# References
1. [完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)
2. [Sequence Modeling using Gated Recurrent Neural Networks](https://arxiv.org/pdf/1501.00299.pdf)
3. [CS224n笔记-lecture9](https://hunto.github.io/cs224n/2018/07/18/CS224n%E7%AC%94%E8%AE%B0-lecture9.html)
4. [CS224n笔记-lecture10](https://hunto.github.io/cs224n/2018/07/19/CS224n%E7%AC%94%E8%AE%B0-lecture10.html)
5. [Attention机制详解（一）——Seq2Seq中的Attention](https://zhuanlan.zhihu.com/p/47063917)

---

在介绍Attention之前，我们先回顾一下RNN及Improved RNN。

## RNN

RNN的hidden state计算公式如下：

$$h_n = f(W^{(hh)}h_{n-1}+W^{(hx)}x_n)$$

其中，$$f$$ 为激活函数。

## GRU

![0_1542005997479_1e63c08c-1c2f-46d1-ba30-a5933294b8b3-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005998955-1e63c08c-1c2f-46d1-ba30-a5933294b8b3-image.png) 

如图，GRU在RNN的基础上引入了`reset gate`和`update gate`两个门，其计算公式如下：

$$r_t = \sigma (W^{(xr)}x_t + W^{(hr)}h_{t-1})$$

$$z_t = \sigma (W^{(xz)}x_t + W^{(hz)}h_{t-1})$$

当前输入状态公式如下：

$$\tilde h = tanh(W^{(xh)}x_t+W^{(hh)}(r_t \circ h_{t-1}))$$

最终的隐藏层状态如下：

$$h_t = z_t \circ h_{t-1} + (1-z_t) \circ \tilde h_t$$

* `reset gate`用于控制前一记忆对当前输入的影响程度，若 $$r_t=0$$，则之前层的记忆被遗忘。
* `update gate`用于控制当前输入对隐藏层状态的影响程度，若 $$z_t=1$$，则当前状态与输入无关，只会复制前一状态。

## LSTM

![0_1542007184774_72474b64-1eb9-4307-91ba-8e2ce8c06d25-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542007185657-72474b64-1eb9-4307-91ba-8e2ce8c06d25-image.png) 

![0_1542007215403_0a7d78c7-574c-4a5f-926d-e6c91021dd4f-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542007216236-0a7d78c7-574c-4a5f-926d-e6c91021dd4f-image.png) 

---

## RNN结构

### 1. 经典的RNN结构 (N to N)

![0_1542004416508_d6554604-a848-4574-885d-71b8b2df4eac-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542004418016-d6554604-a848-4574-885d-71b8b2df4eac-image.png) 

经典的RNN结构输入与输出均为相同 $$N$$ 长度的序列，其将每一节点均作为输出。输入与输出序列长度相同限制了其的应用，可用于视频帧分类、下一字符概率预测等任务。

### 2. N to 1

![0_1542004812087_6449cbf0-90b4-4e66-bfdb-45cf8eaa7513-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542004815240-6449cbf0-90b4-4e66-bfdb-45cf8eaa7513-image.png) 

`N to 1`结构通过长度为 $$N$$ 的输入序列得到长度为1的输出，其将最后节点的hidden state作为输出用于下层网络，主要用于分类问题，如：文本分类，视频分类。

### 3. 1 to N

如果是想通过1得到长度为 $$N$$ 的序列输出呢？

这里有两种方法：

* 只在序列开始时将向量输入进行计算
![0_1542005012036_ead3636d-2864-488e-bf85-5f2f1c7fd8cc-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005013529-ead3636d-2864-488e-bf85-5f2f1c7fd8cc-image.png) 

* 在每一个节点均作为输入进行计算
![0_1542005126158_78b8cd43-76b9-41f5-9721-b96a83dfdc30-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005127636-78b8cd43-76b9-41f5-9721-b96a83dfdc30-image.png) 

这样的应用有：看图说话、通过类别生成音乐等。


### 4. N to M

这种结构是RNN最重要的一种变种，可以通过变长的输入得到变长的输出，这种模型称为Encoder-Decoder模型，也可以称为Sequence-to-Sequence模型。

Seq2Seq结构是由两个RNN构成的，一部分为`N to 1`，一部分为`1 to M`，这样就可以将 模型变为`N to M`，由于`1 to N`有两种方式，因此Seq2Seq也有两种结构：

* Decoder只在序列开始时输入c
![0_1542005448051_2ff99bf7-876e-41bd-a327-40df437d4248-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005449475-2ff99bf7-876e-41bd-a327-40df437d4248-image.png) 

* Decoder在序列的每一步均输入c
![0_1542005462044_6a423770-a11d-403f-b1a6-2eeed2f3f981-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542005463237-6a423770-a11d-403f-b1a6-2eeed2f3f981-image.png) 

Seq2Seq的模型应用范围十分广泛，如：机器翻译、语音识别、文本摘要等。

---

## Attention Mechanism

关于Attention的更详细介绍可见：[CS224n笔记-lecture10](https://hunto.github.io/cs224n/2018/07/19/CS224n%E7%AC%94%E8%AE%B0-lecture10.html)

在Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征c再解码，因此， c中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个c可能存不下那么多信息，就会造成翻译精度的下降。

于是我们想通过将所有encoder的hidden state输出，在不同的decoder时间节点根据当前decoder状态选择不同的c作为输入，来解决这一瓶颈。

**Seq2Seq Attention**

下面我们来详细介绍Seq2Seq中attention的计算

![0_1542008763353_62964331-a51e-4f10-b16c-e412bb5075fc-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/Attention/1542008764039-62964331-a51e-4f10-b16c-e412bb5075fc-image.png) 

在这里，我们先通过encoder得到每一个时间节点的隐藏状态：

$$[h_1, h_2, ..., h_n]$$

假设decoder当前的隐藏状态是 $$s_{t-1}$$ ，那么我们可以得到其与每一个 $$h_j$$ 的关联性

$$e_{tj} = a(s_{t-1}, h_j)$$

$$e_t = [e_{t1}, e_{t2}, ..., e_{tn}]$$

这里，$$a$$ 为计算相关性的方法，常见的有点乘、加权点乘、加权和。

接着我们使用softmax对 $$e_t$$ 进行归一化：

$$a_t = Softmax(e_t)$$

再接着我们用 $$a_t$$ 对 $$h$$ 进行加权求和得到当前输入：

$$c_t = \sum_{j=1}^n a_{tj} \circ h_j$$

因此，我们的decoder隐藏状态为：

$$s_t = f(s_{t-1}, c_t)$$
