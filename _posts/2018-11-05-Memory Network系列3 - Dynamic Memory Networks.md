---
layout: post
cover: 'https://bbs.dian.org.cn/assets/uploads/files/1541421588545-99313151-4dd6-45e6-9a19-6b78ba61ec4c-image.png'
title: 'Memory Network系列3 - Dynamic Memory Networks'
subtitle: ''
date: 2018-11-05
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References

1. [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285) , 24 Jun 2015

---

# Dynamic Memory Networks

之前我们说到了MemNN, MemN2N，现在我们来学习一个新的记忆网络模型——DMN。DMN由论文[Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285) 提出，本文认为，几乎所有的NLP问题都可以认为是一个QA问题。正如标题所示，DMN可以用在很多种类的NLP的任务中，如QA、文本分类、情感分析、序列模型。

---

## 实现

![0_1541421581950_99313151-4dd6-45e6-9a19-6b78ba61ec4c-image.png](https://bbs.dian.org.cn/assets/uploads/files/1541421588545-99313151-4dd6-45e6-9a19-6b78ba61ec4c-image.png)

与MemNN类似，DMN也由4个部分组成：

### Input Module
输入模块将文本转换为向量表示作为memory的输入。本文使用 GRU  处理文本输入，将GRU的hidden state作为输出。需要注意的是，对于不同的输入有不同的输出方法。若输入的是一句话，输出则为句子中每个词的hidden state，最终会得到句子长度个向量；若输入的是多个句子，则将多个句子cat到一起，句子间用一个标志<EOF>间隔，取每次输入<EOF>后的hidden state作为输出，最终会得到句子个数个向量。模块输出为$$c$$。

### Question Module
本部分输入为一个问题，与input module类似，question model也将句子经过一个GRU，但这里我们只需要RNN的final state作为输出。模块输出为 $$q$$ 。最后一层记忆的输出 $$m^k$$ 为最后的模块输出。



### Episodic Memory Module 

本部分由多层memory组成，每一部分都将 $$c$$ 、前一层的输出 $$m^{k-1}$$、$$q$$ 作为输入输入到`Attention Mechanism`中，将Attention得到的 $$g^k$$ 作为`Memory Update Mechanism`的输入，经过`Memory Update Mechanism`后得到该层记忆输出 $$e^k$$。

各层输出通过一个RNN联系到一起，每一层的输出 $$e^k$$ 均输入GRU中，得到的hidden state即为该层网络的输出 $$m^k$$，即： $$m^k = GRU(e^k, m^{k-1})$$。

最后一层网络的输出 $$m^k$$ 即为 Memory Module的输出。

**Memory Update Mechanism**

每一层memory内部记忆的更新都使用了一个修改后的GRU:

$$h_t^k = g_t^k GRU(c_t, h_{t-1}^k) + (1 - g^k_t)h_{t-1}$$

本部分将 $$c_t$$ 与其对应的 $$g_t$$ 输入上述修改后GRU中，取 final state作为该层记忆输出 $$e^k$$。

**Attention Mechanism**  

本文使用了一个门函数作为Attention Mechanism。序列中的每一个节点的gate值都由当前节点值 $$c_t$$、前一层的输出 $$m^{k-1}$$、$$q$$ 决定。

$$g_t^k = G(c_t, m^{k-1}, q)$$

当k=0即当前为记忆网络的第一层时，输入 $$m^{k-1}$$ 为 $$q$$。

门控函数为一个2层神经网络：

$$G(c, m, q) = \sigma (W^{(2)}tanh(W^{(1)}z(c,m,q)+b^{(1)})+b^{(2)})$$

其中 $$z(c, m, q)$$ 为一个长向量用于表示 $$c, m, q$$ 之间的相似度：

$$z(c,m,q) = [c, m, q, c \circ q, c \circ m, |c-q|, |c-m|, c^TW^{(b)}q,c^TW^{(b)}m]$$

where $$\circ$$ is the element-wise product.

### Answer Module
这部分通过memory module的输出与 $$q$$ 生成最后的answer。对于seq2seq问题，这里可以使用一个GRU生成答案，对于一般分类问题，这里将 $$[m^k, q]$$ 输入一层fc即可。