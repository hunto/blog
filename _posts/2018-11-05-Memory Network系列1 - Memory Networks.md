---
layout: post
cover: ''
title: 'Memory Network系列1 - Memory Networks'
subtitle: ''
date: 2018-11-05
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References
1. Weston J, Chopra S and Bordes A. [Memory Networks](https://arxiv.org/abs/1410.3916) , 15 Oct 2014

---

# Memory Networks

Memory Network由FaceBook于2014年提出，是一种相对较新的模型框架，旨在通过为序列中的每个标记提供显式内存表示来缓解序列数据中的长期依赖性问题。
一般的RNN模型虽然能通过hidden states与Attention机制来记忆一些依赖信息，但当依赖关系很长时，难以通过稠密的向量表示出长依赖关系，很容易遗忘较远的记忆，因此本文提出了新的记忆网络框架用于更好地将序列中的信息存储下来。
而实际上，本篇论文所提出的模型在大多数情况下很难实现，真正让Memory Networks成为深度学习中的一个分支的原因在于之后人们对它的发展完善。

## Memory Network 架构

一个Memory Netwok包含一个记忆空间（通常是由向量组成的数组）和四个基本组件。
4个组件 $$I,G,O,R$$ 分别为：

|component|name|description|
|:--:|:--:|:--:|
|I|input feature map|将输入向量转换为内部特征表示|
|G|generalization|使用 $$I$$ 得到的输入更新已有记忆|
|O|output feature map|产生一个新的输出|
|R|response|将输出转换为需要的表示形式|

**Example**
对于一个输入 $$x$$ ($$x$$ 可以是词向量、句子向量、图像特征等等)，其在Memory Network中的传输如下：
1. 将 $$x$$ 转换为内部特征表示： $$I(x)$$
2. 根据输入更新记忆： $$m_i = G(m_i, I(x), m), \forall i$$
   在这里， $$G$$ 可以是一个简单的将输入插入到空闲内存中的操作，也可以更新现有记忆。
3. 根据输入及记忆生成输出： $$o = O(I(x), m)$$
4. 将输出向量decode生成最后输出： $$r = R(o)$$

这样的一个步骤同时用于train和test，不同的地方在于，在test过程中，我们不需要更新 $$I, G, O, R$$ 的权重。

---

## Components 详述
### $G$ component
最简单的实现方式是将输入存储到内存的一个“插槽”（Array的某一行）中：

$$m_{H(x)}=I(x)$$

其中，$$H(x)$$ 用于选择内存中的哪一位置用于存放新记忆。

更复杂的一种实现方式是，$$G$$可以根据当前输入来更新先前的记忆内容。同时如果输入是 char-level 或  word-level 的向量时，可以将一整句中的向量组合到一起成为一个chunk更新到内存的同一位置中。

对于记忆很大的情况，我们需要更好地管理记忆。在本文中，作者提到可以使用多个memory分块的方式进行内存管理，按不同主题或不同文章划分为多个记忆，这样在更新和检索过程中，只需要操作其中的一块内存即可。

同时，本文还提到了一个作者没有实现的构想。当记忆满了的时候，可以引入"forgetting"机制来将内存中的部分内容删去，比如内存中最少使用的内容。

### $$O$$ and $$R$$ components
$$O$$ 主要实现相关信息的检索，对于输入 $$I(x)$$，$$O$$ 可以找出与其最相关的top-k个记忆，将其输入 $$R$$ 中得到最终的输出，$$R$$ 可以是一个 RNN，这样与单独使用 RNN 的不同之处在于，记忆网络通过 $$O$$ 对一长串记忆的检索聚焦，将输入 RNN 的序列长度缩短，且输入的序列为与输入 $$x$$ 最相关的内容。这样就实现了难以遗忘的长期记忆机制。

---

## A MemNN Implementation for Text
本文还介绍了Memory Network在TextQA中的应用，需要完成的任务是根据输入的多个句子，回答基于句子内容的提问。

### $$I$$
本部分直接将输入的一句话转换为向量表示。
### $$G$$ 
定义了函数 $$S(x)$$ 用于返回要写入的记忆的位置，在本例中，使用 $$N$$ 存储当前写入的位置index，每次写入后，$$N$$++；因此，$$G$$ 的操作为: $$m_N = x, N = N + 1$$

**这里的重点在于 $$O$$ 和 $$R$$：**

### $$O$$
这部分的目的在于根据问题输入找到其top-k个相关的句子，本文中作者用的是k=2，我们先来看k=1时的index公式：

$$o_1 = O_1(x, m) = {argmax}_{i=1, ..., N}\ S_O(x, m_i)$$

而当k=2时，我们的$$o_1$$与k=1时一样，$$o_2$$则是根据 $$x, m_{o_1}$$ 得到的输出：

$$o_2 = O_2(x, m) = {argmax}_{i=1,...,N}\ S_O([x, m_{o_1}],m_i)$$

这样就可以把句子间的递推关系找到，我们得到的最终输出 $$o = [x, m_{o_1}, m_{o_2}]$$，这也是 $$R$$ 部分的输入。

### $$R$$

对于训练数据是输出一个句子的情况，这部分可以使用 RNN 实现。
而对于需要输出单词的情况，本文将 $$o$$ 在词汇表上做softmax取出最有可能出现的单词。本部分不做详细介绍。

$$r = {argmax}_{\omega \in W}\ s_R([x, m_{o_1}, m_{o_2}], \omega)$$
$$s(x,y)=\Phi_x(x)^TU^T\Phi_y(y)$$

### Loss Function
本模型的Loss Function如下：

![0_1541402046609_loss.png](https://bbs.dian.org.cn/assets/uploads/files/1541402047077-loss.png) 

实际上，本模型分为两个部分，$$O$$ 和 $$R$$，Loss2产生的梯度是不能够backpropagation到 $$O$$ 处的。

![0_1541402070035_bp.png](https://bbs.dian.org.cn/assets/uploads/files/1541402071205-bp.png) 


### Performance

![0_1541402118929_performance.png](https://bbs.dian.org.cn/assets/uploads/files/1541402119601-performance.png) 

### Sample

![0_1541402134848_sample.png](https://bbs.dian.org.cn/assets/uploads/files/1541402135850-sample.png) 

---

## 本文Memory Network的缺点
* 需要很强的标注信息。由于 $$R$$ 的梯度不能回传给 $$O$$，网络不是端到端的，需要额外的标注用于训练 $$R$$，而并不是所有的数据集都像BabI一样是有关联标注的
* 模型不能做成端到端的。这样造成的结果是，优化器对Loss2的优化无法传到网络前半部分，不能优化前面的参数。


