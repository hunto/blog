---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/MemoryNetworks/1542186893441-3a3d7404-2321-40cb-8913-a8d3237f000e-image.png'
title: 'Memory Network系列6 - Gated End-To-End Memory Networks'
subtitle: 'Gated End-To-End Memory Networks'
date: 2018-11-14
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References
[1] [Gated End-to-End Memory Networks](https://arxiv.org/abs/1610.04211) , 13 Oct 2016   
[2] [Highway Networks](https://arxiv.org/abs/1505.00387)   

---

# Gated End-To-End Memory Networks

到了这里，memory network有一个问题没有解决。我们使用多个hop来解决记忆中的多层依赖问题，但是hop是固定的，在更为复杂的问题例如dialog中，各hop间的依赖关系会更为复杂，如果有几层依赖是我们不需要的怎么办呢，有没有类似于LSTM的方法能够创建hop间的shortcut？我们希望有一个结构能够选择遗忘之前的记忆、忽略当前的记忆、决定当前记忆与之前记忆的比重，这是不是与GRU和LSTM很像？于是这篇论文提出了Gated Memory Networks来实现Memory Networks中的遗忘。

---

## 实现
Gated End-To-End Memory结构与MeMN2N类似，主要是改变了各hop间的连接方式

![0_1542186888752_3a3d7404-2321-40cb-8913-a8d3237f000e-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/MemoryNetworks/1542186893441-3a3d7404-2321-40cb-8913-a8d3237f000e-image.png) 

这篇论文中使用了 [Highway Networks](https://arxiv.org/abs/1505.00387) 中提出的shortcut结构，最终的公式如下：

$$T^k (u^k) = \sigma (W^k_Tu^k + b_T^k)$$

$$u^{k+1} = o^k \circ T^k(u^k)+u^k\circ (1 -  T^k(u^k))$$

---

## Performance

![0_1542194084795_91a205c2-397a-4302-8112-42f923a411e1-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/MemoryNetworks/1542194086824-91a205c2-397a-4302-8112-42f923a411e1-image.png) 

