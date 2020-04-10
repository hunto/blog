---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/blog/master/assets/img/MemoryNetworks/1542161582167-db2dfd8c-60df-4f6e-8416-626dc17cfb7e-image.png'
title: 'Memory Network系列5 - Key-Value Memory Networks'
subtitle: 'Key-Value Memory Networks for Directly Reading Documents'
date: 2018-11-14
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References
1. [Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126) , 9 Jun 2016  
2. [Memory Network系列2 - End-To-End Memory Networks](https://blog/memorynetworks/2018/11/05/Memory-Network%E7%B3%BB%E5%88%972-End-To-End-Memory-Networks.html)

---

# Key-Value Memory Networks
本篇论文主要解决Memory Networks在知识库中因知识库知识稀疏、形式有限产生的问题。以往的研究中，使用构建知识库的方法构建每一个领域对应的闭源知识库，这样造成的后果是知识库的形式有限。因此本文想要通过memory network直接阅读大规模的文献来提升QA的效果。

---

## 实现

Key-Value Memory Network基于End-To-End Memory Network做了修改，主要改变了memory的寻址与输出。

![0_1542161581331_db2dfd8c-60df-4f6e-8416-626dc17cfb7e-image.png](https://raw.githubusercontent.com/hunto/blog/master/assets/img/MemoryNetworks/1542161582167-db2dfd8c-60df-4f6e-8416-626dc17cfb7e-image.png) 

## Memory
由于大数据库中的数据量很大，memory首先通过key hashing操作选出与问题相关的 $$N$$ 条数据，再使用与End-To-End Memory类似的操作在选出的数据中进行操作。

### Key Hashing
本部分可以选出 $$N$$ 个与问题有关的 key-value pair： $$(k_{h_1}, v_{h_1}), (k_{h_2},  v_{h_2}), ..., (k_{h_N}, v_{h_N})$$ 

Key hashing是本篇文章中很重要的部分，使用key对数据进行定位及分类，使用value来得到相关数据的值，相比于MeMN2N使用相同的数据进行定位和取值，这样key-value的方法的好处在于能够自定义key和value的表示以更精准地定位数据及表示数据，例如：你可以用文章的分类作为key，文章内的关键词作为value。本文提出了很多种hash方法，感兴趣可以自己去看一看。

### Key Addressing
这部分计算了问题与key的相关性

$$p_{h_i} = Softmax(q^T \cdot A\Phi_K(k_{h_i}))$$

Where $$\Phi$$ are feature maps of dimension $$D$$, $$A$$ is a $$d\times D$$ matrix.

这里与下文的 $$\Phi$$ 均为对word作embedding操作。

这里的 $$q$$ 与MeMN2N相同，为前一层的输出与输入的和

$$q_j = R_j(q_{j-1}+o_j)$$

where $$R$$ is a $$d \times d$$ matrix.

对于hop1， $$q$$ 为对问题的embedding：

$$q_1 = A\Phi_X(x)$$

### Value Reading

这部分将addressing步骤得到的各数据的概率与其值进行加权求和得到输出。

$$o = \sum_i p_{h_i}A\Phi_V(v_{h_i})$$

### Answer Generation

这部分利用了label的文本信息与最后输出 $$q_{H+1}$$ 进行点乘来得到answer，不同问题的处理方式不一样，这里不作展开。

$$\hat a = argmax_{i=1, ..., C}Softmax(q_{H+1}^TB\Phi_Y(y_i))$$

