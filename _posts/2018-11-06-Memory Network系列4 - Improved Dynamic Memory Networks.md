---
layout: post
cover: 'https://bbs.dian.org.cn/assets/uploads/files/1541474821662-0e4c6bcd-15ac-42e3-b276-4bd7f338a9e3-image.png'
title: 'Memory Network系列4 - Improved Dynamic Memory Networks'
subtitle: ''
date: 2018-11-06
categories: MemoryNetworks
tags: MemoryNetworks 机器学习 深度学习
---

# References
1. [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) , 4 Mar 2016
2. [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057) , 6 Jun 2015

---

# Improved Dynamic Memory Networks
DMN+由[1]提出，为DMN的改进模型，相比于DMN，DMN+主要做了以下改动：

---

## Input Module for Text QA
在之前的DMN中，模型使用了一个单向GRU提取句子特征，将word输入GRU中，在输入句尾标记后输出hidden state作为sentence的特征。DMN虽然在有supporting facts标记的bAbI-1k中表现良好，但在没有supporting facts标记的bAbI-10k中表现得不好。本文推断造成这种差异有两个主要原因：
1. GRU只能捕捉到当前句子之前的依赖关系而捕捉不到之后的依赖。
2. 将word-level的特征输入GRU中会导致句子间的间隔过远难以产生记忆交互。



DNM+使用了一个由两个部分组成的网络替换掉了原来的GRU，第一部分叫做sentence reader，将由word vector组成的一句话encode为sentence vector；第二部分叫做`Input Fusion Layer`，使用了一个双向GRU提取句子及句子间的依赖信息。

![0_1541474820362_0e4c6bcd-15ac-42e3-b276-4bd7f338a9e3-image.png](https://bbs.dian.org.cn/assets/uploads/files/1541474821662-0e4c6bcd-15ac-42e3-b276-4bd7f338a9e3-image.png) 

**Sentence Reader**

Sentence reader可以使用各种encode方式，如GRU、LSTM，本文使用了[2]中提到的positional encoder。

Positional encoder将word vector根据其位置编码后相加得到最终的sentence vector $$f_i$$ ：

$$f_i = \sum^{j=1}_M l_j \circ w_j^i$$

其中， $$\circ$$ 是element-wise product，$$l_j$$ 为 $$D$$ 个元素构成的向量，其表示如下：

$$l_{jd} = (1 - j / M) - (d / D)(1 - 2j / M)$$

$$l_j = [l_{j1}, ..., l_{j_D}]$$

其中，d为embedding index，D为embedding size。

**Input Fusion Layer**

这部分使用了双向GRU，将前一层得到的sentence vector作为输入：

$$\overrightarrow{f} = GRU_{fwd}(f_i, \overrightarrow{f_{i-1}})$$

$$\overleftarrow{f} = GRU_{bwd}(f_i, \overleftarrow{f_{i+1}})$$

$$\overleftrightarrow{f} = \overrightarrow{f} + \overleftarrow{f}$$

---

## Input Module For VQA
对于visual question answering，DMN+也提出了一种input module，其包括3个部分： local region feature extraction, visual feature embedding, input fusion layer 。

![0_1541476893852_09dc68a4-0c07-451f-b1d2-5ad9b022d4d4-image.png](https://bbs.dian.org.cn/assets/uploads/files/1541476894953-09dc68a4-0c07-451f-b1d2-5ad9b022d4d4-image.png) 

**Local Region Feature Extraction**

本部分的目的是提取image特征，可以使用预训练的图像模型作为特征提取器，取最后pool层的前一层的feature作为输出。
例如： $$448 \times 448$$的图像输入，会得到 $$14 \times 14 \times 512$$的输出，也就是 $$196$$ 个region信息。

**Visual Feature Embedding**

本部分为一个简单的fc，将前面的各region embedding。

**Input Fusion Layer**

这部分也像Text一样使用了一个双向GRU。

---

## Episodic Memory Module
这部分相较于DMN，对Attention Mechanism和Memory Update Mechanism均做了改动。



**Memory Update Mechanism**

对于内部记忆的更新，本文提到了两种方式，一种是与DMN相同的Attention Based GRU，另一种是Soft attention。

* Attention Based GRU

$$h_i^k = g_i^k GRU(\overleftrightarrow{f_i}, h_{i-1}^k) + (1 - g^k_i)h_{i-1}$$

$$c^k = h^k_{N}$$

* Soft Attention

$$c^k = \sum_{i=1}^N g_i^k \overleftrightarrow{f_i} $$


在DMN中，各层使用GRU联系：

$$m^k = GRU(c^k, m^{k-1})$$

而在DMN+中，Following the memory update component used in Sukhbaatar et al. (2015) and Peng et al. (2015) we experiment with using a ReLU layer for the memory update, calculating the new episode memory state by

$$m^k = ReLU(W^k[m^{k-1}, c^k, q] + b)$$

**Attention Mechanism**

$$z_i^k = [\overleftrightarrow{f_i} \circ q, \overleftrightarrow{f_i} \circ m^{k-1}, |\overleftrightarrow{f_i} - q|, |\overleftrightarrow{f_i} - m^{k-1}| ]$$

$$Z_i^k = W^{(2)}tanh(W^{(1)}z_i^k+b^{(1)})+b^{(2)}$$

$$g_i^k=Softmax(Z_i^k)$$

其中，$$||$$是 element-wise absolute。

---

## Performance

### Text QA
![0_1541483317489_78916598-329b-44b8-88ff-6f5f40a6a42b-image.png](https://bbs.dian.org.cn/assets/uploads/files/1541483318352-78916598-329b-44b8-88ff-6f5f40a6a42b-image.png) 

### VQA

![0_1541483337556_2e84cca5-8c58-4884-9913-20bdad102202-image.png](https://bbs.dian.org.cn/assets/uploads/files/1541483338050-2e84cca5-8c58-4884-9913-20bdad102202-image.png) 