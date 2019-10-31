---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544351696071-f3a510da-59b7-4a91-b3a5-bcfd9f6a8164-image.png'
title: 'Pattern Recognition and Machine Learning笔记 - section 1'
subtitle: 'Introduction'
date: 2019-10-31
categories: PRML
tags: PRML 机器学习 深度学习
---

# 1. Introduction

## 1.1 Example
### Square Error

$$E(W)=\frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2$$

### Root-Mean-Square Error

$$E_{RMS} = \sqrt {2E(W)/N}$$

这里，开方是为了使 error 与 target 有相同的scale，除以N是为了比较在不同大小数据集中的损失。实质上它与平方误差没什么差别。

### 过拟合

![0_1544350839796_2ab6b301-02b7-478c-84ef-f6b38e3e8646-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544350841331-2ab6b301-02b7-478c-84ef-f6b38e3e8646-image.png) 

如上图，不同阶数多项式拟合出的图像。在数据较少的情况下，方程越复杂，越容易过拟合。

* 增大数据集缓解过拟合

![0_1544351102918_2a02ac07-cace-4358-9519-4917f5ef85a4-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544351104185-2a02ac07-cace-4358-9519-4917f5ef85a4-image.png) 

* 损失函数加正则项缓解过拟合

![0_1544350951485_6260bda8-5698-46cf-adec-27c627f556f4-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544350952606-6260bda8-5698-46cf-adec-27c627f556f4-image.png) 

可以看出，过拟合情况下，高次项系数非常大。

因此，为了减缓过拟合，我们应该抑制方程系数的增大。

$$\tilde E(W) = \frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2 + \sum^M_i (w_i)^2$$

下表为不同$$\lambda$$ 拟合的方程的系数值：

![0_1544351317667_a02bb478-29d0-4d1b-a629-a93a58edf50c-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544351319010-a02bb478-29d0-4d1b-a629-a93a58edf50c-image.png) 

## 1.2 概率论 (Probability Theory)

![0_1544351694358_f3a510da-59b7-4a91-b3a5-bcfd9f6a8164-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544351696071-f3a510da-59b7-4a91-b3a5-bcfd9f6a8164-image.png) $$$$

### 1.2.3 Bayes' theorem

$$p(w|\mathcal{D}) = \frac{P(\mathcal{D}|w)P(w)}{P(D)}$$

$$p(\mathcal{D}) = \int p(\mathcal{D}\ |\ w)p(w)dw$$

* Likelihood Function:$$p(\mathcal{D}|w)$$

### 1.2.4 Gaussian Distribution

$$\mathcal{N}(x|\mu , \sigma^2) = \frac{1}{(2\pi \sigma^2)^{1/2}}exp(-\frac{1}{2\sigma^2}(x - \mu)^2)$$

* Maximum Likelihood solution
    *$$\mu_{ML}=\frac1N\sum^N_{n=1} x_n$$
    *$$\sigma^2_{ML} = \frac1N \sum^N_{n=1}(x_n - \mu_{ML})^2$$

**最大似然估计的方差偏移问题**

![0_1544412407530_49e0a869-d3b2-47ae-b67f-3a9ac42f2555-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544412408889-49e0a869-d3b2-47ae-b67f-3a9ac42f2555-image.png) 

* 计算最大似然解的期望：
    *$$E(\mu_{ML}) = \mu$$ 
    *$$E(\sigma^2_{ML}) = (\frac{N-1}{N})\sigma^2$$


* 由上公式，我们可以将$$\sigma^2_{ML}$$ 乘以$$\frac{N}{N-1}$$ 得到$$\sigma^2$$ 的无偏估计：

$$\tilde \sigma^2 = \frac{N}{N-1}\sigma^2_{ML} = \frac1{N-1} \sum^N_{n=1}(x_n - \mu_{ML})^2$$

通过上式可以看出，当样本量较小时，使用极大似然估计得到的方差偏移很大，均值无偏移，当样本量趋于无穷大时，极大似然方差的误差可以忽略。这样的偏移是前面多项式过拟合问题的核心，在机器学习中，带有更多参数的复杂模型更会加重这一现象，在本书的后面会详细阐释。

### 1.2.5 通过高斯分布重新考察曲线的过拟合问题

多项式曲线的拟合是通过误差最小化实现的，我们这里从概率的角度理解过拟合和正则化。

曲线拟合问题的目标是根据给定$$X = \{ x_1, x_2, ..., x_n \}$$ 及$$T = \{ t_1, t_2, ..., t_n\}$$ 得到对于新的$$x$$ 的值$$t$$。

假设，$$X, T$$ 满足高斯分布，分布的均值为$$y(x, W)$$ ，有：

$$y(x, W) = \sum^M_{j=0}w_jx^j\ \ \ \ (1.1)$$

$$p(t\ |\ x,W,\beta)= \mathcal{N}(t\ |\ y(x,W),\beta^{-1})\ \ \ \ \ \ \ (1.60)$$

为了与后续章节统一，这里定义了精度参数$$\beta$$ ，为方差的倒数。

![0_1544419019668_4a0f52c0-284d-4da2-85f6-d25e6c6dfa01-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544419020233-4a0f52c0-284d-4da2-85f6-d25e6c6dfa01-image.png) 

由公式1.60我们可以得到最大似然方程如下：

![0_1544419423041_4ab8f055-78f8-4869-aaa4-ee66291d069c-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544419423625-4ab8f055-78f8-4869-aaa4-ee66291d069c-image.png) 

* **使用最大似然解表示$$W$$**

我们首先确定$$W$$ 的最大似然解，首先，式1.62中后两项与$$W$$ 无关，可以忽略掉，同时，使用一个正常数系数来缩放对数似然函数并不会改变其关于$$W$$ 的最大值的位置，因此可以用$$\frac12$$ 来代替$$\frac1\beta$$。我们再对上面得到的式子取负对数，这样最大化似然函数就变为了最小化负对数似然函数。最后得到的负对数似然函数等价于之前的平方误差函数：

$$E(W)=\frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2\ \ \ \ (1.2)$$

因此，我们之前使用的平方误差函数其实是使用高斯噪声函数下，最大化似然函数的一个自然结果。

* **使用贝叶斯定理表示$$W$$**

在得到了$$W_{ML}$$ 之后，我们接着使用最大似然法确定$$\beta$$：

$$\frac1{\beta_{ML}} = \frac1N\sum_{n=1}^N[y(x_n, W_{ML})- t_n]^2$$

把上面的两个最大似然参数代入1.60中，我们可以得到t的概率分布的预测分布：

$$p(t, W_{ML}, \beta_{ML}) = \mathcal{N}(t\ |\ y(x,W_{ML}),\beta^{-1}_{ML})$$

根据贝叶斯定理，我们想要得到在$$x, t$$ 条件下的关于$$W$$ 的后验概率$$p(W\ |\ x, t)$$ ，还需要有$$W$$ 的先验分布，简单起见，我们用高斯分布来表示$$W$$ 的分布：

$$p(W\ |\ \alpha) = \mathcal{N}(W\ |\ 0, \alpha^{-1}I)=(\frac{\alpha}{2\pi})^{\frac{M+1}2}exp(-\frac\alpha2W^TW)$$

这里$$\alpha$$ 是超参数，用于控制分布的精度。

因此，由贝叶斯定理，我们有：

$$p(W\ |\ X,T,\alpha,\beta)\propto p(T\ |\ X, W, \beta) p(W\ |\ \alpha)$$

通过寻找最可能的$$W$$ 值（即最大化后验概率）来确定$$W$$ ，这种技术被称为最大后验 (maximum posterior)，简称MAP。我们取上面公式的负对数，结合之前的方法可得，最大化后验概率就是最小化下式：

$$\frac\beta2 \sum_{n=1}^N[y(x_n,W)-t_n]^2 + \frac\alpha2 W^TW$$

从这里我们可以看出，最大化后验概率等价于最小化正则化的平方和误差函数，正则化参数为$$\lambda = \frac\alpha\beta$$

$$\tilde E(W) = \frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2 + \frac\lambda2W^TW\ \ \ \ (1.4)$$

### 1.2.6 贝叶斯曲线拟合 (Bayesian Curve fitting)

虽然在上一节已经使用了先验分布来得到后验概率分布，但对于$$W$$ 来说，我们仍在进行它的分布估计，这并不是贝叶斯观点，在纯粹的贝叶斯方法中，我们应该自始至终地应用概率的乘规则和加规则。所以在之后，我们会对所有$$W$$ 值进行积分，对于模式识别来说，**这种积分是贝叶斯方法的核心**。

在曲线拟合问题中，我们知道$$X, T$$ ，目标是预测$$t$$ 的值，因此我们想要估计$$t$$ 的分布$$p(t\ |\ x, X, T)$$

  $$ p(t\ |\ x, X, T) = \int p(t\ |\ x, W)p(W | X, T)dW$$

这里与上一节的不同之处在于，$$W$$ 的分布是通过$$X, T$$ 确定的，而非上面为了简单起见定义的高斯分布。

![0_1544424416208_1a172aaa-0a8f-4074-8186-cdfd5c5831a9-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544424417329-1a172aaa-0a8f-4074-8186-cdfd5c5831a9-image.png) 

我们可以从公式1.69中看到，$$t$$ 的分布的均值和方差都依赖于$$X$$ ，公式1.71的第一项表示预测值 t 的不确定性，在最大似然求解方法中，这种不确定性用$$\beta^{-1}$$ 表达。然而，第二项也对$$W$$ 的不确定性有影响，这是贝叶斯方法得到的结果。

![0_1544424771011_372da5aa-4556-494f-9242-3311fe35b7ad-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544424777463-372da5aa-4556-494f-9242-3311fe35b7ad-image.png) 

---

## Exercises

### 1.1

$$y(x, W) = \sum^M_{j=0}w_jx^j\ \ \ \ (1.1)$$

$$E(W)=\frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2\ \ \ \ (1.2)$$

将(1.1)代入(1.2)得

$$E(W)=\frac12 \sum^N_{n=1}(\sum^M_{j=0}w_j x_n^j - t_n)^2$$

对$$w_i$$ 求偏导，有：

* $$\frac{\partial E(W)}{\partial w_i} = \sum^N_{n=1}(\sum^M_{j=0}w_jx_n^j - t_n) x_n^i = 0$$

* $$\sum^N_{n=1}(\sum^M_{j=0}w_j x_n^j) (x_n)^i= \sum^N_{n=1}t_n(x_n)^i$$

### 1.2

$$\tilde E(W) = \frac12 \sum^N_{n=1}(y(x_n,W)-t_n)^2 + \frac\lambda2 \sum^M_i (w_i)^2\ \ \ \ (1.4)$$

同样将 (1.1) 代入 (1.2) 得

$$\tilde E(W)=\frac12 \sum^N_{n=1}(\sum^M_{j=0}w_j x_n^j - t_n)^2 + \sum^M_i (w_i)^2$$

对$$w_i$$ 求偏导，有：

* $$\frac{\partial \tilde E(W)}{\partial w_i} = \sum^N_{n=1}(\sum^M_{j=0}w_jx_n^j - t_n) x_n^i + \lambda w_i = 0$$

* $$\sum^M_{j=0}(\sum^N_{n=1}x_n^{(j+i)} + \frac{\lambda w_i}{Mw_j})w_j= \sum^N_{n=1}t_n(x_n)^i$$

* $$\sum^M_{j=0}(A_{ij}+\frac{\lambda w_i}{Mw_j})w_j=T_i$$


### 1.3 

$$p(a)=p(r, a) + p(b, a) + p(g, a) = 0.34$$

$$p(g|o) = \frac{p(o|g)p(g)}{p(o)} = \frac{p(o|g)p(g)}{p(r,o) + p(b, o) + p(g, o)} = 0.5$$


### 1.5 方差公式推导

$$var[f(x)] = E[(f(x) - E[f(x)])^2] \\ = E[f(x)^2 - 2f(x)E[f(x)] + E[f(x)]^2] \\ = E[f(x)^2] - 2E[f(x)]^2 + E[f(x)]^2 \\ = E[f(x)^2] - E[f(x)]^2$$

### 1.6 两独立事件的协方差为0证明

$$cov[x, y] = E_{x,y}[x,y] - E[x]E[y] \\ =\sum_x\sum_yp(x,y)xy - E[x]E[y] \\ = \sum_x\sum_yp(x)p(y)xy - E[x]E[y] \\ = E[x]E[y] - E[x]E[y] \\ = 0$$

### 1.7 高斯分布归一性证明

$$I^2 = \int_{-\infty}^\infty\int_{-\infty}^\infty exp(-\frac{1}{2\sigma^2}(x^2+y^2))dxdy$$

极坐标变换，得

$$I^2 = \int_0^{2\pi}\int_0^\infty exp(-\frac{r^2}{2\sigma^2})drd\theta \\ = 2\pi \int_0^\infty exp(-\frac{r^2}{2\sigma^2})dr$$

令$$u = r^2$$， 得

$$I^2 = 2\pi \int_0^\infty exp(-\frac{u}{2\sigma^2})\frac12du \\ = 2\pi\sigma^2$$

因此，有

$$\int_{-\infty}^\infty \mathcal{N}(x | u, \sigma^2) = \int_{-\infty}^\infty \frac{1}{(2\pi\sigma^2)^{\frac12}}exp(-\frac{1}{2\sigma^2}(x-u))dx$$

令$$v = x - u$$，得

$$\int_{-\infty}^\infty \mathcal{N}(x | u, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{\frac12}} \int_{-\infty}^\infty exp(-\frac{1}{2\sigma^2}v)dv = 1$$

### 1.8 高斯分布的期望与方差推导

$$E[x] = \int_{-\infty}^\infty \mathcal{N}(x | \mu, \sigma^2)  = \int_{-\infty}^\infty \frac{1}{(2\pi\sigma^2)^\frac12}exp(-\frac{1}{2\sigma^2}(x-\mu)^2)xdx$$

令$$y = x-\mu$$ 得

$$E[x] = \int_{-\infty}^\infty \frac{1}{(2\pi\sigma^2)^\frac12}exp(-\frac{1}{2\sigma^2}y^2)(y+\mu)dy$$

由$$y = x - \mu$$ 关于x轴对称分布，得

$$E[x] = \mu \int_{-\infty}^\infty \mathcal{N}(x | u, \sigma^2) dx = \mu$$

![0_1544596120441_6411c701-5162-4776-9b50-3c30e2a021d5-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544596121964-6411c701-5162-4776-9b50-3c30e2a021d5-image.png)

### 1.9 高斯分布的众数（即概率最大值）推导

$$\frac{d\mathcal{N}(x | \mu, \sigma^2)}{dx} = - \mathcal{N}(x | \mu, \sigma^2) \frac{x-\mu}{\sigma^2}$$

当$$x = \mu$$ 时，有最大值。

![0_1544596983506_2933c4ff-2900-40ab-ad28-50830ae01039-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/PRML/1544596984570-2933c4ff-2900-40ab-ad28-50830ae01039-image.png)

### 1.10 独立变量相加的期望方差证明

由$$x, z$$ 相互独立，得$$x + z$$ 的联合概率为$$p(x)p(z)$$ ，有

$$E[x + z] = \int (x+z)p(x)p(z)dxdz = \int xp(x)dx + \int zp(z)dz = E[x] + E[z]$$

$$var[x+z] = E[(x+z)^2] - E[x+z]^2 \\ = E[x^2 + z^2 + 2xz] - (E[x] + E[z])^2 \\ = E[x^2] - E[x]^2 + E[z^2] - E[z]^2 \\ = var[x] + var[z]$$

### 1.11
