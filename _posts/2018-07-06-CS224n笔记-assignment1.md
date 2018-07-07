---
layout: post
title: 'CS224n笔记 - assignment1'
subtitle: '课程简介'
date: 2018-07-06
categories: 机器学习 NLP CS224n
tags: CS224n 机器学习 深度学习 NLP
---

所有课件及Assignments可见我的[Github:hunto/CS224n](https://github.com/hunto/CS224n)

# Assignment1
## q1_softmax
softmax函数很简单，每一个元素的e次幂除以所有元素的e次幂之和，即：
$$softmax(x)=\frac{e^{x_i}}{\sum_j e^{x_j}}$$

但是直接把公式放到计算机中使用numpy计算肯定是不行的。e的1000次方肯定爆，且计算量太大，于是我们根据定理：
$$softmax(x+c)=softmax(x)$$
可以令c=max(x)，这样就不会产生计算错误出现无穷数infty的情况了。

实现的代码如下：
```python
if len(x.shape) > 1:
    # Matrix
    ### YOUR CODE HERE
    cal_softmax = lambda x: np.exp(x) / np.repeat(np.sum(np.exp(x), axis=1, keepdims=True), x.shape[1], axis=1)
    x = cal_softmax(x - np.reshape(np.max(x, axis=1), [x.shape[0], 1]))
    ### END YOUR CODE
else:
    # Vector
    ### YOUR CODE HERE
    cal_softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    x = cal_softmax(x - np.max(x))
    ### END YOUR CODE
```

## q2
### q2_sigmoid
首先需要实现sigmoid函数，sigmoid function定义如下：
$$\sigma(x)=\frac{1}{1+e^{-x}}$$

q2_sigmoid.py实现代码如下：
* sigmoid(x)
    ```python
    s = 1 / (1 + np.exp(-x))
    ```
* sigmoid_grad(s)
    ```python
    ds = s * (1 - s)
    ```

### q2_gradcheck
接着实现grad_check用于之后的测试。
```python
x[ix] += h

random.setstate(rndstate)
new_f1 = f(x)[0]

x[ix] -= 2 * h

random.setstate(rndstate)
new_f2 = f(x)[0]

x[ix] += h
```

### q2_neural
最关键的地方，这里定义了一个简单的网络：
![0_1530862803490_9d8b0efe-88e4-4f7a-8ade-c370d54338bd-image.png](http://bbs.dian.org.cn/assets/uploads/files/1530862803989-9d8b0efe-88e4-4f7a-8ade-c370d54338bd-image.png) 
其中，
$$h=sigmoid(xW_1+b_1), \hat{y}=softmax(hW_2+b_2)$$
这些即为网络的前端部分。
```python
h = sigmoid(np.dot(X, W1) + b1) # hidding layer
y_hat = softmax(np.dot(h, W2) + b2) # output y
```
网络后端部分为loss与梯度。
```python
cost = np.sum(-np.log(y_hat[labels == 1])) / X.shape[0]
d3 = (y_hat - labels) / X.shape[0]
gradW2 = np.dot(h.T, d3)
gradb2 = np.sum(d3, axis=0, keepdims=True)

dh = np.dot(d3, W2.T)
grad_h = sigmoid_grad(h) * dh

gradW1 = np.dot(X.T, grad_h)
gradb1 = np.sum(grad_h, axis=0)
```

到这里，一个简单的网络就定义完了

## q3_word2vec
### normalizeRows  
首先需要实现一个normalizeRows函数用于将向量单位化（ljht这个名称我还给老师了），也就是向量元素平方和为1。
```python
x = x / np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))
```

### softmaxCostAndGradient
该函数与q2中实现的大致相同
```python
y_hat = softmax(np.dot(predicted, outputVectors))
cost = np.log(np.sum(np.exp(np.dot(predicted, outputVectors)),
                        axis=0)) - outputVectors[target].dot(predicted)

gradPred = np.sum(y_hat.reshape(-1, 1) * outputVectors,
                    axis=0) - outputVectors[target]

grad = y_hat.reshape(-1, 1) * np.tile(predicted, (outputVectors.shape[0], 1))

grad[target] -= predicted
```

**后面的内容见github吧**