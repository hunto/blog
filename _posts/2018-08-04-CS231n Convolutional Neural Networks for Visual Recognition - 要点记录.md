---
layout: post
cover: 'https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533314431152-88059bbd-1468-437f-a9b9-ffc3109d6364-image.png'
title: 'CS231n Convolutional Neural Networks for Visual Recognition - 要点记录'
subtitle: 'CS231n Convolutional Neural Networks for Visual Recognition'
date: 2018-08-04
categories: CS231n
tags: CS231n 机器学习 深度学习 CV
---

所有课件及Assignments可见我的[Github:hunto/CS224n](https://github.com/hunto/CS231n)

# Lecture2
## 1. Distance Metric
### 1.1 L1 (Manhattan) distance

$$d_1(I_1,I_2)=\sum_p |I_1^p - I_2^p|$$

### 1.2 L2 (Euclidean) distance

$$d_2(I_1, I_2)=\sqrt{\sum_p(I_1^p-I_2^p)^2}$$

---

# Lecture3
## 1. SVM Hinge loss

$$L_i = \sum_{j\neq y_i}max(0, s_j - s_{y_i}+1)$$

## 2. Regularization
**Prevent overfit.**

$$L(W) = \frac1N \sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W)$$

### 2.1 L1 regularization

$$R(W) = \sum_k \sum_l W^2_{k,l}$$

### 2.2 L2 regularization

$$R(W) = \sum_k \sum_l |W_{k,l}|$$

### 2.3 Elastic net (L1 + L2)

$$R(W) = \sum_k \sum_l \beta W_{k,l}^2 + |W_{k,l}|$$

## 3. Softmax -- score -> probabilities

$$softmax_i(X)=\frac{exp(X_i)}{\sum_j^N exp(X_j)}$$

## 4. Softmax cross-entropy loss

$$L_i = -log(softmax_i(X))$$

## 5. Gradient descent

---

# Lecture4

* Backpropagation
* Chain rule

---

# Lecture5

## 1. Fully Connected Layer -- change dims & length
![0_1533312327089_42b2e6af-943e-4e2d-b55a-d1056a29519e-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533312331611-42b2e6af-943e-4e2d-b55a-d1056a29519e-image.png)  

## 2. Convolution Layer

## 3. Pooling Layer
* Max pool
* Average pool

---

# Lecture6

## 1. Mini-batch SGD
Loop:
1. **Sample** a batch of data
2. **Forward** computation & calculate loss
3. **Backprop** to calculate gradients
4. **Optimizer** to update the parameters using gradients

## 2. Activation Functions
* Sigmoid

  $$\sigma (x) = \frac{1}{1+e^{-x}}$$

* tanh

  $$tanh(x)$$

* ReLU

  $$max(0,x)$$

![0_1533312898795_cdb45c2b-6f24-46ea-b6a9-480d62d51418-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533312902745-cdb45c2b-6f24-46ea-b6a9-480d62d51418-image.png) 

## 3. Vanishing Gradients & Exploding Gradients

[机器学习中梯度消失与梯度爆炸问题详解](https://hunto.github.io/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2018/07/17/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E9%97%AE%E9%A2%98%E8%AF%A6%E8%A7%A3.html)

## 4. Weight Initialization
* Small randon numbers -- NO!
* Xavier initialization
  ```Python
  W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
  ```
* ...

## 5. Batch Normalization
1. Compute the empirical mean and variance independently for each dimension
2. Normalize

  $$\hat x^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt {Var[x^{(k)}]}}$$

**Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.**

![0_1533313764133_12cd2165-9c01-48a0-971f-55a7a1c0219f-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533313772058-12cd2165-9c01-48a0-971f-55a7a1c0219f-image.png) 

![0_1533356513089_f9789dc0-5c43-435c-9b8c-78bc63b2d91d-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533356518210-f9789dc0-5c43-435c-9b8c-78bc63b2d91d-image.png) 

**Features**
* Improves gradient flow through the network
* Allows higher learning rates
* Reduces the strong dependence on initialization
* Acts as a form of regularization in a funny way, and slightly reduces the need for dropout, maybe

---

# Lecture7
## 1. Optimizer
* SGD

$$x_{t+1}=x_t-\alpha \nabla f(x_t)$$

* SGD + Momentum

$$v_{t+1}=\rho v_t+\nabla f(x_t)$$

$$x_{t+1}=x_t-\alpha v_{t+1}$$

* Nesterov Momentum

$$v_{t+1}=\rho v_t-\alpha\nabla f(x_t+\rho v_t)$$

$$x_{t+1}=x_t+v_{t+1}$$

* AdaGrad

$$dx=\nabla f(x_t)$$

$$g = g+(dx)^2$$

$$x_{t+1}=x_t - \alpha \frac{dx}{\sqrt g +  10^{-7}}$$

* RMSProp

$$dx=\nabla f(x_t)$$

$$g = \beta g+(1-\beta)(dx)^2$$

$$x_{t+1}=x_t - \alpha \frac{dx}{\sqrt g +  10^{-7}}$$

* Adam (almost)
  ```python
  first_moment = 0
  second_moment = 0
  while True:
      dx = compute_gradient(x)
      first_moment = beta1 * first_moment + (1 - beta1) * dx
      second_moment = beta2 * sencond_moment + (1 - beta2) * dx * dx
      x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
  ```

* Adam (full form)
![0_1533359223607_c45b7ca1-dee8-42e7-9fdd-475de3048f58-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533359225868-c45b7ca1-dee8-42e7-9fdd-475de3048f58-image.png) 

## 2. Dropout
![0_1533359359049_b68f889a-cfac-4caa-9f3a-97a7ed978d74-image.png](https://raw.githubusercontent.com/hunto/hunto.github.io/master/assets/img/CS231n/1533359363244-b68f889a-cfac-4caa-9f3a-97a7ed978d74-image.png) 
