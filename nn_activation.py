#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import copy


# 参考:
#   https://blog.csdn.net/cckchina/article/details/79915181
#   https://zhuanlan.zhihu.com/p/37740860

def logistic(z):
    o = copy.deepcopy(z)
    o[o > 0] = 1 / (1 + np.exp(-o[o > 0]))
    o[o <= 0] = np.exp(o[o <= 0]) / (1 + np.exp(o[o <= 0]))
    return o 

def logistic_deriv(y):
    return np.multiply(y, (1 - y)) 

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def softmax_deriv(y):
    return np.multiply(y, (1 - y)) 


def relu(z):
    return np.maximum(z, 0)


def relu_deriv(y):
    o = np.ones(y.shape)
    o[y < 0] = 0 
    return o


def tanh(z):
    return np.tanh(z)


def tanh_deriv(y):
    return 1 - y ** 2


def arctan(z):
    return np.arctan(z)


def arctan_deriv(z):
    return 1 / (1 + z ** 2)

def crossEntropy_cost(y, t): 
    """ 
    交叉熵损失函数
    """
    o = copy.deepcopy(y)
    o[o < 1e-10] = 1e-10
    o[o > 0.99] = 1 - 1e-10 
    return - np.multiply(t, np.log(o)).sum() / o.shape[0]

def crossEntropy_cost_deriv(y, t): 
    """ 
    交叉熵损失函数对预测值y的梯度
    """
    return (y - t) / y / (1 - y) / y.shape[0]
