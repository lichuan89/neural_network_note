#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np

def logistic(z): 
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y):
    return np.multiply(y, (1 - y)) 

def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def softmax_deriv(y):
    return np.multiply(y, (1 - y)) 

def crossEntropy_cost(y, t): 
    """ 
    交叉熵损失函数
    """
    return - np.multiply(t, np.log(y)).sum() / y.shape[0]

def crossEntropy_cost_deriv(y, t): 
    """ 
    交叉熵损失函数对预测值y的梯度
    """
    return (y - t) / y / (1 - y) / y.shape[0]
