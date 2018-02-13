#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

def collect_train_data_1x1y():
    """
    data: y = x * 2 + noise
    """
    def f(x): return x * 2
    np.random.seed(seed=1)
    x = np.random.uniform(0, 1, 20)
    noise_variance = 0.2  
    noise = np.random.randn(x.shape[0]) * noise_variance
    y = f(x) + noise
    return x, y


def predict(x, w): 
    """
    predict. linear model 
    """
    return x * w


def cost(y, t): 
    """
    cost. error
    """
    return ((y - t) ** 2).sum() 


def cost_deriv(y, t):
    """
    gradient. dcost / dy
    """
    return 2 * (y - t)

def get_params_grad(w, x, t):
    """
    gradient. dcost / dw = dcost / dy * dy / dw
    """
    return x.T.dot(cost_deriv(predict(x, w), t))


def grad_desc_train(x, t, init_w, iterations=8, learning_rate=0.05):
    """
    train. use gradient descent
    """
    w = init_w
    w_costs = [(w, cost(predict(x, w), t))]
    for i in range(iterations):
        dw = learning_rate * get_params_grad(w, x, t)
        w = w - dw
        w_costs.append((w, cost(predict(x, w), t)))  
        if len(w_costs) > 3:
            if w_costs[-1][1] >= w_costs[-2][1] >= w_costs[-3][1]:
                break
    return (w, w_costs)


def show_train_data_1x1y(x, y, predict=None, w=None):
    """
    show data and predict function. 
    one input one output.
    """
    plt.plot(x, y, 'o', label='t')
    if predict is not None:
        xs = np.linspace(np.min(x), np.max(x), num=100)
        ys = predict(xs, w)
        plt.plot(xs, ys, 'r-', label='y')
    plt.grid()
    plt.legend(loc=2)
    plt.xlabel('$x$', fontsize=15)
    plt.ylabel('$t$', fontsize=15)
    plt.title('train data: t = f(x)')
     
    
def show_cost_space_1x1y(x, y, w_cost=None):
    """
    show cost function.
    cost = func(w)
    """
    ws = np.linspace(0, 4, num=100)
    cost_ws = np.vectorize(lambda w: cost(predict(x, w) , y))(ws)
    plt.plot(ws, cost_ws, 'r-')
    if w_cost is not None:
        for i in range(0, len(w_cost)-2):
            w1, c1 = w_cost[i]
            w2, c2 = w_cost[i + 1]
            plt.plot(w1, c1, 'bo')
            plt.plot([w1, w2], [c1, c2], 'b-')
            plt.text(w1, c1 + 0.5, '$w({})$'.format(i))
    plt.title('cost_space: error = cost(w)')
    plt.xlabel('$w$', fontsize=15)
    plt.ylabel('$error$', fontsize=15)
    plt.grid()
 
def test():
    x, t = collect_train_data_1x1y()
    
    show_train_data_1x1y(x, t, predict=None, w=None)
    plt.show() 
    show_cost_space_1x1y(x, t, w_cost=None)
    plt.show()
    
    
    w, w_cost = grad_desc_train(x, t, 0.1)    
    
    show_train_data_1x1y(x, t, predict, w)
    plt.show()
    show_cost_space_1x1y(x, t, w_cost)
    plt.show()
    
test()
