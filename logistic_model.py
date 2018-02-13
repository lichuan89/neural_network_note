#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

%matplotlib inline


def collect_train_data_2x1y():
    """
    data: (x1, x2) -> (y1)
    2 kind of points: red and blue.
    
    """
    n = 20
    # var: 1.2, mean: [-1, 0]
    x_red = np.random.randn(n, 2) * 1.2 + [-1, 0] 
    x_blue = np.random.randn(n, 2) * 1.2 + [1, 1]
    x = np.vstack((x_red, x_blue))
    y = np.vstack((np.zeros((n,1)), np.ones((n,1))))
    return x, y


def predict(x, w): 
    """
    predict. logistic model
    """
    z = x.dot(w.T)
    return 1 / (1 + np.exp(-z))


def cost(y, t):
    """
    crossEntropy cost with logistic model.
    """
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))



def get_params_grad(w, x, t):
    """
    dcost / dw. use crossEntropy cost with logistic model.
    """
    return (predict(x, w) - t).T * x


def grad_desc_train(x, t, init_w, iterations=8, learning_rate=0.05):
    """
    train. use gradient descent
    """
    w = init_w
    w_costs = [(w, cost(predict(x, w), t))]
    for i in range(iterations):
        delt_w = learning_rate * get_params_grad(w, x, t)
        w = w - delt_w
        w_costs.append((w, cost(predict(x, w), t)))  
        if len(w_costs) > 3:
            if w_costs[-1][1] >= w_costs[-2][1] >= w_costs[-3][1]:
                break
    return (w, w_costs)
        

def show_2_dim_2_label_space(
        x, 
        y, 
        label_red='class red', 
        label_blue='class blue', 
        title='red vs blue',
        predict=None,
        w=None
    ):
    """
    显示数据空间：二维特征、2个标签
    """
    x_red = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    x_blue = np.array([x[i] for i in range(len(x)) if y[i] == 1])
     
    plt.plot(x_red[:,0], x_red[:,1], 'ro', label=label_red)
    plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label=label_blue)
    plt.grid()
    plt.legend(loc=2)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    area = [np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1, np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1]
    plt.axis(area)
    plt.title(title)
    
    if predict is not None:
       
        n = 200
        xs1 = np.linspace(area[0], area[1], num=n)
        xs2 = np.linspace(area[2], area[3], num=n)
        xx, yy = np.meshgrid(xs1, xs2)
        plane = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                plane[i,j] = np.around(predict(np.asmatrix([xx[i,j], yy[i,j]]) , w))
        cmap = ListedColormap([
                colorConverter.to_rgba('r', alpha=0.30),
                colorConverter.to_rgba('b', alpha=0.30)
            ])
        plt.contourf(xx, yy, plane, cmap=cmap)


def show_cost_space_2x1y(x, t, w_iters=None):
    n = 100  
    ws1 = np.linspace(-5, 5, num=n) 
    ws2 = np.linspace(-5, 5, num=n) 
    ws_x, ws_y = np.meshgrid(ws1, ws2)  
    cost_ws = np.zeros((n, n)) 
    for i in range(n):
        for j in range(n):
            cost_ws[i,j] = cost(predict(x, np.asmatrix([ws_x[i,j], ws_y[i,j]])) , t)
    plt.contourf(ws_x, ws_y, cost_ws, 20, cmap=cm.pink)
    
    if w_iters is not None:
        for i in range(1, len(w_iters)): 
            w1, c1 = w_iters[i-1]
            w2, c2 = w_iters[i]
            plt.plot(w1[0, 0], w1[0, 1], 'bo')  # Plot the weight cost value
            plt.plot([w1[0, 0], w2[0, 0]], [w1[0, 1], w2[0, 1]], 'b-')
            plt.text(w1[0, 0] - 0.2, w1[0, 1] + 0.4, '$w({})$'.format(i), color='b')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$cost$', fontsize=15)
    plt.xlabel('$w_1$', fontsize=15)
    plt.ylabel('$w_2$', fontsize=15)
    plt.title('cost_space: error = cost(w)')
    plt.grid()
    

def test():
    x, t = collect_train_data_2x1y()
    show_2_dim_2_label_space(x, t)
    plt.show()
    show_cost_space_2x1y(x, t, w_iters=None)
    plt.show()  

    w, w_cost = grad_desc_train(x, t, np.asmatrix([-4, -2]), iterations=6, learning_rate=0.05)
    show_cost_space_2x1y(x, t, w_iters=w_cost)
    plt.show()  
    show_2_dim_2_label_space(x, t, predict=predict, w=w) 
    plt.show()

if __name__ == "__main__":
    test()
