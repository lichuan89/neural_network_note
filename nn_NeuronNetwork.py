#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import itertools
import collections
from sklearn import metrics
from nn_activation import logistic
from nn_activation import logistic_deriv
from nn_activation import softmax
from nn_activation import softmax_deriv
from nn_activation import crossEntropy_cost
from nn_activation import crossEntropy_cost_deriv 
from plt_common import show_images, show_array, show_predict_numbers  
g_open_debug = False 

class Layer(object):
    """
    神经网络的层
    """
    
    def get_params_iter(self):
        """
        参数
        """
        
        return []
    
    def get_params_grad(self, X, output_grad):
        """
        损失函数在参数上的梯度。其中output_grad为损失函数在该层输出的梯度。
        """
        return []
    
    def get_output(self, X):
        """
        神经网络这一层的输出
        """
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """
        损失函数在该层输入的梯度。output_grad为损失函数在该层输出的梯度。
        """
        pass

class LinearLayer(Layer):
    """
    线性回归
    """
    
    def __init__(self, n_in, n_out):
        # n_out个神经元，每个神经元接收n_in个输入特征
        self.W = np.random.randn(n_in, n_out) * 0.1 
        self.b = np.zeros(n_out)
        
    def get_params_iter(self):
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        # 输出矩阵：sample_num * n_out
        return X.dot(self.W) + self.b
        
    def get_params_grad(self, X, output_grad):
        # 各个样本的对应梯度之和
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
    
    def get_input_grad(self, Y, output_grad):
        return output_grad.dot(self.W.T)

class LogisticLayer(Layer):
    def get_output(self, X):
        return logistic(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(logistic_deriv(Y), output_grad)


class SoftmaxLayer(Layer):
    def get_output(self, X):
        return softmax(X)
    
    def get_input_grad(self, Y, output_grad):
        return np.multiply(softmax_deriv(Y), output_grad)
    


    
def forward_step(input_samples, layers):
    """
    前向传播，取输入和每一层输出
    """
    activations = [input_samples] 
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)   
        activations.append(Y)   
        X = activations[-1]   
    return activations  

 
    
def backward_step(activations, targets, layers, cost_grad_func):
    """
    后向传播，取损失函数在每一层的梯度
    """
    param_grads = collections.deque()  
    output_grad = None
    for layer in reversed(layers):   
        Y = activations.pop() 
        if output_grad is None:
            output_grad = crossEntropy_cost_deriv(Y, targets)  
        input_grad = layer.get_input_grad(Y, output_grad)

        X = activations[-1]
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        output_grad = input_grad
    return list(param_grads)

class NeuronNetwork(object):
    def __init__(
            self,
            input_feature_num, # 输入数据集样本数 
            layer_neuron_nums, # 每一层的输出节点数 
            layer_active_funcs, # 每一层的激活函数
            cost_func, # 损失函数
            cost_grad_func # 损失函数对应的梯度函数
        ):

        # 构建神经网络
        self.layers = []
        for (neuron_num, active_func) in zip(layer_neuron_nums, layer_active_funcs):
            # 构建每一层: 线性网络 + 激活函数
            self.layers.append(LinearLayer(input_feature_num, neuron_num))
            self.layers.append(active_func())
            input_feature_num = neuron_num
        self.cost_func = cost_func
        self.cost_grad_func = cost_grad_func
    
    def test_accuracy(self, X_test, T_test):
        y_test = self.predict(X_test)
        y_true = np.argmax(T_test, axis=1) 
        y_pred = np.argmax(y_test, axis=1)   
        test_accuracy = metrics.accuracy_score(y_true, y_pred)  
        return test_accuracy
    
    def predict(self, X):
        activations = forward_step(X, self.layers) 
        return activations[-1]
    
 
    def train_once(self, X, T, learning_rate):
        activations = forward_step(X, self.layers) 
        cost = self.cost_func(activations[-1], T) 
        if learning_rate is None: 
            return cost
        param_grads = backward_step(activations, T, self.layers, self.cost_grad_func)   
        for layer, layer_backprop_grads in zip(self.layers, param_grads):
            for param, grad in itertools.izip(layer.get_params_iter(), layer_backprop_grads):
                param -= learning_rate * grad
        return cost
        
    def train_random_grad_desc(
            self,
            X_train,
            T_train,
            X_validation,
            T_validation,
            batch_size,
            max_nb_of_iterations, 
            learning_rate
            ):
        nb_of_batches = X_train.shape[0] / batch_size  # 批处理次数
        # 从训练集中分批抽取(X, Y) 
        XT_batches = zip(
            np.array_split(X_train, nb_of_batches, axis=0),  # X samples
            np.array_split(T_train, nb_of_batches, axis=0))  # Y targets
       
        minibatch_costs = []
        training_costs = []
        validation_costs = []

        # 每一轮迭代
        for iteration in range(max_nb_of_iterations):
            # 每一轮迭代中的批量训练
            for X, T in XT_batches:  # For each minibatch sub-iteration
                cost = self.train_once(X, T, learning_rate)
                minibatch_costs.append(cost)
            cost = self.train_once(X_train, T_train, learning_rate=None)
            training_costs.append(cost)
            cost = self.train_once(X_validation, T_validation, learning_rate=None)
            validation_costs.append(cost)
            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    break
    
        nb_of_iterations = iteration + 1
        costs_vec = [minibatch_costs, training_costs, validation_costs]
        return (validation_costs[-1], nb_of_iterations, costs_vec)
    
    def train_grad_desc(
            self,
            X_train,
            T_train,
            X_validation,
            T_validation,
            max_nb_of_iterations, 
            learning_rate
            ):
        training_costs = []
        validation_costs = []

        for iteration in range(max_nb_of_iterations):
            cost = self.train_once(X_train, T_train, learning_rate)
            training_costs.append(cost)
            cost = self.train_once(X_validation, T_validation, learning_rate=None)
            validation_costs.append(cost)
            if len(validation_costs) > 3:
                if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                    break
    
        nb_of_iterations = iteration + 1
        costs_vec = [training_costs, validation_costs]
        return (validation_costs[-1], nb_of_iterations, costs_vec)





def collect_train_data():
    # 返回数字图片的dict, images字段:图片数组，target字段：图片上的数字数组
    # 1797张8*8图片
    from sklearn import datasets, cross_validation, metrics
    digits = datasets.load_digits()
    
    
    # 用one-hot-encoding的10维向量表示10个数字
    T = np.zeros((digits.target.shape[0],10)) 
    T[np.arange(len(T)), digits.target] += 1

    
    # 把数据集拆分成训练集和测试集
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
            digits.data, T, test_size=0.4)
    # 把测试集拆分成校验集和最终测试集
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
            X_test, T_test, test_size=0.5)

    if g_open_debug:
        print '训练集、校验集、测试集的X量级：', X_train.shape, X_validation.shape, X_test.shape
        print '训练集、校验集、测试集的Y量级：', T_train.shape, T_validation.shape, T_test.shape
    
        # 显示数据集
        for i in range(3):
            show_images(digits.images[i * 10: i * 10 + 10])
    
    return X_train, T_train, X_validation, T_validation, X_test, T_test


def test():  

    # 准备数据集：训练, 校验, 测试 
    X_train, T_train, X_validation, T_validation, X_test, T_test = collect_train_data()

    # 构建神经网络
    nn = NeuronNetwork(
            X_train.shape[1], 
            [20, 20, T_train.shape[1]],
            [LogisticLayer, LogisticLayer, SoftmaxLayer],
            crossEntropy_cost,
            crossEntropy_cost_deriv 
        )

    # 训练
    train_method = 'random_grad_desc'
    if train_method == 'random_grad_desc':
        (cost, terations, costs_vec) = nn.train_random_grad_desc(
                X_train,
                T_train,
                X_validation,
                T_validation,
                batch_size=25,
                max_nb_of_iterations=300, 
                learning_rate=0.1
            )
        minibatch_costs, training_costs, validation_costs = costs_vec
        arrs = (minibatch_costs, training_costs, validation_costs)
        labels = ('cost minibatches', 'cost training set', 'cost validation set')
        show_array(arrs, labels, title='Decrease of cost over backprop iteration')
    elif train_method == 'grad_desc':
        (cost, terations, costs_vec) = nn.train_grad_desc(
                X_train,
                T_train,
                X_validation,
                T_validation,
                max_nb_of_iterations=1600, 
                learning_rate=0.1
            )
        training_costs, validation_costs = costs_vec
        arrs = (training_costs, validation_costs)
        labels = ('cost full training set', 'cost validation set')
        show_array(arrs, labels, title='Decrease of cost over backprop iteration')
        
    # 评估准确率 
    test_accuracy = nn.test_accuracy(X_test, T_test)
    print 'test_accuracy:', test_accuracy   

    
    # 预测
    y_test = nn.predict(X_test)
    y_true = np.argmax(T_test, axis=1) 
    y_pred = np.argmax(y_test, axis=1)   
    show_predict_numbers(y_true, y_pred)    

if __name__ == "__main__": 
    test()
