#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""


import numpy as np
import os
import io
from nn_NeuronNetwork import small_train, small_test 
from cv2_common import collect_image_fea 
from common import muti_process

def split_data(data, T):
    from sklearn import datasets, cross_validation, metrics
    X_train, X_test, T_train, T_test = cross_validation.train_test_split(
            data, T, test_size=0.4)
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
            X_test, T_test, test_size=0.5)
    if True:
        print '训练集、校验集、测试集的X量级：', X_train.shape, X_validation.shape, X_test.shape
        print '训练集、校验集、测试集的Y量级：', T_train.shape, T_validation.shape, T_test.shape
    return X_train, T_train, X_validation, T_validation, X_test, T_test


def single_thread_collect_train_data(lines, args):
    """
    单线程处理部分数据
    """
    files = {} 
    for line in lines:
        fpath, tag = line.split('\t')
        files[fpath] = tag
    data, T = collect_image_fea(files, expend_num=2, use_fea='pixel_rgbhist')
    return [(data, T)]


def collect_train_data():
    thread_num = 11
    files = {}
    train_image_fpath = 'data/train_data.list'
    if os.path.exists('%s.fea' % train_image_fpath):
        data = np.loadtxt('%s.fea' % train_image_fpath, delimiter=",")
        T = np.loadtxt('%s.val' % train_image_fpath, delimiter=",")
    else:
        lines = [line[: -1] for line in open(train_image_fpath).readlines()]
        pairs = muti_process(lines, thread_num, single_thread_collect_train_data, args=[])
        data = None
        T = None
        for pair in pairs:
            sub_data, sub_T = pair
            data = np.vstack((sub_data, data)) if data is not None else sub_data
            T = np.vstack((sub_T, T)) if T is not None else sub_T
        np.savetxt('%s.fea' % train_image_fpath, data, fmt="%d", delimiter=",")
        np.savetxt('%s.val' % train_image_fpath, T, fmt="%d", delimiter=",")

    #data = data[:, 1600:]
    #data = data[:, :1600]
    X_train, T_train, X_validation, T_validation, X_test, T_test = split_data(data, T)
    return X_train, T_train, X_validation, T_validation, X_test, T_test 


if __name__ == "__main__":
    print '载入数据集...'
    X_train, T_train, X_validation, T_validation, X_test, T_test = collect_train_data()
    print '训练数据集...'
    small_train(X_train, T_train, X_validation, T_validation, X_test, T_test, 150, 50, save_file='data/tmp.trademark_nn.txt')
    print '预测...'
    small_test('data/tmp.trademark_nn.txt', X_test, T_test)
