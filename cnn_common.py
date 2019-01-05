#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import math
import itertools
import sys
import os
import io
from common import log

g_open_debug = False 

def template_2_col(template):
    """
    输入卷积核矩阵:行数 * 列数 * 通道数 * 卷积核数
    将卷积核展开成img_2_col需要的格式
    """
    row, col, channel, n = template.shape
    #template = template.transpose(3, 2, 0, 1).reshape(n, channel, -1).reshape(n, -1).transpose(1, 0)
    #template = template.transpose(3, 0, 1, 2).reshape(n, -1).transpose(1, 0)
    template = template.reshape((-1, template.shape[-1]))
    return template 


def img_2_col(img, ksize, stride=1):
    """
    4维图片: num, row, col, channel
    返回的每个图像的每一行是一个卷积覆盖的像素, 而且是channel通道,
    格式为 位置1通道1灰度， 位置1通道2灰度, ..., 位置2通道1灰度, ...
    """
    n = img.shape[0]
    cols = np.zeros((n, (img.shape[1] - ksize[0] + 1) * (img.shape[2] - ksize[1] + 1), ksize[0] * ksize[1] * img.shape[3]))
    k = 0
    for r in range(0, img.shape[1] - ksize[0] + 1, stride):
        for c in range(0, img.shape[2] - ksize[1] + 1, stride):
            # 取卷积核覆盖的区域
            col = img[:, r: r + ksize[0], c: c + ksize[1], :]
            # 展开: 像素1通道1， 像素1通道2, ..., 像素2通道1, ...
            #col = col.transpose(0, 3, 1, 2).reshape((n, -1))
            col = col.reshape((n, -1))
            # 每一行表示第k个卷积核覆盖区域展开的矩阵
            cols[:, k] = col
            k += 1 
    cols = np.array(cols)
    return cols

def pad(img, trow, tcol):
    delt_trow = -1 if trow % 2 == 0 else 0
    delt_tcol = -1 if tcol % 2 == 0 else 0
    img = np.pad(img, ((0, 0), (trow / 2, trow /2 + delt_trow), (tcol / 2, tcol /2 + delt_tcol), (0, 0)), 'constant', constant_values=0)
    return img

def conv(img, template, bias=None, stride=1):
    """
    bias的维数 = 卷积核数 tn
    """
    log('begin to conv.', img.shape, template.shape)
    num, row, col, channel = img.shape
    trow, tcol, tchannel, tn = template.shape
    if bias is None:
        bias = np.zeros(tn)
    img = pad(img, trow, tcol) 
    cols = img_2_col(img, template.shape, stride)
    exp_template = template_2_col(template)
    if g_open_debug:
        print >> sys.stderr, 'ori image shape:%s, template shape:%s' % (img.shape, template.shape)
        print >> sys.stderr, 'expand image shape:%s, template shape:%s' % (cols.shape, exp_template.shape)
    output = np.dot(cols, exp_template) + bias
    #output = output.transpose(0, 2, 1).reshape((num, tn, row, col)).transpose(0, 2, 3, 1)
    output = output.reshape((num, row, col, tn))
    log('finish to conv.')
    return output, cols

def conv_gradient(X, output_grad, W, b, stride=1):
    log('begin to conv gradient.', X.shape, output_grad.shape)
    y, x_cols = conv(X, W, b, stride)
    output_grad_col = output_grad.reshape((output_grad.shape[0], -1, output_grad.shape[-1]))
    log('begin to conv JW.')
    JW = np.zeros(W.shape)
    Jb = np.zeros(b.shape)
    for i in range(X.shape[0]):
        JW += np.dot(x_cols[i].transpose((1, 0)), output_grad_col[i]).reshape(W.shape)
    
    log('finish to calc JW.')
    Jb = np.sum(output_grad_col, axis=(0, 1))

    log('finish to calc Jb.')
    expand_output_grad = pad(output_grad, W.shape[0], W.shape[1])
    flip_W_col = W.reshape([-1, W.shape[-2], W.shape[-1]])
    flip_W_col = flip_W_col[::-1, ...]
    flip_W_col = flip_W_col.swapaxes(1, 2)
    W_col = flip_W_col.reshape([-1, W.shape[-2]]) 
    expand_output_grad_col = img_2_col(expand_output_grad, W.shape, stride)   
    JX = np.dot(expand_output_grad_col, W_col).reshape(X.shape) 
    log('finish to calc JX.')
    log('finish to conv gradient.')
    return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))], JX



def maxpool(x, ksize=(2, 2), stride=2):
    """
    4维图片: num, row, col, channel
    """
    log('begin to calc maxpool:', x.shape, ksize)
    num, row, col, channel = x.shape
    out = np.zeros((num, int(math.ceil(1.0 * row / stride)), int(math.ceil(1.0 * col / stride)), channel))
    index = np.zeros((num, row, col, channel))
    for r in range(0, row, stride):
        for c in range(0, col, stride):
            rend = min(r + ksize[0], row)
            cend = min(c + ksize[1], col)
            region = x[:, r: rend, c: cend, :].reshape(num, -1, channel)
            out[:, int(math.floor(1.0 * r / stride)), int(math.floor(1.0 * c / stride)), :] = np.max(region, axis=1)
            box = region.transpose(0, 2, 1)
            idx = np.argmax(box, axis=2)
            mask = np.zeros(box.shape)
            mask[np.arange(mask.shape[0])[:,None],np.arange(mask.shape[1]),idx] = 1
            mask = mask.transpose(0, 2, 1).reshape((num, rend - r, cend - c, channel))
            index[:, r: rend, c: cend, :] = mask
    log('finish to calc maxpool:', x.shape, ksize)
    return out, index


def maxpool_v2(x, ksize=(2, 2), stride=2):
    """
    4维图片: num, row, col, channel
    """
    log('begin to calc maxpool:', x.shape, ksize)
    num, row, col, channel = x.shape
    out = np.zeros((num, int(math.ceil(1.0 * row / stride)), int(math.ceil(1.0 * col / stride)), channel))
    index = np.zeros((num, row, col, channel))
    for n in range(num):
        for m in range(channel):
            for r in range(0, row, stride):
                for c in range(0, col, stride):
                    rend = min(r + ksize[0], row)
                    cend = min(c + ksize[1], col)
                    out[n, int(math.floor(1.0 * r / stride)), int(math.floor(1.0 * c / stride)), m] = np.max(x[n, r: rend, c: cend, m])
                    idx = np.argmax(x[n, r: rend, c: cend, m])
                    index[n, r + idx / (cend - c), c + idx % (cend - c), m] = 1
    log('finish to calc maxpool:', x.shape, ksize)
    return out, index


def maxpool_gradient(grad, index, ksize=(2, 2), stride=1):
    log('begin to calc maxpool_gradient:', grad.shape, ksize)
    d = np.repeat(np.repeat(grad, stride, axis=1), stride, axis=2) * index
    log('finish to calc maxpool_gradient')
    return d

def test(flag):
    if flag == 'conv_image':
        from cv2_common import cv2image_2_file, file_2_cv2image
        img = file_2_cv2image('test.jpg')
        print img.shape
        t1 = np.array([
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
            ])
        t2 = np.array([
                [ 1,  1,  1],
                [ 0,  0,  0],
                [-1, -1, -1]
            ])
        imgs = np.array([img])
        template = np.array([ # 3个卷积核, 3个通道
                    [t1, t1, t1],
                    [t2, t2, t2],
                    [t1, t2, t1],
            ])
        template = template.transpose(2, 3, 1, 0)
        bias = np.array([200, 200, 100])
        output, cols =  conv(imgs, template, bias, stride=1)
        print output.shape
        #output = output.transpose(0, 2, 1).reshape((1, 3, 348, 348)).transpose(0, 2, 3, 1)
        cv2image_2_file(output[0], 'test.bmp')   
        Jparam, JX = conv_gradient(imgs, np.random.standard_normal((imgs.shape)), template, bias, stride=1)

    if flag == 'pool_image':
        from cv2_common import cv2image_2_file, file_2_cv2image
        from img_common import rgb_resize
        img = file_2_cv2image('test.jpg')
        img = rgb_resize(img, (400, 200))
        imgs = np.array([img])
        size = imgs.shape
        imgs = imgs.reshape((size[0], -1)).reshape((size[0], -1, size[-1])).reshape(size)
        output, index = maxpool(imgs, ksize=(4, 4), stride=2)
        cv2image_2_file(output[0], 'test.bmp')


if __name__ == "__main__":
    #test('pool_image') 
    img = np.random.standard_normal((100, 10, 12))
    imgs = np.array([img, img])
    o1, i1 = maxpool(imgs, ksize=(2, 2), stride=2)
    o2, i2 = maxpool_v2(imgs, ksize=(2, 2), stride=2)
    print set(list((i1 == i2).reshape(-1)))
    print set(list((o1 == o2).reshape(-1)))
    
