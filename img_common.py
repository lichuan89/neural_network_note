#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""

import os
import io
import numpy as np
import urllib


def rgb_2_gray(image):
    """
    gray = r * 0.3 + g * 0.59 + b * 0.11
    等价于: np.asarray(image.convert('L'))
    等价于: cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    """
    gray = (image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11).astype('int')
    return gray

def gray_2_rgb(image):
    rgb = np.zeros([image.shape[0], image.shape[1], 3])
    rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = image
    return rgb

def gray_2_hist(grayImage):
    hist = np.zeros((1, 256), np.float32)
    for rr in range(grayImage.shape[0]):
        for cc in range(grayImage.shape[1]):
            hist[0, grayImage[rr, cc]] += 1
    return hist

def rgb_2_hist(image, l=16):
    hist = np.zeros([l*l*l, 1], np.float32)
    hsize = 256/l
    rows, cols, channels = image.shape
    for rr in range(rows):
        for cc in range(cols):
            b, g, r = image[rr, cc]
            index = np.int(b/hsize)*l*l + np.int(g/hsize)*l + np.int(r/hsize)
            hist[np.int(index), 0] = hist[np.int(index), 0] + 1
    hist = hist.reshape(1, hist.shape[0] * hist.shape[1])
    return hist


def rgb_resize(image, size):
    height, width = image.shape[: 2]
    dstHeight, dstWidth = size
    dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
    for row in range(dstHeight):
        for col in range(dstWidth):
            oldRow = int(row * (height * 1.0 / dstHeight))
            oldCol = int(col * (width * 1.0 / dstWidth))
            dstImage[row, col] = image[oldRow, oldCol]
    return dstImage



def rgb_2_feature(image, gray_size=(50, 50), hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False):
    """
    参数如果是False,表示不用这个特征
    """ 
    vec = np.ones((1, 0))
    if rgb_hist_dim != False or gray_hist != False:
        fea_img = rgb_resize(image, hist_size)
    # rgb直方图
    if rgb_hist_dim != False:
        rgb_hist = rgb_2_hist(fea_img, l=rgb_hist_dim)
    else:
        rgb_hist = np.ones((1, 0))

    # 灰度直方图
    if gray_hist != False: 
        gray_img = rgb_2_gray(fea_img)
        gray_hist = gray_2_hist(gray_img)
    else:
        gray_hist = np.ones((1, 0))

    # 像素矩阵
    if gray_size != False: 
        img = rgb_resize(image, gray_size)
        gray_fea = rgb_2_gray(img).reshape(1, gray_size[0] * gray_size[1])
    else:
       gray_fea = np.ones((1, 0))
    vec = np.hstack((rgb_hist, gray_hist, gray_fea)) 
    return vec
   
 
def test():
    image = np.array( # 2行3列的RGB图像
        [
            [
                [254,   0,   5],
                [255,   4,   0],
                [  0,   0,   2],
                [ 98, 162, 224],
            ],
            [
                [252,   2,   3],
                [  1, 255,   0],
                [  0,   3, 255],
                [254, 253, 249],
            ]
        ]
        )
    gray = rgb_2_gray(image)
    hist = gray_2_hist(gray)
    rgb = gray_2_rgb(gray)
    hist = rgb_2_hist(image, l=4)
    bigImage = rgb_resize(image, (200, 100))
    fea = rgb_2_feature(image, gray_size=(50, 50), rgb_hist_dim=16, gray_hist=True)

if __name__ == "__main__":
    # cv2和numpy中的图片shape都是(height, width), pil中的图片size是(width, height)
    # cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    test()
