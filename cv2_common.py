#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""

import numpy as np
import sys
import random
import os
import cv2
import sys
import urllib
import io
import sys
import copy
import traceback
import common
import img_common

g_open_debug = False 

def cv2image_2_file(image, fpath):
    cv2.imwrite(fpath, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def format_cv2image(image):
    """
    rgb -> bgr or bgr -> rgb
    注: 
    cv2.imread(..) 读取的图像矩阵是BGR的,需要转换一下;
    cv2.imwrite(..)写入文件的图像矩阵也必须是BGR的,需要转换一下 
    """
    mat = np.zeros(image.shape)
    mat[:, :, 0] = image[:, :, 2]
    mat[:, :, 1] = image[:, :, 1]
    mat[:, :, 2] = image[:, :, 0]
    return mat


def image_2_cv2image(image):
    return format_cv2image(image)


def cv2image_2_image(image):
    return format_cv2image(image)

def file_2_cv2image(fpath):
    try:
        if fpath.find('http') == 0:
            #fpath = fpath.replace("ms.bdimg.com", "su.bcebos.com")
            #fpath = fpath.replace("boscdn.bpc.baidu.com", "su.bcebos.com")
            context = urllib.urlopen(fpath).read()
            id = os.getpid() 
            fpath = '/tmp/tmp.%d.jpg' % id
            common.str_2_file(context, fpath)
        image = cv2.imread(fpath)
        return image # BGR np.array
    except IOError as error:
        print >> sys.stderr, error
        return None

def move_image(image, x, y):
    """
    左移x，下移y
    """
    image = image.copy()
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def rotate_image(image, angle, center=None, scale=1.0):
    """
    左旋转angle度
    """
    image = image.copy()
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def light_image(image, a=1, b=0):
    """
    亮度缩放a倍,偏置b
    """
    image = image.copy()
    image = image * a + b
    image[image[:, :, :] > 255] = 255
    image[image[:, :, :] < 0] = 0 
    return image

def resize_image(image, width, height): 
    return cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)


def sobel_gray_image(img):
    """ 
    sobel_gray_image 
    """
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  
  
    absX = cv2.convertScaleAbs(x)   
    absY = cv2.convertScaleAbs(y)  
  
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)    
    return dst



def template_image(template, image, point=255):
    """
    对模版template上,灰度为point的点，在image上取0值 
    """
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if template[x][y] != point:
                image[x][y] = 0
            elif image[x][y] == 0:
                image[x][y] = 1
    points = np.array([
            image[x][y] \
            for x in range(image.shape[0]) \
            for y in range(image.shape[1]) if template[x][y] == point
        ])
    return image, points


def check_gray_muzzy(image, width):
    try:
        image = cv2.resize(image, (width, width * image.shape[0] / image.shape[1]))
        laplacian_muzzy = cv2.Laplacian(image, cv2.CV_64F).var()

        #image = cv2.blur(image, (3,3))
        sobel_image = sobel_gray_image(image)
        sobel_q = sobel_image.var()
        
        ret, otsu_image = cv2.threshold(sobel_image, 0, 255, cv2.THRESH_OTSU)
    
        kernel = np.ones((3, 3), np.uint8)
        adjust_image = cv2.dilate(otsu_image, kernel, 1)
        adjust_image = cv2.erode(adjust_image, kernel, 1)
        kernel = np.ones((11, 11), np.uint8)
        adjust_image = cv2.dilate(adjust_image, kernel, 1)
       
        sobel_sobel_image = sobel_gray_image(sobel_image)
        target_image, points = template_image(adjust_image, sobel_sobel_image)
        q = points.var()
        check_image, _ = template_image(adjust_image, image)
        cache = {
                'sobel_image': sobel_image,
                'otsu_image': otsu_image,
                'sobel_sobel_image': sobel_sobel_image,
                'adjust_image': adjust_image,
                'target_image': target_image,
                'check_image': check_image,
            }
        return q, cache
    except IOError as error:
        return None


def random_expend_image(image):
    prob = random.random()
    a = max(0.3, prob * 2)
    prob = random.random()
    b = prob * 16 - 8
    image = light_image(image, a=a, b=b)

    prob = random.random()
    angle = prob * 60 - 30
    image = rotate_image(image, angle=angle)
    prob = random.random()
    x = prob * 20 -10 
    prob = random.random()
    y = prob * 20 - 10
    image = move_image(image, x, y)
    return image


def collect_image_fea(files, expend_num=10, use_fea='pixel'):
    """
    files = {'image_file1': '1', 'image_url1': '0'}
    expend_num, 扩展图片的个数，针对1的case
    """
    mats = None
    data = None
    T = None
    idx = 0
    for fname in files:
        idx += 1

        # 抓图
        cv2image = file_2_cv2image(fname)
        if cv2image is None:
            print >> sys.stderr, 'failed to wget image:%s' % fname 
            continue
        image =  cv2image_2_image(cv2image)
        if image is None:
            print >> sys.stderr, 'failed to format image:%s' % fname 
            continue

        # 扩充图片
        imgs = [image]
        if files[fname] != '0': 
            imgs += [random_expend_image(image) for i in range(expend_num)]

        i = 0
        if g_open_debug:
            print >> sys.stderr, 'begin to process %s' % fname 
        for img in imgs:
            # 抽取特征
            if use_fea == 'pixel':
                vec = img_common.rgb_2_feature(image, gray_size=(40, 40), hist_size=False, rgb_hist_dim=False, gray_hist=False)
            elif use_fea == 'rgbhist':
                vec = img_common.rgb_2_feature(image, gray_size=False, hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
            elif use_fea == 'pixel_rgbhist':
                vec = img_common.rgb_2_feature(image, gray_size=(40, 40), hist_size=(100, 100), rgb_hist_dim=16, gray_hist=False)
            data = np.vstack((data, vec)) if data is not None else vec
            if g_open_debug:
                mat = np.array(img_common.rgb_2_gray(resize_image(image, 50, 50)), ndmin=3)
                mats = np.vstack((mats, mat)) if mats is not None else mat
            t = np.array([[int(files[fname]), 1 - int(files[fname])]])
            T = np.vstack((T, t)) if T is not None else t 
            if g_open_debug:
                cv2image_2_file(image_2_cv2image(img), 'output/%s.%d.jpg' % (idx, i))
            i += 1
        if g_open_debug:
            print >> sys.stderr, 'finish to process %s' % fname 
    if g_open_debug:
        import plt_common
        plt_common.show_2d_images(mats[0: 50])
    return data, T 

def test_read_write():
    image = np.array( # 2行3列的RGB图像
        [
            [[254,   0,   5], [255,   4,   0], [  0,   0,   2], [ 98, 162, 224],],
            [[252,   2,   3], [  1, 255,   0], [  0,   3, 255], [254, 253, 249],],
        ], dtype='uint8'
        )
    cv2image_2_file(image_2_cv2image(image), 'code.bmp')
    image = cv2image_2_image(file_2_cv2image('code.bmp'))
    print image

def test_opt():
    image = cv2image_2_image(file_2_cv2image('test.jpg'))
    #image = rotate_image(image, 20)
    #image = light_image(image, 1, -40)
    #image = resize_image(image, width=100, height=50)
    #image = move_image(image, 50, 50)
    image = random_expend_image(image)
    cv2image_2_file(image_2_cv2image(image), 'test.bmp')


def test_img_fea():
    files = {
        'http://ms.bdimg.com/dsp-image/1371700633.jpg': '1', # badcase
        'http://ms.bdimg.com/dsp-image/1373219344.jpg': '1', 
        'http://ms.bdimg.com/dsp-image/259532757.jpg': '1',
        'http://ms.bdimg.com/dsp-image/828016156.jpg': '0',
        'http://ms.bdimg.com/dsp-image/949458037.jpg': '0',
        'http://ms.bdimg.com/dsp-image/553264679.jpg': '0',
    }
    #data, T = collect_image_fea(files, expend_num=2, use_fea='pixel_rgbhist')
    data, T = collect_image_fea(files, expend_num=2, use_fea='pixel')
    #data, T = collect_image_fea(files, expend_num=2, use_fea='rgbhist')
    print data.shape, T.shape


def test_muzzy():
    image = file_2_cv2image('muzzy/1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_image = sobel_gray_image(image)
    cv2image_2_file(sobel_image, 'test.bmp')
    muzzy_q, cache = check_gray_muzzy(image, width=200)
     

def test_proces_dir(path):
    arr = os.listdir(path)
    for v in arr:
        if v.find('.') == 0:
            continue
        f = '%s/%s' % (path, v)
        if os.path.isfile(f):
            image = file_2_cv2image(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            muzzy_q, cache = check_gray_muzzy(image, width=500)
            for k, img in cache.items():  
                cv2image_2_file(img, 'output/%s.%s.bmp' % (v, k))
            print '%s\t%f' % (f, muzzy_q)
            


if __name__ == "__main__":
    test_proces_dir('muzzy')
