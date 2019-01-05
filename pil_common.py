#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan89@126.com
    @date   2016/09/12  
    @note
"""


from PIL import Image
import numpy as np
import urllib
import random
import io
import os
import sys
import plt_common

g_open_debug = False 


def image_2_pilimage(image):
    dst = Image.new('RGB', image.size)
    



def change_image(image, arg):
    """
    调整图像 
    """
    try:
        image = image.copy()
        if 'light' in arg: 
            image = image.point(lambda p: p * arg['light'])
        if 'rotate' in arg:
            image = image.rotate(arg['rotate'])
        if 'transpose' in arg:
            image = image.transpose(Image.FLIP_LEFT_RIGHT if arg['transpose'] == 'right' else Image.FLIP_RIGHT_LEFT)
        if 'deform' in arg:
            image = image.resize(arg['deform'])
        if 'crop' in arg: # 提取4个角落和中心区域
            #scale = 0.875
            scale = arg['scale']
            w, h = image.size 
            ww = int(w * scale)
            hh = int(h * scale)
            x = y = 0
            if arg['crop'] == 'left,up':
                # 图像起点，左上角坐标
                # 切割左上角
                x_lu = x
                y_lu = y
                image = image.crop((x_lu, y_lu, x_lu + ww, x_lu + hh))
            elif arg['crop'] == 'left,down': 
                # 切割左下角
                x_ld = int(x)
                y_ld = int(y + (h - hh))
                image = image.crop((x_ld, y_ld, x_ld + ww, y_ld + hh))
            elif arg['crop'] == 'right,up':
                # 切割右上角
                x_ru = int(x + (w - ww))
                y_ru = int(y)
                image = image.crop((x_ru, y_ru, x_ru + w, y_ru + hh))
            elif arg['crop'] == 'right,down':
                # 切割右下角
                x_rd = int(x + (w - ww))
                y_rd = int(y + (h - hh))
                image = image.crop((x_rd, y_rd, x_rd + w, y_rd + h))
            elif arg['crop'] == 'center': 
                # 切割中心
                x_c = int(x + (w - ww) / 2)
                y_c = int(y + (h - hh) / 2)
                image = image.crop((x_c, y_c, x_c + ww, y_c + hh))
    except Exception as e:
        print >> sys.stderr, e
        return None
   
    return image 


def test_change_image():
    try:
        #image = Image.open('test.jpg')
        context = open('test.jpg').read()
        image = Image.open(io.BytesIO(context))
        arg = {
                #'light': 2, 
                #'rotate': 40, 
                'transpose': 'right',
                'deform': (100, 100),
                #'crop': 'right,down',
                #'scale': 0.7,
        }
        image = change_image(image, arg) 
        image.save('out.jpg')
    except Exception as e:
        print >> sys.stderr, e
        pass


def expend_image(image):
    arg = {}
    prob = random.random()
    arg['light'] = max(0.3, prob * 2)

    prob = random.random()
    arg['rotate'] = prob * 20 # prob * 360

    prob = random.random()
    arg['scale'] = max(0.8, prob)
    
    idx_tag = {0: 'left,up', 1: 'left,down', 2: 'right,up', 3: 'right,down', 4: 'center'}
    arg['crop'] = idx_tag[int(prob * 10)/2]

    image = change_image(image, arg)
    arg = {'deform': (100, 100)}
    image = change_image(image, arg)
    return image


def collect_rgb_hist(image, l=16):
    width, height = image.size
    pix = image.load()
    hist = np.zeros([l*l*l, 1], np.float32)
    hsize = 256/l
    for x in range(width):
        for y in range(height):
            if type(pix[x, y]) == int:
                r = g = b = pix[x, y]
            elif len(pix[x, y]) == 3:
                r, g, b = pix[x, y]
            elif len(pix[x, y]) == 4:
                r, g, b, c = pix[x, y] 
            index = np.int(b/hsize)*l*l + np.int(g/hsize)*l + np.int(r/hsize)
            hist[np.int(index), 0] = hist[np.int(index), 0] + 1 
    return hist

def file_2_pilimage(fpath, is_return_mat=False):
    try:
        if fpath.find('http') != 0:
            image = Image.open(fpath)
        else:
            fpath = fpath.replace("ms.bdimg.com", "su.bcebos.com")
            fpath = fpath.replace("boscdn.bpc.baidu.com", "su.bcebos.com")
            context = urllib.urlopen(fpath).read()
            image = Image.open(io.BytesIO(context))
        if not is_return_mat:
            return image
        pix = image.load()
        width, height = image.size
        mat = np.zeros([height, width, 3])
        for x in range(width):
            for y in range(height):
                if type(pix[x, y]) == int:
                    r = g = b = pix[x, y]
                elif len(pix[x, y]) == 3:
                    r, g, b = pix[x, y]
                elif len(pix[x, y]) == 4:
                    r, g, b, c = pix[x, y]
                mat[y, x, :] = [r, g, b]
        return image, mat
    except IOError as error:
        print >> sys.stderr, error
        return None


def collect_basic_image_feature(img, use_hist=False, size=(50, 50)):
    if use_hist:
        rgb_hist = collect_rgb_hist(img, l=16)
        rgb_hist = rgb_hist.reshape(1, rgb_hist.shape[0] * rgb_hist.shape[1])
    img = change_image(img, {'deform': size})
    vec = np.asarray(img.convert('L'), dtype='int').reshape(1, img.size[0] * img.size[1])
    if use_hist:
        vec = np.hstack((rgb_hist, vec))
    return vec


def collect_image_train_data(files, expend_num=10, use_hist=False):
    mats = None
    data = None
    T = None
    idx = 0
    for fname in files:
        idx += 1

        # 抓图
        image = file_2_pilimage(fname)
        if image is None:
            continue

        # 扩充图片
        imgs = [image]
        if files[fname] != '0': 
            imgs = imgs + [expend_image(image) for i in range(expend_num)]

        for img in imgs:
            size = (100, 100)
            img = change_image(img, {'deform': size})

            # 抽取特征
            vec = collect_basic_image_feature(img, use_hist, size=(40, 40))
            data = np.vstack((data, vec)) if data is not None else vec
            mat = np.array(np.asarray(img.convert('L'), dtype='int'), ndmin=3)
            mats = np.vstack((mats, mat)) if mats is not None else mat
            t = np.array([[int(files[fname]), 1 - int(files[fname])]])
            T = np.vstack((T, t)) if T is not None else t 
            if g_open_debug:
                img.save('output/%s.%d.jpg' % (idx, i))
        print 'finish to process %s' % fname
    if g_open_debug:
        plt_common.show_2d_images(mats[0: 50])
    return data, T 
    
 
def test_collect_image_train_data():
    if True:
        url = 'http://ms.bdimg.com/dsp-image/1371700633.jpg'
        url = 'example.jpg'
        image = file_2_pilimage(url)
        image = change_image(image, {'deform': (30, 50)})
        vec = collect_basic_image_feature(image)
    if False:
        files = {}
        train_image_fpath = 'data/train_data.list.1'
        if os.path.exists('%s.fea' % train_image_fpath):
            data = np.loadtxt('%s.fea' % train_image_fpath, delimiter=",")
            T = np.loadtxt('%s.val' % train_image_fpath, delimiter=",")
        else:
            for line in open(train_image_fpath).readlines():
                line = line[: -1]
                fpath, tag = line.split('\t')
                files[fpath] = tag
            data, T = collect_image_train_data(files, expend_num=2)
            np.savetxt('%s.fea' % train_image_fpath, data, fmt="%d", delimiter=",")
            np.savetxt('%s.val' % train_image_fpath, T, fmt="%d", delimiter=",")
# https://blog.csdn.net/hanging_gardens/article/details/79014160
if __name__ == "__main__":
    image = np.array( # 2行3列的RGB图像
        [
            [[254,   0,   5], [255,   4,   0], [  0,   0,   2], [ 98, 162, 224],],
            [[252,   2,   3], [  1, 255,   0], [  0,   3, 255], [254, 253, 249],],
        ]
        )
    url = 'code.bmp'
    image = file_2_pilimage(url)
    print image.size
    print '1>', np.asarray(image)
    print '2>', np.asarray(image.convert('L')) # r * 0.3 + g * 0.59 + b * 0.11
    image = np.array(image, dtype=float) 
    print '3>', (image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11).astype('int') 
