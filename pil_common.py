#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""


from PIL import Image
import numpy as np
import random
import io
import os
import sys

# https://blog.csdn.net/weixin_41803874/article/details/81201699

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


def test():
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
    arg['scale'] = max(0.7, prob)
    
    idx_tag = {0: 'left,up', 1: 'left,down', 2: 'right,up', 3: 'right,down', 4: 'center'}
    arg['crop'] = idx_tag[int(prob * 10)/2]

    image = change_image(image, arg)
    arg = {'deform': (100, 100)}
    image = change_image(image, arg)
    return image


if __name__ == "__main__":
    #test()
    context = open('test.jpg').read()
    image = Image.open(io.BytesIO(context))
    image = expend_image(image)
    image = image.convert('L')
    mat = np.asarray(image, dtype='int')
    
    image.save('out.jpg')

