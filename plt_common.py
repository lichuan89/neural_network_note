#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def show_images(images, is_show=True):
    fig = plt.figure(figsize=(10, 1), dpi=100)
    for i in range(len(images)):
        ax = fig.add_subplot(1,len(images),i + 1)
        ax.matshow(images[i], cmap='binary') 
        ax.axis('off')
    if is_show:
        plt.show()


def show_2d_images(images, is_show=True):
    cols = 10 
    rows = len(images) / cols + 1
    fig = plt.figure(figsize=(cols, rows), dpi=100)

    for i in range(len(images)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.matshow(images[i], cmap='binary') 
        ax.axis('off')
    if is_show:
        plt.show()


def show_array(arrs, labels, title='show array', is_show=True):
    colors = ['r-', 'k-', 'b-']
    colors = (colors * (len(arrs) / len(colors) + 1))[:len(arrs)]
    linewidths = [3, 2, 0.5]
    linewidths = (linewidths * (len(arrs) / len(linewidths) + 1))[:len(arrs)]
    n = min([len(arr) for arr in arrs])
    min_v = 10000000
    max_v = -100000000
    for arr, label, color, linewidth in zip(arrs, labels, colors, linewidths):  
        min_v = min(min(arr), min_v)
        max_v = max(max(arr), max_v)
        num = len(arr)
        idxs = np.linspace(1, n, num=num)
        plt.plot(idxs, arr, color, linewidth=linewidth, label=label)
    plt.xlabel('array_idx')
    plt.ylabel('array_value', fontsize=15)
    plt.title(title)
    plt.legend()
    plt.axis((0,n,min_v,max_v))
    plt.grid()
    if is_show:
        plt.show()

def show_predict_numbers(y_true, y_pred, is_show=True):
    # Show confusion table
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0,10)
    minor_ticks = [x + 0.5 for x in range(0, 10)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.suptitle('Confusion table', y=1.03, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in xrange(conf_matrix.shape[0]):
        for y in xrange(conf_matrix.shape[1]):
            color = 'w' if x == y else 'k'
            color = 'r'
            ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)
             

    if is_show:
        plt.show()

if __name__ == "__main__": 
    #show_array(arrs=[(12.0, 1.5, 1.0, 0.3), (12.1, 1.6, 0.9, 0.1)], labels=['one', 'two'], title='show array', is_show=True)
    show_predict_numbers(np.array([1, 0, 1, 1]), np.array([1, 1, 1, 1]), is_show=True)
