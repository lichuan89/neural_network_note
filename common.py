#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note
"""
import os 
import sys
import struct
import hashlib
import logging
import json 
import datetime
import re
import fileinput
import urllib
import urllib2
import threading
import urlparse
import time 
import multiprocessing
import fcntl 


def str_2_json(uni):
    """
    unicode字符串转json 
    """
    obj = {}
    try:
        obj = json.loads(uni)
    except Exception as e:
        print >> sys.stderr, e
        return None
    return obj


def json_2_str(obj):
    """
    json转unicode字符串
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        return None 


def file_2_str(fpath):
    """
    从文件读取字符串
    """
    with open(fpath, 'rb') as f:
        return f.read()


def str_2_file(string, fpath):
    """
    将字符串写入文件
    """
    with open(fpath, 'wb') as f:
        f.write(string)


def muti_process_stdin(worker, args, batch_line_num, thread_running_num):
    """
    多进程处理标准输入流, 批处理，并输出到标准输出流
    每一批开启的进程数:thread_running_num, 单进程处理行数:batch_line_num
    worker格式如:
    def worker(lines, args): return ['%s:%s' % (args[0], line) for line in lines]
    """

    idx = 0 
    batch_arr = {}
    for line in sys.stdin:
        line = line[: -1].decode('utf8', 'ignore')
        if idx not in batch_arr:
            batch_arr[idx] = []
        if len(batch_arr[idx]) >= batch_line_num:
            idx += 1
            batch_arr[idx] = []
        batch_arr[idx].append(line)
        if idx >= thread_running_num - 1 and len(batch_arr[idx]) >= batch_line_num:
            lines = muti_process(batch_arr, worker, args)
            print '\n'.join(lines).encode('utf8', 'ignore')
            sys.stdout.flush()
            idx = 0 
            batch_arr = {}
    if batch_arr != {}:
        muti_process(batch_arr, worker, args)


def muti_process(batch_arr, worker, args): 
    """ 
    多进程处理数据, 并输出 
    """
    manager = multiprocessing.Manager()
    contexts = manager.dict()
    threadpool = []
    for idx in batch_arr:
        th = Processor(worker, batch_arr[idx], args + [contexts])
        threadpool.append(th)
    idx = 0 
    threads = []
    for th in threadpool:
        th.start()
        
    for th in threadpool:
        th.join()

    lines = []
    for k, v in contexts.items(): 
        for line in v:
            lines.append(line)
    return lines


class Processor(multiprocessing.Process):
    """
    processor
    """
    def __init__(self, worker, lines, args):
        multiprocessing.Process.__init__(self)
        self.worker = worker
        self.lines = lines
        self.args = args[: -1]
        self.share = args[-1]
    
    def run(self):
        """ 
        run 
        """
        result = self.worker(self.lines, self.args)
        id = os.getpid() 
        self.share.setdefault(id, result)


def test():
    def worker(lines, args): return ['%s:%s' % (args[0], line) for line in lines]
    muti_process_stdin(worker, ['prefix'], batch_line_num=2, thread_running_num=2)


if __name__ == "__main__":
    test()
