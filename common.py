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


g_open_log = True 
g_current_time = None

def log(*arg):
    """
    write_log
    """
    global g_open_log
    global g_current_time
    if not g_open_log:
        return
    delt_time = time.time() - g_current_time if g_current_time is not None else 0
    g_current_time = time.time()
    prefix = '[log] [%s] [%.3f]' % (time.strftime("%Y-%m-%d %H:%M:%S"), delt_time)
    coding = 'utf8'
    if type(arg) == type((1, 2)):
        arg = [v if v is not None else str(None) for v in arg]
        arg = [arg if type(arg) not in set([type([]), type({})]) else json_2_str(arg)]
        li = [elem.encode(coding, 'ignore') for elem in arg]
        print >> sys.stderr, prefix, ' '.join(li)
    else:
        if arg is None:
            arg = str(None)
        print >> sys.stderr, prefix, arg.encode(coding, 'ignore')


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


def file_2_str(fpath, decode=None):
    """
    从文件读取字符串
    """
    with open(fpath, 'rb') as f:
        context = f.read()
        if decode is not None:
            context = context.decode(decode, 'ignore')
        return context

def str_2_file(string, fpath, encode=None):
    """
    将字符串写入文件
    """
    with open(fpath, 'wb') as f:
        if encode is not None:
            string = string.encode(encode, 'ignore')
        f.write(string)


def clear_dir(path):
    """
    创建文件夹,或者删除文件夹下的直接子文件
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        arr = os.listdir(path)
        for v in arr:
            f = '%s/%s' % (path, v) 
            if os.path.isfile(f):
                os.remove(f)

def read_dir(path, decode='utf8'):
    arr = os.listdir(path)
    output = []
    for v in arr:
        f = '%s/%s' % (path, v)
        if os.path.isfile(f):
            context = file_2_str(f)
            if decode is not None:
                context = context.decode(decode, 'ignore')
            output.append(context)
    return output 


def muti_process_stdin(worker, args, batch_line_num, thread_running_num, use_share_path=None):
    """
    多进程处理标准输入流, 批处理，并输出到标准输出流
    每一批处理batch_line_num行数据，每一批开启thread_running_num个线程 
    worker格式如:
    def worker(lines, args): return ['%s:%s' % (args[0], line) for line in lines]
    """
    idx = 0 
    batch = []
    for line in sys.stdin:
        line = line[: -1].decode('utf8', 'ignore')
        batch.append(line)
        if len(batch) >= batch_line_num:
            output_lines = muti_process(batch, thread_running_num, worker, args, use_share_path)
            print '\n'.join(output_lines).encode('utf8', 'ignore')
            sys.stdout.flush()
            batch = []
    if batch != []:
        output_lines = muti_process(batch, thread_running_num, worker, args, use_share_path)
        print '\n'.join(output_lines).encode('utf8', 'ignore')


def muti_process(lines, thread_running_num, worker, args, use_share_path=None): 
    """ 
    多进程处理数据, 并输出 
    """
    if use_share_path is None:
        # 使用共享内存
        manager = multiprocessing.Manager()
        contexts = manager.dict()
    else:
        # 先并发输入文件,再统一搜集输出
        clear_dir(use_share_path)
        contexts = use_share_path 
    threadpool = []
    batch_arr = {}
    for i in range(len(lines)):
        k = i % thread_running_num 
        batch_arr.setdefault(k, [])
        batch_arr[k].append(lines[i])
    
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
    if use_share_path is not None:
        lines = read_dir(use_share_path)
        clear_dir(use_share_path)
    else: 
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
        if type(self.share) == unicode or type(self.share) == type(''):
            # 用临时文件存储结果
            str_2_file('\n'.join(result).encode('utf8', 'ignore'), '%s/tmp.%s' % (self.share, id))
        else:
            self.share.setdefault(id, result)

def test():
    if False:
        clear_dir('tmp.muti_process')
        str_2_file('abcd\nefg', 'tmp.muti_process/1')
        str_2_file('abcd\nefg', 'tmp.muti_process/2')
        print read_dir('tmp.muti_process')
        clear_dir('tmp.muti_process')
    if True: 
        def worker(lines, args): return ['%s:%d:%s' % (args[0], os.getpid(), line) for line in lines]
        muti_process_stdin(worker, ['prefix'], batch_line_num=30, thread_running_num=7, use_share_path='tmp.muti_process/')


if __name__ == "__main__":
    test()
