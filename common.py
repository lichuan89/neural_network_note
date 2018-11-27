#!/usr/bin/env python
# -*- coding: gbk -*-
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
    str_2_json
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
    json to str
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        return None 


def file_2_str(fpath):
    with open(fpath, 'rb') as f:
        return f.read()


def str_2_file(string, fpath):
    with open(fpath, 'wb') as f:
        f.write(string)
