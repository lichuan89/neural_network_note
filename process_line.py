#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
    @author lichuan01@baidu.com
    @date   2016/09/12  
    @note   
"""

import traceback
import base64
import re
import sys
import time 
import datetime
import json


def format_image(tag=[5, 150]):
    """  
    mapper_prod_stream
    """
    col_num, width = int(tag[0]), int(tag[1])
    lines = {}   
    j = 0  
    for line in sys.stdin:
        line = line[:-1]
        image, name = line.split('\t')

        lines.setdefault(j, [])  
        if len(lines[j]) >= col_num:
            j += 1 
        lines.setdefault(j, [])  
        lines[j].append((image, name))
    print '''  
        <style type="text/css">
        table td{ height:10px; width:%d; font-size:5px}
        </style>
        ''' % width
    print '<table border="0.1" border-spacing:0px  cellspacing="0" cellpadding="0">'
    for i in range(0, j + 1):  
        arr = lines[i]
        th = ['<th><img src="%s" width=%d style="border:1px solid #ff0000" /></th>' % (v[0], width) for v in arr]   
        td = ['<td>%s</td>' % v[1] for v in arr]   
        print '<tr>%s</tr> <tr>%s</tr>\n' % (''.join(th), ''.join(td))
    print '</table>'


if __name__ == "__main__":
    func_arg = sys.argv[1]
    arr = func_arg.split('____')
    if len(arr) == 1:
        func = arr[0]
        eval(func)()
    else:   
        func = arr[0]
        arg = arr[1:] 
        eval(func)(arg)  
