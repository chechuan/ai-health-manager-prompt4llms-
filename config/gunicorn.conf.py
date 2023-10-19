# -*- encoding: utf-8 -*-
'''
@Time    :   2022-11-18 14:11:55
@Author  :   songhaoyang
@Version :   1.0
@Contact :   songhaoyanga@enn.cn
'''


workers = 4
daemon = 'false'
worker_class = "gevent"
worker_connections = 2000
bind = "0.0.0.0:9900"
