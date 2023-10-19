# -*- encoding: utf-8 -*-
'''
@Time    :   2022-12-20 10:26:48
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
from flask_apscheduler.auth import HTTPBasicAuth

class schConfig:
    # 设置时区，时区不一致会导致定时任务的时间错误
    SCHEDULER_TIMEZONE = 'Asia/Shanghai'
    # 一定要开启API功能，这样才可以用api的方式去查看和修改定时任务
    SCHEDULER_API_ENABLED = True
    # api前缀（默认是/scheduler）
    SCHEDULER_API_PREFIX = '/scheduler'
    # auth验证。默认是关闭的，
    SCHEDULER_AUTH = HTTPBasicAuth()