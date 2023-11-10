# -*- encoding: utf-8 -*-
'''
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import functools
import json
import time
from urllib import parse

import numpy as np
import pandas as pd
import requests
from sqlalchemy import MetaData, Table, create_engine

from utils.Logger import logger


def req_data(url=None, payload=None, headers=None):
    '''request.post()请求数据
    '''
    if not payload:
        payload = {'data': '{"params":[],"returnFieldsType":2,"isFrontShow":false}'}
    res = json.loads(requests.post(url, data=payload, headers=headers).text)
    logger.trace(f'url: {url},\tpayload: {payload}')
    return res

def clock(func):
    """info level 函数计时装饰器"""
    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        if time_cost > 0.0:
            logger.info(func.__name__ + " -> {} ms".format(int(1000*time_cost)))
        return result
    return clocked

def loadJS(path):
    return json.load(open(path, 'r'))

class NpEncoder(json.JSONEncoder):
    """json npencoder, dumps时对np数据格式转为python原生格式
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class MysqlConnector:
    def __init__(self, 
                 user: str=None, 
                 passwd: str=None, 
                 ip: str="0.0.0.0", 
                 port: int=3306, 
                 db_name: str="localhost") -> None:
        """
        user: 用户名
        passwd: 密码
        ip: 目标ip
        port: mysql端口
        db_name: 目标库名
        """
        passwd = parse.quote_plus(passwd)
        self.url = f"mysql+pymysql://{user}:{passwd}@{ip}:{port}/{db_name}"
        self.engine = create_engine(self.url)
        self.metadata = MetaData(self.engine)
        self.connect = self.engine.connect()

    def reconnect(self):
        """
        mysql重连
        """
        self.engine = create_engine(self.url)
        self.metadata = MetaData(self.engine)
        self.connect = self.engine.connect()

    def insert(self, table_name, datas):
        """
        表插入接口
        table_name: 指定表名
        datas: 要插入的数据 形如 [{col1:d1, col2:d2}, {col1:d1, col2:d2}] 字段名和表列名保持一致
        """
        table_obj = Table(table_name, self.metadata, autoload=True)
        try:
            self.connect.execute(table_obj.insert(), datas)
        except Exception as error:
            try:
                self.reconnect()
                self.connect.execute(table_obj.insert(), datas)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        print(table_name, '\tinsert ->\t', len(datas))

    def query(self, sql, orient="records"):
        """
        pd.read_sql_query
        sql: sql查询语句
        orient: 默认"orient" 返回的数据格式 [{col1:d1, col2:d2},{}]
        """
        try:
            # res = self.engine.connect().execute(sql)
            res = pd.read_sql_query(sql, self.connect).to_dict(orient=orient)
        except Exception as error:
            try:
                self.reconnect()
                res = pd.read_sql_query(
                    sql, self.connect).to_dict(orient=orient)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        return res

    def execute(self, sql):
        """
        执行sql语句
        """
        try:
            res = self.connect.execute(sql)
        except Exception as error:
            try:
                self.reconnect()
                res = self.connect.execute(sql)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        return res