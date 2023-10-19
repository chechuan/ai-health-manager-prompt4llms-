# -*- encoding: utf-8 -*-
'''
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import time
import pymysql
import requests
import functools
import traceback
import numpy as np
from utils.Logger import logger
from dbutils.pooled_db import PooledDB
from py2neo import Graph

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

class MysqlDbPool():
    """
    PooledDB() 参数含义
    - creator：使用链接数据库的模块
    - maxconnections：连接池允许的最大连接数，0和None表示没有限制
    - mincached：初始化时，连接池至少创建的空闲的连接，0表示不创建
    - maxcached：连接池空闲的最多连接数，0和None表示没有限制
    - maxshared：连接池中最多共享的连接数量，0和None表示全部共享，ps:其实并没有什么用，因为pymsql和MySQLDB等模块中的threadsafety都为1，所有值无论设置多少，_maxcahed永远为0，所以永远是所有链接共享
    - blocking：链接池中如果没有可用共享连接后，是否阻塞等待，True表示等待，False表示不等待然后报错
    - setsession：开始会话前执行的命令列表
    - ping：ping Mysql 服务端，检查服务是否可用

    ================================================================
    ```python
    cfg = {
        'ip': '',
        'user': '',
        'passwd': '',
        'port': 3306,
        'db_name': ''
    }
    pool = MysqlDbPool(cfg)
    sql = "select * from table;"
    data = pool.execute(sql)
    ```
    """

    def __init__(self, cfg) -> None:
        self.mysql_cfg = cfg
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=15,
            mincached=0,
            maxcached=20,
            maxshared=0,
            blocking=True,
            setsession=[],
            ping=5,
            host=self.mysql_cfg['ip'],
            port=self.mysql_cfg['port'],
            user=self.mysql_cfg['user'],
            password=self.mysql_cfg['passwd'],
            database=self.mysql_cfg['db_name'],
            charset='utf8mb4'
        )

    def execute(self, sql):
        # 使用连接池管理，每次获取连接，创建cursor
        # 执行sql, update/insert需要commit, select需要fetchall
        # 执行完毕后关闭连接，关闭游标
        res = None
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            res = cursor.fetchall()
        except Exception as error:
            logger.error("sql执行异常, msg=%s" % (repr(error)))
            logger.error(traceback.format_exc())
        finally:
            cursor.close()
            conn.close()
        return res
    
class GBConnector:
    def __init__(self, url, auth=("neo4j", "neo4j")):
        """
        :param url[Str]: 要创建的连接url 例 "http://0.0.0.0:7474"
        :param auth[Tuple]: (user, passwd)
        """
        self.url = url
        if auth:
            self.graph = Graph(url, auth=auth)
        else:
            self.graph = Graph(url)

    @clock
    def run_cypher(self, cypher):
        """
        :param cypher[Str]: cypher查询语句

        ======================
        
        example
        ```python
        db = GBConnector(url="http://0.0.0.0:7474", auth=None)
        data = db.run_cypher("MATCH (n) RETURN n.name as name, n.code as code")
        ```
        """
        data = None
        try:
            data = self.graph.run(cypher).data()
        except Exception as e:
            try:
                self.graph = Graph(self.url)
                data = self.graph.run(data).data()
            except Exception as e:
                logger.error(repr(e))
                logger.error(traceback.format_exc())
        finally:
            if data:
                return data
            else:
                logger.error("查询失败.")
                return None