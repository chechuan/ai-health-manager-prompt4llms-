# -*- encoding: utf-8 -*-
'''
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import functools
import json
import sys
import time
from typing import Tuple
from urllib import parse

import numpy as np
import pandas as pd
import requests
from sqlalchemy import MetaData, Table, create_engine

from src.utils.Logger import logger


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical(exc_value.message, exc_info=(exc_type, exc_value, exc_traceback))


def handle_error(func):
    def __inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            handle_exception(exc_type, exc_value, exc_tb)
        finally:
            print(e.message)
    return __inner

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

def get_intent(text):
    """通过关键词解析意图->code
    """
    if '创建提醒' in text:
        code = 'create_alert'
    elif '饮食' in text and '咨询' in text:
        code = 'food'
    elif '菜谱' in text:
        code = 'recipe_consult'
    elif '音乐' in text:
        code = 'play_music'
    elif '天气' in text:
        code = 'check_weather'
    elif '辅助诊断' in text:
        code = 'auxiliary_diagnosis'
    elif '医师' in text:
        code = 'call_doctor'
    elif '运动师' in text:
        code = 'call_sportMaster'
    elif '心理' in text:
        code = 'call_psychologist'
    elif '修改提醒' in text:
        code = 'change_alert'
    elif '取消提醒' in text:
        code = 'cancel_alert'
    elif '营养师' in text:
        code = 'call_dietista'
    elif '健管师' in text:
        code = 'call_health_manager'
    elif '其它意图' in text:
        code = 'other'
    elif '日程管理'in text:
        code = 'schedule_manager'
    elif '网络' in text:
        code = 'search_network'
    elif '首都' in text:
        code = 'capital'
    #elif '彩票' in text:
        #code = 'call_dietista'
    #elif '营养师' in text:
        #code = 'call_dietista'
    #elif '营养师' in text:
    #    code = 'call_dietista'
    #elif '营养师' in text:
    #    code = 'call_dietista'
    #elif '营养师' in text:
    #    code = 'call_dietista'
    #elif '营养师' in text:
    #    code = 'call_dietista'
    else:
        code = 'other'
    logger.debug(f'识别出的意图:{text} code:{code}')
    return code

def get_doc_role(code):
    if code == 'call_dietista':
        return 'ROLE_NUTRITIONIST'
    elif code == 'call_sportMaster':
        return 'ROLE_EXERCISE_SPECIALIST'
    elif code == 'call_psychologist':
        return 'ROLE_EMOTIONAL_COUNSELOR'
    elif code == 'call_doctor':
        return 'ROLE_DOCTOR'
    elif code == 'call_health_manager':
        return 'ROLE_HEALTH_SPECIALIST'
    else:
        return 'ROLE_HEALTH_SPECIALIST'

def _parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    h = text.find('Thought:')
    i = text.find('\nAction:')
    j = text.find('\nAction Input:')
    k = text.find('\nObservation:')
    l = text.find('\nFinal Answer:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_thought = text[h + len('Thought:'):i].strip()
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_thought, plugin_name, plugin_args
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len('Thought:'):l].strip()
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            plugin_args.split("\n")[0]
            return plugin_thought, "直接回复用户问题", plugin_args
        else:
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            return "I know the final answer.", "直接回复用户问题", plugin_args
    return '', ''

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
