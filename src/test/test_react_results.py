# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-11 15:54:04
@desc    :   https://devops.aliyun.com/projex/task/VOSE-3942# 《回复内容格式异常(json格式) - 测试历史会话数据校验》 ReAct生成结果异常校验
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import pickle
import sys
import time

import requests
from tqdm import tqdm

sys.path.append('.')
from pathlib import Path

from src.utils.Logger import logger
from src.utils.module import MysqlConnector, load_yaml

ENV = "test"
save_dir = Path(".cache")


def extract_conv_log():
    """读取ENV环境里的历史会话记录
    """
    conv_log_save_path = save_dir.joinpath("conv_log.pkl")
    # 1. 读取历史会话
    if not conv_log_save_path.exists():
        mysql_config = load_yaml(Path("config","mysql_config.yaml"))[ENV]
        logger.info(f"Connect {ENV} mysql: {mysql_config}")
        mysql_conn = MysqlConnector(**mysql_config)
        data = mysql_conn.query("select * from alg_conversation")
        if data:
            pickle.dump(data, open(conv_log_save_path, "wb"))
    else:
        data = pickle.load(open(conv_log_save_path, "rb"))

    input_text = {}
    for item in tqdm(data):
        item['connect_record'] = json.loads(item['connect_record'])
        for record in item['connect_record']:
            if record.get('role') == '1':
                input_text[record['content']] = 1
    input_text_list = list(input_text.keys())
    return input_text_list

def extract_react_intent_code():
    """从local数据库读取react模式的intentCode
    """
    react_intent_code_data_path = save_dir.joinpath("test_react_intent_code.pkl")
    # 1. 读取历史会话
    if not react_intent_code_data_path.exists():
        mysql_config = load_yaml(Path("config","mysql_config.yaml"))["local"]
        logger.info(f"Connect local mysql: {mysql_config}")
        mysql_conn = MysqlConnector(**mysql_config)
        data = mysql_conn.query("select * from ai_prompt_event")
        if data:
            pickle.dump(data, open(react_intent_code_data_path, "wb"))
    else:
        data = pickle.load(open(react_intent_code_data_path, "rb"))
    react_intent_code_dict = {i['intent_code']: 1 for i in data if i['process_type'] == 'react'}
    return react_intent_code_dict

def req_intent_code(text):
    url = "http://localhost:6500/intent/query"
    intent_payload = {"history": [{"role": 1, "content": text}]}
    tst = time.time()
    r_JS = session.post(url, data=json.dumps(intent_payload)).json()
    cost = time.time() - tst
    return r_JS['items']['intentCode'], round(cost, 2)

def make_payload(text, intentCode):
    payload = {
        "orgCode": "sf",
        "customId": "test_songhaoyang",
        "prompt": "",
        "intentCode": intentCode,
        "history": [{"msgId": "1908745280","role": "1","content": None}],
        "backend_history": []
    }
    payload['history'][0]['content'] = text
    return payload

def make_return_item(**kwds):
    return {"query": kwds['query'], "intentCode": kwds['code'], "response_text": kwds['r_txt']}

def post(text, intentCode):
    payload = make_payload(text, intentCode)
    url = "http://localhost:6500/chat_gen"
    r_txt = session.post(url, data=json.dumps(payload)).text
    ret_item = make_return_item(query=text, code=intentCode, r_txt=r_txt)
    try:
        r_parse = [i.strip() for i in r_txt.split("event: delta\ndata: ") if i]
        r_jsonfy = [json.loads(i) for i in r_parse]
        ret_item["r_jsonfiy"] = r_jsonfy
        reply = [i['message'] for i in r_jsonfy if i['end'] ==  True][0]
        ret_item["reply"] = reply
    except Exception as e:
        ...
    return ret_item

if __name__ == '__main__':
    session = requests.Session()
    result_file_path = save_dir.joinpath("test_react_results.pkl")
    # 提取所有会话历史,将用户说的话作为第一句话传入
    input_text_list = extract_conv_log()
    react_intent_code_dict = extract_react_intent_code()
    result = {}
    for id, text in tqdm(enumerate(input_text_list)):
        intent_code, intent_cost = req_intent_code(text)
        # 筛选react模式的intentCode
        if not react_intent_code_dict.get(intent_code):
            continue
        ret_item = post(text, intent_code)
        result[id] = ret_item
        if id % 1 == 0:
            ...
    ...