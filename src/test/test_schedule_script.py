# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-05 09:53:59
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


import copy
import json
from re import T

import requests


def dumpJS(obj, ensure_ascii=True):
    return json.dumps(obj, ensure_ascii=ensure_ascii)

def get_response(payload):
    response = session.post(url, headers=headers, data=dumpJS(payload), stream=True)
    for chunk in response.iter_content(decode_unicode=True, chunk_size=10000):
        if len(chunk) < 20:
            continue
        chunk_response = chunk[19:].strip()
        js = json.loads(chunk_response)
        if js['end']:
            for i in js['mid_vars']:
                if i['key'] == 'parse_time_desc':
                    print(f"任务-时间描述: {i['output_text']['except_result']}")
            print(f"回复: {js['message']}")
            print("=="*50)

url = "http://127.0.0.1:6500/chat/complete"

payload = {
    "orgCode": "sf",
    "customId": "test_songhaoyang",
    "intentCode": "schedule_manager",
    "history": [{"msgId": "1908745280","role": "1","content": "3分钟后提醒我喝牛奶"}],
    "backend_history": []
}
headers = {'Content-Type': 'application/json'}
session = requests.Session()

query_list = [
    "帮我创建今天下午3点在新奥集团的项目讨论会",
    "3分钟后提醒我喝牛奶",
    "明天7点叫我起床",
    "明天9点提醒我吃水果",
    "1小时后叫我",
    "半小时后提醒我去上篮球课",
    # "查询明天的日程",
    # "查询今天下午的日程",
    # "今天下午的日程",
    # "今天的日程",
    # "今天上午日程",
    # "明天吃水果的时间改到下午3点",
    # "下午3点的日程改为下午5点",
    # "今天吃水果改为喝牛奶",
    # "下午3点的日程改为下午5点项目组会议",
    # "明天下午5点上课改为今天下午3点",
    # "下午五点的会议取消",
    # "明天的聚餐取消",
    # "今天下午3点的血压测量取消",
    # "今天所有日程取消",
    # "今天所有血压测量提醒取消",
]

for query in query_list:
    input_param = copy.deepcopy(payload)
    input_param["history"][0]['content'] = query
    print(f"Query: {query}")
    result = get_response(input_param)
