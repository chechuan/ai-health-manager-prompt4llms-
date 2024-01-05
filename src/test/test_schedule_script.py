# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-05 09:53:59
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


import copy
import json
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
import pandas as pd
import requests

from src.utils.Logger import logger
from src.utils.module import clock


def dumpJS(obj, ensure_ascii=True):
    return json.dumps(obj, ensure_ascii=ensure_ascii)

@clock
def get_response(payload):
    response = session.post(url, headers=headers, data=dumpJS(payload), stream=True)
    time_desc, reply = [], ""
    for chunk in response.iter_content(decode_unicode=True, chunk_size=10000):
        if len(chunk) < 20:
            continue
        chunk_response = chunk[19:].strip()
        js = json.loads(chunk_response)
        if js['end']:
            for i in js['mid_vars']:
                if i['key'] == 'parse_time_desc':
                    time_desc = i['output_text']['except_result']
                # if i['key'] == 'confirm_query_time_range':
                #     print(f"查询时间范围: {i['output_text']}")
            reply = js['message']
    df.loc[index, 'reply'] = reply
    if time_desc:
        df.loc[index, 'ret_date'] = time_desc[0][2][:10]
        df.loc[index, 'ret_time'] = time_desc[0][2][11:]
    if time_desc:
        payload = {
            "customId": payload['customId'],
            "orgCode": payload['orgCode'],
            "taskName": time_desc[0][0],
            "cronDate": time_desc[0][2],
            "taskType": "reminder",
            "intentCode": "CANCEL"
        }
        response = session.post(ai_backedn_url + "/manage", headers=headers, data=json.dumps(payload)).json()
    return reply, time_desc

ai_backedn_url = "https://gate-qa.op.laikang.com/aihealthmanager-alg/alg-api/schedule"
url = "http://127.0.0.1:6500/chat/complete"

payload = {
    "orgCode": "sf",
    "customId": "test_songhaoyang",
    "intentCode": "schedule_manager",
    "history": [{"msgId": "1908745280","role": "1","content": "3分钟后提醒我喝牛奶"}],
    "backend_history": [],
    "debug": True
}
headers = {'Content-Type': 'application/json'}
session = requests.Session()

# 创建测试
file_path = Path(".cache/日程管理测试集v2-20240105.csv")
df = pd.read_csv(file_path, encoding='gbk')
print(f"CSV Columns: {list(df.columns)}")

# pool = ThreadPoolExecutor(max_workers=10)
for index, row in df.iterrows():
    query = row['测试用例']
    input_param = copy.deepcopy(payload)
    input_param["history"][0]['content'] = query
    if isinstance(row['reply'], str):
        continue
    # future1 = pool.submit(get_response, input_param)
    reply, time_desc = get_response(input_param)
    # reply, time_desc = future1.result()
    logger.debug(f"Index:{index} {query} -> {reply}\n{time_desc}")
    if index % 50 == 0:
        df.to_csv(file_path, encoding='gbk', index=False)
        logger.info(f"save file index up to {index}.")

# print(df.head())
# df.to_csv(file_path, encoding='gbk', index=False)