# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-05 11:37:39
@desc    :   清空用户当前日程 
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import json

import requests

url = "https://gate-dev.op.laikang.com/aihealthmanager-dev/alg-api/schedule"

customId = "test_songhaoyang"
orgCode = "sf"
payload = json.dumps({
  "orgCode": orgCode,
  "customId": customId,
  "startTime": "2024-01-28 00:00:00",
  "endTime": "2024-02-30 23:59:59"
})
headers = {
  'Content-Type': 'application/json'
}
session = requests.Session()
sch_list = session.post(url + "/query", headers=headers, data=payload).json()['data']


for sch in sch_list:
    payload = {
        "customId": customId,
        "orgCode": orgCode,
        "taskName": sch['taskName'],
        "cronDate": sch['cronDate'],
        "taskType": "reminder",
        "intentCode": "CANCEL"
    }
    headers = {'Content-Type': 'application/json','Cookie': 'acw_tc=0f91360f-a5df-4727-8199-b699c5ec74e9cc029cf1de2cb9c3feb6fd046c056eae'}
    response = session.post(url + "/manage", headers=headers, data=json.dumps(payload)).json()
    print(f"task - {payload['taskName']}, time - {payload['cronDate']}, ops - {payload['intentCode']}, message - {response['message']}, data - {response['data']}")