from asyncio import Task
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def task(i):
    print(f"task {i} start")
    url = "http://0.0.0.0:6500/aigc/functions"
    payload = '{\r\n    "intentCode": "aigc_functions_consultation_summary",\r\n    "user_profile": {\r\n        "age": 35,\r\n        "gender": "male",\r\n        "height": 185,\r\n        "weight": 70,\r\n        "disease_history": ["高血压", "肥胖"],\r\n        "allergic_history": ["花生", "鳗鱼"],\r\n        "surgery_history": ["无"]\r\n    },\r\n    "messages": [\r\n        {"role": "user", "content": "我肚子疼"},\r\n        {"role": "assistant", "content": "你的腹痛是突然出现的还是慢慢发生的？"},\r\n        {"role": "user", "content": "突然出现的"},\r\n        {"role": "assistant", "content": "你能描述一下腹痛的部位吗，是在上腹部、下腹部还是两侧？"},\r\n        {"role": "user", "content": "下腹痛"},\r\n        {"role": "assistant", "content": "这种疼痛是一阵一阵的还是持续不断的？疼痛的程度如何，能否忍受？"},\r\n        {"role": "user", "content": "一阵一阵的，还可以忍受"},\r\n        {"role": "assistant", "content": "你的腹痛是刺痛、钝痛还是绞痛？同时有没有伴随其他症状，比如发热、恶心、呕吐、腹泻或者便秘？"},\r\n        {"role": "user", "content": "绞痛吧，有点恶心"},\r\n        {"role": "assistant", "content": "你的恶心感是与腹痛同时出现的吗？你有拉肚子或者便秘的情况吗？饮食和排便习惯近期有没有改变？"},\r\n        {"role": "user", "content": "同时出现，没有别的症状"},\r\n        {"role": "assistant", "content": "你近期有没有做过剧烈运动或者饮食上吃了不易消化的食物？这种腹痛是在月经期间更容易发生吗？"},\r\n        {"role": "user", "content": "同时出现，没有别的症状"}\r\n    ]\r\n}'
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(f"task {i} done")


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=1) as executor:
        task_list = []
        for i in range(10):
            task_list.append(executor.submit(task, i))
