# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-25 13:47:22
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
import time
from pathlib import Path

import openai

sys.path.append(str(Path.cwd()))
from src.prompt.model_init import callLLM
from src.utils.module import (accept_stream_response, clock, curr_time, curr_weekday,
                              date_after_days, initAllResource, this_sunday)

openai.api_base = "http://10.228.67.99:26921/v1"
openai.api_key = "EMPTY"

support_model_list = [i['id'] for i in openai.Model.list()['data']]
print(f"Support model list: {support_model_list}")

head_str = '''[["'''
prompt = f'''请你扮演一个功能强大的日程管理助手，帮用户提取描述中的日程名称和时间，提取的数据将用于为用户创建日程提醒，下面是一些要求:
1. 日程名称尽量简洁明了并包含用户所描述的事件信息，如果未明确，则默认为`提醒`
2. 事件可能是一个或多个, 每个事件对应一个时间, 请你充分理解用户的意图, 提取每个事件-时间
3. 输出格式: [["事件1", "时间1"], ["事件2", "时间2"]]

# 示例
用户输入: 3分钟后叫我一下,今晚8点提醒我们看联欢晚会, 待会儿看电视
输出: 
[["提醒", "3分钟后"],["看联欢晚会", "今晚8点"],["看电视", None]]
    
用户输入: 十分钟后削个苹果吃, 下午4点我要去健身, 两分钟后提醒我下, 这周六下午三点打羽毛球,下周二早上7点起做早餐,下周5下午3点半接孩子放学,十五分钟后
输出: 
{head_str}'''

def get_currct_time_from_desc(time_desc: str):
    """根据时间描述获取正确的时间
    - Args
        time_desc [str]: 时间的描述 如:今晚8点
    - Return
        target_time [str]: %Y-%m-%d %H:%M:%S格式的时间
    """
    current_time = curr_time() + " " +  curr_weekday()
    prompt = (
            "你是一个功能强大的时间理解及推理工具,可以根据描述和现在的时间推算出正确的时间(%Y-%m-%d %H:%M:%S格式)\n"
            f"现在的时间是: {current_time}\n"
            f"{time_desc}对应的时间是: "
        )
    response = callLLM(prompt, model=model_list[1], temperature=0.7, top_p=0.5, stream=True, stop="\n")
    target_time = accept_stream_response(response, verbose=False)[:19]
    return target_time

start_time = time.time()
model_list = ['Baichuan2-7B-Chat', 'Qwen-14B-Chat', 'Qwen-1_8B-Chat', 'Qwen-72B-Chat', 'Yi-34B-Chat']

response = callLLM(prompt, model=model_list[1], temperature=0.7, top_p=0.8,stop="\n\n",stream=True)
text = head_str + accept_stream_response(response, verbose=False)
text = eval(text)
print(text)

except_result, unexpcept_result = [], []
for item in text:
    event, tdesc = item
    if not (event and tdesc):
        item.append(None)
        unexpcept_result.append(item)
    current_time = get_currct_time_from_desc(tdesc)
    item.append(current_time)
    except_result.append(item)
except_result.sort(key=lambda x: x[2])

