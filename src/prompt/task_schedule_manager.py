# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-27 13:35:10
@desc    :   日程管理pipeline
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import AnyStr, Dict, List

import openai
import requests
import yaml
from langchain.prompts import PromptTemplate

sys.path.append(".")
from config.constrant import (TEMPLATE_TASK_SCHEDULE_MANAGER,
                              task_schedule_return_demo)
from config.constrant_for_task_schedule import (
    task_schedule_parameter_description,
    task_schedule_parameter_description_for_qwen)
from src.prompt.model_init import ChatCompletionRequest, chat_qwen
from src.prompt.qwen_openai_api import create_chat_completion
from utils.Logger import logger


def call_qwen(messages, functions=None):
    model_name = "Qwen-14B-Chat"
    openai.api_base = "http://10.228.67.99:26926/v1"
    openai.api_key = "empty"
    # logger.debug(messages)
    if functions:
        response = openai.ChatCompletion.create(model=model_name, messages=messages, functions=functions)
    else:
        response = openai.ChatCompletion.create(model=model_name, messages=messages)
    logger.debug("Assistant:\t", response.choices[0].message.content)
    messages.append(json.loads(json.dumps(response.choices[0].message, ensure_ascii=False)))
    return messages

class taskSchedulaManager:
    def __init__(self, 
                 api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']):
        """
        用户输入: {input}
        """
        self.api_config = api_config
        prompt = PromptTemplate(
            input_variables=["task_schedule_return_demo", "task_schedule_parameter_description", "curr_plan", "tmp_time"], 
            template=TEMPLATE_TASK_SCHEDULE_MANAGER
        )
        self.sys_prompt = prompt.format(
            task_schedule_return_demo=task_schedule_return_demo,
            task_schedule_parameter_description=task_schedule_parameter_description,
            curr_plan=[],
            tmp_time=datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
            )
        self.conv_prompt = PromptTemplate(
            input_variables=["input"],
            template="用户输入: {input}\n你的输出(json):"
            )

    
    def run(self, query, **kwds):
        content = self.sys_prompt + self.conv_prompt.format(input=query)
        if kwds.get("verbose"):
            logger.debug(content)
        ret = chat_qwen(content)
        logger.debug(eval(ret))
        return ret

    def tool_ask_for_time(self, messages, msg):
        content = input(f"tool input(晚上7点半): ")
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content, 
                "function_call": {"name":msg.function_call['name'],"arguments":  content},
                "schedule": self.get_init_schedule()
            }
        )
        return messages
    
    def update_schedule(self, sch_list: List[Dict], sch: Dict or AnyStr):
        """更新日程"""
        if isinstance(sch, dict):
            ...
        elif isinstance(sch, str):
            try:
                sch = eval(sch)
            except Exception as err:
                logger.debug(repr(err))
        task_list = [i['task'] for i in sch_list]
        if sch['task'] not in task_list:
            sch_list.append(sch)
        else:
            idx = task_list.index(sch['task'])
            sch_list[idx] = sch
        return sch_list

    def tool_create_schedule(self, messages, msg):
        content = input(f"tool input(改到4点半): ")
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content, 
                "function_call": {"name":msg.function_call['name'],"arguments": content},
                "schedule": self.update_schedule(self.get_init_schedule(), msg.function_call['arguments'])
            }
        )
        return messages
    
    def tool_modify_schedule(self, messages, msg):
        content = input(f"tool input(算了，取消吧): ")
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content, 
                "function_call": {"name":msg.function_call['name'],"arguments": content},
                "schedule": self.update_schedule(self.get_init_schedule(), msg.function_call['arguments'])
            }
        )
        return messages

    def get_real_time_schedule(self, **kwds):
        """查询用户实时日程
        """
        assert kwds.get("orgCode"), KeyError("orgCode is required")
        assert kwds.get("customId"), KeyError("customId is required")

        url = self.api_config["ai_backend"] + "/alg-api/schedule/query"
        payload = {
            "orgCode": "sf",
            "customId": "007"
        }
        headers = {"content-type": "application/json"}
        response = requests.request("POST", url, json=payload, headers=headers).text
        resp_js = json.loads(response)
        if resp_js['code'] == 200:
            data = resp_js['data']
            return [{"task": i['taskName'], "time": i['cronDate']} for i in data]
        else:
            return [{"task": "开会", "time": "2023-10-31 17:00:00"}]
    
    def _run(self, messages: List[Dict], **kwds):
        """对话过程以messages形式利用历史信息
        - Args
            messages (List[Dict])
                历史信息 包括user/assistant/function_call
        
        - return
            output (str) 
                直接输出的文本
        """
        schedule = self.get_real_time_schedule(**kwds)
        
        if len(messages) == 1:
            logger.debug(f"Init user input: {messages[0]['content']}")
        request = ChatCompletionRequest(model="Qwen-14B-Chat", 
                                        functions=task_schedule_parameter_description_for_qwen,
                                        messages=messages,)
        msg = create_chat_completion(request, schedule).choices[0].message
        if kwds.get("verbose"):
            logger.debug(msg.content[msg.content.rfind("Thought:"):])
            if msg.function_call:
                logger.debug("call func: ", msg.function_call['name'])
                logger.debug("arguments: ", msg.function_call['arguments'])
        if msg.function_call['name'] == "ask_for_time":
            # conv continue
            content = eval(msg.function_call['arguments'])['ask'] if msg.function_call else msg.content
            messages = self.tool_ask_for_time(messages, msg)
        elif msg.function_call['name'] == "create_schedule":
            # message rollout
            # 本示例中创建任务后跟随修改时间
            # messages = self.tool_create_schedule(messages, msg)
            content = eval(msg.function_call['arguments'])['ask']
        elif msg.function_call['name'] == "modify_schedule":
            # message rollout
            # messages = self.tool_modify_schedule(messages, msg)
            content = eval(msg.function_call['arguments'])['ask']
        elif msg.function_call['name'] == "cancel_schedule":
            content = "已为您取消该日程"
        return content

if __name__ == "__main__":
    t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    tsm = taskSchedulaManager()
    t_claim_str = f"现在的时间是{t}\n"
    schedule = [{"task": "开会", "time": "2023-11-03T11:40:00"}]
    # debug 任务时间
    # content = f"{t_claim_str}下午开会,提前叫我"
    # debug 直接取消
    # content = f"{t_claim_str}5分钟后的日程取消"
    # debug 提醒规则
    # content = f"现在的时间是{t}\n明天下午3点24开会,提前叫我"
    # debug 创建日程
    # content = f"明天下午3点40开会"
    content = "开会时间改到明天下午4点"
    history = [{"role":"user", "content": content}]
    while True:
        content = tsm._run(history, schedule=schedule, verbose=True, orgCode="sf", customId="007")
        history.append({"role":"assistant", "content": content})
        history.append({"role":"user", "content": input(f"{content}: ")})
    # tsm._run(f"开会时间改到明天下午3点", verbose=True, schedule=schedule)
    
