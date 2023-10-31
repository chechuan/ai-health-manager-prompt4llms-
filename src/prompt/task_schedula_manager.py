# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-27 13:35:10
@desc    :   日程管理pipeline
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from pathlib import Path
from typing import AnyStr, Dict, List

import openai
# import yaml
from langchain.prompts import PromptTemplate

sys.path.append(".")
from config.constrant import (TEMPLATE_TASK_SCHEDULE_MANAGER,
                              task_schedule_return_demo)
from config.constrant_for_task_schedule import (
    task_schedule_parameter_description,
    task_schedule_parameter_description_for_qwen)
from src.prompt.model_init import ChatCompletionRequest, chat_qwen
from src.prompt.qwen_openai_api import create_chat_completion


def call_qwen(messages, functions=None):
    model_name = "Qwen-14B-Chat"
    openai.api_base = "http://10.228.67.99:26926/v1"
    openai.api_key = "empty"
    # print(messages)
    if functions:
        response = openai.ChatCompletion.create(model=model_name, messages=messages, functions=functions)
    else:
        response = openai.ChatCompletion.create(model=model_name, messages=messages)
    print("Assistant:\t", response.choices[0].message.content)
    messages.append(json.loads(json.dumps(response.choices[0].message, ensure_ascii=False)))
    return messages

class taskSchedulaManager:
    def __init__(self):
        """
        用户输入: {input}
        """
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
            print(content)
        ret = chat_qwen(content)
        print(eval(ret))
        return ret

    def tool_ask_for_time(self, messages, msg):
        content = "晚上7点半"
        print(f"tool input: {content}")
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
                print(repr(err))
        task_list = [i['task'] for i in sch_list]
        if sch['task'] not in task_list:
            sch_list.append(sch)
        else:
            idx = task_list.index(sch['task'])
            sch_list[idx] = sch
        return sch_list

    def tool_create_schedule(self, messages, msg):
        content = "改到4点半"
        print(f"tool input: {content}")
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content, 
                "function_call": {"name":msg.function_call['name'],"arguments": content},
                "schedule": self.update_schedule(self.get_init_schedule(), msg.function_call['arguments'])
            }
        )
        return messages

    def get_init_schedule(self):
        return [{"task": "开会", "time": "2023-10-31T17:00:00"}]
    
    def _run(self, query, **kwds):
        """对话过程以messages形式利用历史信息
        :param query: 用户当前输入
        :param messages: 历史信息 包括user/assistant/function_call

        :return messages
        """
        messages = kwds.get("messages", [])
        if query:
            messages.append({"role": "user", "content": query, "schedule": kwds.get("schedule", [])})
        if not query and not messages:
            raise ValueError("Query and messages can't be empty at the same time.")
        while True:
            request = ChatCompletionRequest(model="Qwen-14B-Chat", content=query, functions=task_schedule_parameter_description_for_qwen,messages=messages,)
            msg = create_chat_completion(request).choices[0].message
            if kwds.get("verbose"):
                print(msg.content)
                if msg.function_call:
                    print("call func: ", msg.function_call['name'])
                    print("arguments: ", msg.function_call['arguments'])
            if msg.function_call['name'] == "ask_for_time":
                # conv continue
                content = eval(msg.function_call['arguments'])['ask'] if msg.function_call else msg.content
                # messages.append({"role": msg.role, "content": content})
                messages = self.tool_ask_for_time(messages, msg)
            elif msg.function_call['name'] == "create_schedule":
                # message rollout
                # 本示例中创建任务后跟随修改时间
                messages = self.tool_create_schedule(messages, msg)
            elif msg.function_call['name'] == "modify_schedule":
                # message rollout
                ...
            else:
                messages.append(
                    {
                        "role": "assistant", 
                        "content": msg.content, 
                        "function_call": {"name":msg.function_call['name'],"arguments": input(f"{content}:")}
                    }
                )
            print("messages:")
            [print(i) for i in messages]
            

if __name__ == "__main__":
    from datetime import datetime
    t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    t_claim_str = f"现在的时间是{t}\n"
    schedule = [{"task": "开会", "time": "2023-10-31T17:00:00"}]
    tsm = taskSchedulaManager()
    # tsm.run("我下午开会,提前叫我", verbose=True)
    # debug 任务时间
    tsm._run(f"下午开会,提前叫我", verbose=True, schedule=schedule)
    # tsm._run(f"开会时间改到明天下午3点", verbose=True, schedule=schedule)
    # debug 提醒规则
    # tsm._run(f"现在的时间是{t}\n明天下午3点24开会,提前叫我", verbose=True)
    # debug 创建日程
    # tsm._run(f"明天下午3点40开会", verbose=True)
