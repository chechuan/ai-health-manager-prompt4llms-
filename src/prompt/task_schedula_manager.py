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
from typing import Dict

import openai
# import yaml
from langchain.prompts import PromptTemplate

sys.path.append(".")
from config.constrant import (TEMPLATE_TASK_SCHEDULE_MANAGER,
                              task_schedule_parameter_description,
                              task_schedule_parameter_description_for_qwen,
                              task_schedule_return_demo)
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
    
    def tool_create_plan(self,):
        ...
    
    def tool_occur_time_confirm(self,):
        ...

    def tool_remind_rule_confirm(self,):
        ...

    def tool_cancel_plan(self,):
        ...

    def tool_change_plan(self, ):
        ...

    def tool_search_plan(self, ):
        ...

    def run_tool(self, *args, **kwds):
        ...
    
    def _run(self, query, **kwds):
        # messages = [{"role": "user", "content": query}]
        # while True:
        #     messages = call_qwen(messages, task_schedule_parameter_description_for_qwen)
        #     if messages[-1].get("function_call"):
        #         args = messages[-1]["function_call"]
        #         print(f"Function args: {args}")
        #         input_text = input("Function Called: ")
        #         if input_text.lower() == "stop":
        #             break
        #         messages.append({"role": "function","name": messages[-1]["function_call"]["name"],"content": input_text})
        #     else:
        #         # if_continue = input("Conversation is finished, new one?[y]/[n]:")
        #         # if if_continue.lower() == "n":
        #         #     break
        #         # else:
        #         input_text = input("User: ")
        #         messages.append({"role": "function","name": "askForRemindTime","content": input_text})
            # messages.append([{"role": "user", "content": input_text}])
        messages = [{"role": "user","content": query}]
        while True:
            request = ChatCompletionRequest(
                model="Qwen-14B-Chat", 
                content=query, 
                functions=task_schedule_parameter_description_for_qwen,
                messages=messages
            )
            response = create_chat_completion(request)
            msg = response.choices[0].message
            if kwds.get("verbose"):
                print(msg)
            content = eval(msg.function_call['arguments'])['ask'] if msg.function_call else msg.content
            messages.append({"role": msg.role, "content": content})
            messages.append({"role": "user", "content": input("Tool output:")})

if __name__ == "__main__":
    tm = taskSchedulaManager()
    # tm.run("我下午开会,提前叫我", verbose=True)
    t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    # 
    # debug 任务时间
    # tm._run(f"现在的时间是{t}\n\n我明天开会,提前叫我", verbose=True)
    # debug 提醒规则
    # tm._run(f"现在的时间是{t}\n明天下午3点24开会,提前叫我", verbose=True)
    # debug 创建日程
    tm._run(f"现在的时间是{t}\n明天下午3点40开会", verbose=True)
