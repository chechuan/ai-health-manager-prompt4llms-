# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-27 13:35:10
@desc    :   日程管理pipeline
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import openai
import yaml
from langchain.prompts import PromptTemplate

sys.path.append(".")
from config.constrant import (TEMPLATE_TASK_SCHEDULE_MANAGER,
                              task_schedule_parameter_description,
                              task_schedule_return_demo)


class taskSchedulaManager:
    api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
    openai.api_base = api_config['llm']
    openai.api_key = "EMPTY"
    def __init__(self):
        """
        用户输入: {input}
        你的输出(json):
        """
        prompt = PromptTemplate(
            input_variables=["task_schedule_return_demo", "task_schedule_parameter_description", "tmp_time"], 
            template=TEMPLATE_TASK_SCHEDULE_MANAGER
        )
        self.sys_prompt = prompt.format(
            task_schedule_return_demo=task_schedule_return_demo,
            task_schedule_parameter_description=task_schedule_parameter_description,
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
        completion = openai.ChatCompletion.create(
            model="Qwen-14B-Chat",
            messages=[
                {"role": "user","content": content}
            ],
            top_k=0, top_p=0.8, repetition_penalty=1.1
        )
        ret = completion['choices'][0]['message']['content'].strip()
        return ret

if __name__ == "__main__":
    tm = taskSchedulaManager()
    tm.run("今晚准备7点开始做饭，提前15分钟提醒我", verbose=True)