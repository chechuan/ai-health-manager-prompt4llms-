# -*- encoding: utf-8 -*-
"""
@Time    :   2024-02-22 17:45:56
@desc    :   XXX
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import json
import re
import sys
from typing import Dict

sys.path.append(".")
from src.prompt.model_init import callLLM


class ChatterGailyAssistant:
    
    def __init__(self) -> None: ...

    def has_chinese_chars(self, data) -> bool:
        text = f"{data}"
        return len(re.findall(r"[\u4e00-\u9fff]+", text)) > 0

    def __parser_function__(self, function: Dict) -> str:
        """
        Text description of function
        """
        tool_desc_template = {
            "zh": "### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}",
            "en": "### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters：{parameters} {args_format}",
        }
        if self.has_chinese_chars(function):
            tool_desc = tool_desc_template["zh"]
        else:
            tool_desc = tool_desc_template["en"]

        name = function.get("name", None)
        name_for_human = function.get("name_for_human", name)
        name_for_model = function.get("name_for_model", name)
        assert name_for_human and name_for_model
        args_format = function.get("args_format", "")
        return tool_desc.format(
            name_for_human=name_for_human,
            name_for_model=name_for_model,
            description_for_model=function["description"],
            parameters=json.dumps(function["parameters"], ensure_ascii=False),
            args_format=args_format,
        ).rstrip()

    def __get_predefined_msg__(self):
        """
        Function Demo:
        ```json
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description":
                            "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        },
                    },
                    "required": ["location"],
                },
            }
        ```
        """
        self.prompt_react = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""
        functions = [
            {
                "name": "查询知识库",
                "description": "知识查询工具可以通过查询指定外部知识库获取query相关专业知识, 当问题需要专业的知识时, 优先使用该工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge_base_name": {
                            "type": "string",
                            "description": "知识库名称",
                            "options": ["新奥百科知识库", "健康知识", "高血压"],
                        }
                    },
                    "required": ["knowledge_base_name"],
                },
            },
            {
                "name": "搜索引擎",
                "description": "搜索引擎可以快速找到相关的网页, 查询到相关的知识内容, 当问题需要知识且知识库不匹配时, 优先使用该工具",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "搜索的关键字"}},
                    "required": ["query"],
                },
            },
            {
                "name": "日常闲聊",
                "description": "日常闲聊工具可以进行简单的聊天",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string", "description": "话题"}},
                    "required": ["topic"],
                },
            },
        ]
        tool_descs = "\n\n".join(self.__parser_function__(function) for function in functions)
        tool_names = ",".join(function["name"] for function in functions)

        return {"tool_descs": tool_descs, "tool_names": tool_names}

    def get_next_step(self, history: Dict):
        """通过ReAct选择下一步的策略"""
        def compose_prompt_react(history):
            messages = []
            for turn_idx in range(len(history)):
                msg = history[turn_idx]
                if turn_idx == 0:
                    messages.append(f"Question: {msg['content']}")

        tool_msg = self.__get_predefined_msg__()
        tool_desc, tool_names = tool_msg["tool_descs"], tool_msg["tool_names"]
        system_prompt = self.prompt_react.format(tool_descs=tool_desc, tool_names=tool_names)
        messages = [{"role": "system", "content": system_prompt}]
        messages 
        
        
        return prompt_react

    def run(self, *args, **kwargs): ...
