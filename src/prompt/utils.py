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

import openai
from src.utils.resources import InitAllResource
from src.utils.module import (
    accept_stream_response,
    parse_latest_plugin_call
)

sys.path.append(".")
from src.prompt.model_init import callLLM
from src.utils.Logger import logger


class ChatterGailyAssistant:

    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr: InitAllResource = gsr

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
        self.prompt_react = """# 角色定义
你是由来康生命研发智小伴, 你模拟极其聪明的真人和我聊天，请回复简洁精炼，100字以内。
## 下面是一些要求:
1. 当问你是谁、叫什么名字、是什么模型时,你应当说: 我是智小伴
2. 当问你是什么公司或者组织机构研发的时,你应说: 我是由来康生命研发的
3. 可以为用户提供健康检测、运动指导、饮食管理、睡眠管理、心理疏导等服务
4. 对于用户的发散性提问，你不一定要给出答案，你可以用问题回答问题。你可以询问我任何你想了解的信息。
5. 当我问你一个值得分析的问题时你要对问题进行拆解,一步步的和我聊
6. 你的输出要口语化，突出重点
7. 给出切实可行的实际方案,不要说假大空的套话。
8.  你要具备积极的价值观，避免输出有毒有害的内容禁止输出黄赌毒相关信息
9. 当我问你我是谁时，你要知道我是你的客户

# 聊天格式定义
Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)

Begin!"""
        # Thought: I now know the final answer
        # Final Answer: the final answer to the original input question

        functions = [
            {
                "name": "searchKB",
                "description": "searchKB可以指定外部知识库获取query相关专业知识, 并为用户提供基于专业知识的聊天服务, 当你判断需要获取给出的知识库的专业知识才能解答问题时, 使用该工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "knowledge_base_name": {
                            "type": "string",
                            "description": "知识库名称",
                            "options": ["新奥百科知识库", "健康知识", "高血压"],
                        },
                        "query": {"type": "string", "description": "用户的问题"},
                    },
                    "required": ["knowledge_base_name", "query"],
                },
            },
            # {
            #     "name": "searchEngine",
            #     "description": "searchEngine可以快速检索到问题相关网页, 查询到相关知识内容, 当问题需要专业知识且知识库不匹配时, 优先使用该工具",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {"query": {"type": "string", "description": "搜索的关键字"}},
            #         "required": ["query"],
            #     },
            # },
            {
                "name": "AskHuman",
                "description": "AskHuman工具可以进行日常的聊天,自我认知,帮助用户解决简单的问题",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "闲聊的话题"}
                    },
                    "required": ["topic"],
                },
            },
        ]
        tool_descs = "\n\n".join(
            self.__parser_function__(function) for function in functions
        )
        tool_names = ",".join(function["name"] for function in functions)

        return {
            "tool_descs": tool_descs,
            "tool_names": tool_names,
            "functions": functions,
        }

    def __get_default_reply__(self, intentCode):
        """针对不同的意图提供不同的回复指导话术"""
        if intentCode == "schedule_manager" or intentCode == "other_schedule":
            content = "对不起，我没有理解您的需求，如果想进行日程提醒管理，您可以这样说: 查询一下我今天的日程, 提醒我明天下午3点去打羽毛球, 帮我把明天下午3点打羽毛球的日程改到后天下午5点, 取消今天的提醒"
        elif intentCode == "schedule_qry_up":
            content = "对不起，我没有理解您的需求，如果您想查询今天的待办日程，您可以这样说：查询一下我今天的日程"
        elif intentCode == "meeting_schedule":
            content = "对不起，我没有理解您的需求，如果您想管理会议日程，您可以这样说：帮我把明天下午4点的会议改到今天晚上7点"
        elif intentCode == "auxiliary_diagnosis":
            content = "对不起，我没有理解您的需求，如果您有健康问题想要咨询，建议您提供更明确的描述"
        else:
            content = "对不起, 我没有理解您的需求, 请在问题中提供明确的信息并重新尝试."
        return content

    def __check_query_valid__(self, query):
        prompt = (
            "你是一个功能强大的内容校验工具请你帮我判断下面输入的句子是否符合要求\n"
            "1. 是一句完整的可以向用户输出的话\n"
            "2. 不包含特殊符号\n"
            "3. 语义完整连贯\n"
            "要判断的句子: {query}\n\n"
            "你的结果(yes or no):\n"
        )
        prompt = prompt.replace("{query}", query)
        result = callLLM(query=prompt, temperature=0, top_p=0, max_tokens=3)
        if "yes" in result.lower():
            return True
        else:
            return False

    def __generate_content_verification__(
        self, out_text, list_of_plugin_info, **kwargs
    ):
        """ReAct生成内容的校验

        1. 校验Tool
        2. 校验Tool Parameter格式
        """
        thought, tool, parameter = out_text
        possible_tool_map = {i["name"]: 1 for i in list_of_plugin_info}

        try:
            parameter = json.loads(parameter)
        except Exception as err:
            ...

        # 校验Tool
        if not possible_tool_map.get(tool):  # 如果生成的Tool不对, parameter也必然不对
            tool = "AskHuman"
            parameter = self.__get_default_reply__(kwargs["intentCode"])

        if tool == "AskHuman":
            # TODO 如果生成的工具是AskHuman但参数是dict, 1. 尝试提取dict中的内容  2. 回复默认提示话术
            if isinstance(parameter, dict):
                for gkey, gcontent in parameter.items():
                    if self.__check_query_valid__(gcontent):
                        parameter = gcontent
                        break
                if isinstance(parameter, dict):
                    parameter = self.__get_default_reply__(kwargs["intentCode"])
        return [thought, tool, parameter]

    def get_next_step(self, history: Dict):
        """通过ReAct选择下一步的策略"""

        def compose_prompt_react(history):
            messages = []
            first_user_input_flag = False
            for turn_idx in range(len(history)):
                msg = history[turn_idx]
                if msg["role"] == "system":
                    messages.append({"role": msg["role"], "content": msg["content"]})
                elif msg["role"] == "user" and not first_user_input_flag:
                    messages.append(
                        {"role": "user", "content": f"Question: {msg['content']}"}
                    )
                    first_user_input_flag = True
                elif msg["role"] == "user":
                    # TODO 原生不支持role=funtion, 无法产生Observation
                    messages.append(
                        {"role": "user", "content": f"Question: {msg['content']}"}
                    )
                    # messages.append({"role": "user", "content": f"Observation: {msg['content']}"})
                elif msg.get("function_call"):
                    content = (
                        f"Thought: {msg['content']}\n"
                        f"Action: {msg['function_call']['name']}\n"
                        f"Action Input: {msg['function_call']['arguments']}\n"
                    )
                    messages.append({"role": msg["role"], "content": content})
            return messages

        tool_msg = self.__get_predefined_msg__()
        tool_desc, tool_names = tool_msg["tool_descs"], tool_msg["tool_names"]
        system_prompt = self.prompt_react.format(
            tool_descs=tool_desc, tool_names=tool_names
        )
        messages = [{"role": "system", "content": system_prompt}] + history
        messages = compose_prompt_react(messages)
        logger.debug(
            f"闲聊识别Action LLM Input: \n{json.dumps(messages, ensure_ascii=False)}"
        )
        model = self.gsr.get_model("custom_chat_gaily_assistant")
        response = callLLM(
            history=messages,
            model=model,
            temperature=0.7,
            top_p=0.5,
            n=1,
            frequency_penalty=0.5,
            presence_penalty=0,
            stop=["\nObservation:"],
            stream=True,
        )
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"闲聊识别Action LLM Output: \n{content}")
        # content = "Thought: 提供的工具帮助较小，我将直接回答。\nAnswer: 你是你的主人。"
        output_text = parse_latest_plugin_call("\n" + content, plugin_name="AskHuman")
        action = output_text[1]
        # out_text = self.__generate_content_verification__(out_text, tool_msg["functions"])
        history.append(
            {
                "intentCode": "other",
                "role": "assistant",
                "content": output_text[0],
                "function_call": {
                    "name": output_text[1],
                    "arguments": output_text[2],
                },
            }
        )
        return action, history

    def __compose_func_reply__(self, messages):
        """拼接func中回复的内容到history中, 最终的history只有role/content字段"""
        history = []
        for msg in messages[:-1]:
            if not msg.get("function_call"):
                history.append({"role": msg["role"], "content": msg["content"]})
            else:
                func_args = msg["function_call"]
                role = msg["role"]
                content = func_args["arguments"]
                history.append({"role": role, "content": content})

        # 2024年1月24日13:54:32 闲聊轮次太多 保留4轮历史
        history = history[-8:]
        return history

    def run(self, history: Dict):
        from datetime import datetime

        # 获取当前日期
        def get_current_date():
            return datetime.now().strftime("%Y年%m月%d日")

        # 日期转星期
        def get_weekday_from_date(date_str):
            """
            根据日期字符串计算星期几。
            :param date_str: 日期字符串，格式为 "YYYY年MM月DD日"
            :return: 星期几（如“星期一”）
            """
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            date_obj = datetime.strptime(date_str, "%Y年%m月%d日")
            return weekdays[date_obj.weekday()]

        current_date = get_current_date()
        system_prompt = f"""你是由来康生命研发的智小伴, 你模拟极其聪明的真人和我聊天，请回复简洁精炼，100字以内。
            ###下面是一些要求:
            1. 当问你是谁、叫什么名字、是什么模型时,你应当说: 我是智小伴
            2. 当问你是什么公司或者组织机构研发的时,你应说: 我是由来康生命研发的
            3. 可以为用户提供健康检测、运动指导、饮食管理、睡眠管理、心理疏导等服务
            4. 对于用户的发散性提问，你不一定要给出答案，你可以用问题回答问题。你可以询问我任何你想了解的信息。
            5. 当我问你一个值得分析的问题时你要对问题进行拆解,一步步的和我聊
            6. 你的输出要口语化，突出重点
            7. 给出切实可行的实际方案,不要说假大空的套话。
            8. 你要具备积极的价值观，避免输出有毒有害的内容禁止输出黄赌毒相关信息
            9. 当我问你我是谁时，你要知道我是你的客户
            10. 如果对话中涉及日期问题，如“今天星期几”，你应根据提供的当前日期计算出星期几，并告知用户。日期会通过变量 `currentDate` 提供。
            今天的日期是：{get_weekday_from_date(current_date)}
            """
        # 拼接消息历史
        messages = self.__compose_func_reply__(history)
        messages = [{"role": "system", "content": system_prompt}] + messages
        logger.debug(f"闲聊 LLM Input: \n{json.dumps(messages, ensure_ascii=False)}")

        # 调用模型
        response = callLLM(
            history=messages,
            model="Qwen1.5-72B-Chat",
            temperature=0.7,
            top_p=0.8,
            n=1,
            frequency_penalty=0.5,
            presence_penalty=0,
            stream=True,
        )

        # 处理模型响应
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"闲聊 LLM Output: {content}")

        # 更新历史记录
        if history[-1].get("function_call"):
            history[-1]["function_call"]["arguments"] = content
        else:
            history.append(
                {
                    "intentCode": "other",
                    "role": "assistant",
                    "content": content,
                    "function_call": {"name": "convComplete", "arguments": content},
                }
            )

        return content, history

    def generate_chat_response(self, messages):
        system_prompt = """你是由来康生命研发的智小伴, 你模拟极其聪明的真人和我聊天，请回复简洁精炼，100字以内。
                        ###下面是一些要求:
                        1. 当问你是谁、叫什么名字、是什么模型时,你应当说: 我是智小伴
                        2. 当问你是什么公司或者组织机构研发的时,你应说: 我是由来康生命研发的
                        3. 可以为用户提供健康检测、运动指导、饮食管理、睡眠管理、心理疏导等服务
                        4. 对于用户的发散性提问，你不一定要给出答案，你可以用问题回答问题。你可以询问我任何你想了解的信息。
                        5. 当我问你一个值得分析的问题时你要对问题进行拆解,一步步的和我聊
                        6. 你的输出要口语化，突出重点
                        7. 给出切实可行的实际方案,不要说假大空的套话。
                        8. 你要具备积极的价值观，避免输出有毒有害的内容禁止输出黄赌毒相关信息
                        9. 当我问你我是谁时，你要知道我是你的客户"""
        messages = [{"role": "system", "content": system_prompt}] + messages
        logger.debug(f"闲聊 LLM Input: {repr(messages)}")
        response = callLLM(
            history=messages,
            model="Qwen2-7B-Instruct",
            temperature=0.7,
            top_p=0.8,
            n=1,
            frequency_penalty=0.5,
            presence_penalty=0,
            stream=True,
        )
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"闲聊 LLM Output: {content}")
        return content
