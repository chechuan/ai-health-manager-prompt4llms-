# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-31 09:29:34
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

from datetime import datetime
from re import T

t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
t_claim_str = f"现在的时间是{t}\n"

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """You are a powerful schedule management tool, please follow the following requirements to complete the corresponding tasks：
- Answer the following questions as best you can.
- You have access to the following APIs:

{tools_text}

- Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action, should be a JSON object
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)

当前日程:
{temp_schedule}

现在时间是: {tmp_time}
Begin!

"""

# - if user want to create schedule and time is unclear, call `ask_for_time` tool once to get schedule time before `create_schedule`
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
# {"name": "cron", "description": "在Linux和Unix系统中定期执行任务的时间调度器", "required": True, "schema": {"type": "string[int]"}}, 
task_schedule_parameter_description = [
    {"name": "task", "description": "任务名称", "required": True, "schema": {"type": "string"}}, 
    {"name": "remind_time", "description": "提醒时间", "required": True, "schema": {"type": "string","format": "yyyy-MM-dd HH:mm:ss"}}, 
    {"name": "ask", "description": "当日程信息不全时,进一步询问;当输入信息完整,当前任务已完成时,输出: 已为你执行日程操作", "required": True, "schema": {"type":"string"}}
]

task_schedule_parameter_description_for_qwen = [
    {
        "name_for_human": "创建日程",
        "name_for_model": "create_schedule",
        "description_for_model": "一个用于创建日程的工具,提取日程名称(不含时间信息)和提醒的时间用来创建日程. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "简洁的日程名称,名称中不要有时间的描述","required": True,"schema": {"type": "string"}},
            {"name": "time", "description": "提醒的时间", "required": True, "schema": {"type": "string","format": "yyyy-MM-dd HH:mm:ss"}}, 
            {"name": "ask","description": "告知用户日程创建完成及具体的提醒时间","required": True,"schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "询问日程时间",
        "name_for_model": "ask_for_time",
        "description_for_model": "向用户询问时间的工具. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "ask","description": "向用户询问日程提醒的时间","required": True,"schema": {"type": "string"}} 
        ]
    },
    {
        "name_for_human": "取消日程",
        "name_for_model": "cancel_schedule",
        "description_for_model": "取消日程是已给帮助用户取消当前日程的工具. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "ask","description": "告知用户日程已取消","required": True,"schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "修改日程",
        "name_for_model": "modify_schedule",
        "description_for_model": "修改日程是一个帮助用户修改当前日程的工具，可使用本工具修改对应日程时间. Format the arguments as a JSON object.",
        "parameters": [
            # {"name": "ask", "descripton": "告知用户日程修改完成和提醒时间", "required": True, "schema": {"type": "string"}},
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "time", "description": "当前用户希望日程提醒的时间", "required": True, "schema": {"type": "string","format": "yyyy-MM-dd HH:mm:ss"}},
        ]
    },
    {
        "name_for_human": "查询日程",
        "name_for_model": "query_schedule",
        "description_for_model": "查询日程是一个帮助用户查询特定的日程信息的工具. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}}
        ]
    },
]

_tspdfq = [
    {
        "name_for_human": "创建日程",
        "name_for_model": "create_schedule",
        "description_for_model": "一个用于创建日程的工具,提取日程名称(不含时间信息)和提醒的时间用来创建日程. Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "task",
                "schema": {
                    "type": "string"
                },
                "required": True,
                "description": "日程名,尽量简洁,只包含事件,不包含日期,时间信息"
            },
            {
                "name": "time",
                "schema": {
                    "type": "string",
                    "format": "yyyy-MM-ddHH:mm:ss"
                },
                "required": True,
                "description": "提醒的时间"
            },
            {
                "name": "ask",
                "schema": {
                    "type": "string"
                },
                "required": True,
                "description": "告知用户日程创建完成及具体的提醒时间"
            }
        ]
    },
    {
        "name_for_human": "取消日程",
        "name_for_model": "cancel_schedule",
        "description_for_model": "取消日程是已给帮助用户取消当前日程的工具. Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "task",
                "schema": {
                    "type": "string"
                },
                "required": True,
                "description": "日程名称"
            },
            {
                "name": "ask",
                "schema": {
                    "type": "string"
                },
                "required": True,
                "description": "告知用户日程已取消"
            }
        ]
    },
    {
        "name_for_human": "修改日程",
        "name_for_model": "modify_schedule",
        "description_for_model": "修改日程是一个帮助用户修改当前日程的工具，可使用本工具修改对应日程时间. Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "task",
                "schema": {
                    "type": "string"
                },
                "required": True,
                "description": "日程名称"
            },
            {
                "name": "time",
                "schema": {
                    "type": "string",
                    "format": "yyyy-MM-dd HH:mm:ss"
                },
                "required": True,
                "description": "当前用户希望日程提醒的时间"
            }
        ]
    },
    {
        "name_for_human": "查询日程",
        "name_for_model": "query_schedule",
        "description_for_model": "查询日程是一个帮助用户查询特定的日程信息的工具. Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "task",
                "schema": {
                    "type": "string"
                },
                "required": False,
                "description": "日程名称"
            }
        ]
    }
]

query_schedule_template = (
    "你将扮演智能健康管家,现在时间是{{cur_time}},请你根据用户的日程列表,生成用户的日程提醒,要求语言表达自然流畅,态度温和,请仿照给出的示例回复\n\n"
    "示例:\n"
    "用户日程为：\n"
    "还需完成5项任务任务\n"
    "事项：血压测量,时间：8:00、20:00\n"
    "事项：用餐,时间：11：00、17：00\n"
    "事项：会议,时间：14：00\n"
    "事项：用药,时间：21：00\n"
    "事项：慢走20min,今日完成,时间：21：00\n\n"
    "日程提醒:\n"
    "您还有5项日程需要完成\n"
    "14点您有1项会议,请合理安排时间,我会在会议开始前通知您\n"
    "请根据食谱合理搭配11点的中餐、下午5点的晚餐\n"
    "晚上20点钟需要测量血压,睡前21点还要服用药物\n"
    "今日需要完成一项慢走20分钟的运动\n\n"
)