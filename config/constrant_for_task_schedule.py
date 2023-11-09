# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-31 09:29:34
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

from datetime import datetime

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
    {"name": "remind_time", "description": "提醒时间", "required": True, "schema": {"type": "string","format": "timestamp"}}, 
    {"name": "ask", "description": "当日程信息不全时,通过此字段进一步询问;当输入信息完整,当前任务已完成时,输出: 已为你执行日程操作", "required": True, "schema": {"type":"string"}}
]

task_schedule_parameter_description_for_qwen = [
    {
        "name_for_human": "创建日程",
        "name_for_model": "create_schedule",
        # "description_for_model": "创建日程提醒是一个用于创建日程的工具，在调用询问日程提醒时间工具明确时间后，调用此工具创建日程提醒. Format the arguments as a JSON object.",
        "description_for_model": "一个用于创建日程的工具,如果日程时间不明确,先使用ask_for_time工具询问日程时间. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "time", "description": "提醒的时间", "required": True, "schema": {"type": "string","format": "timestamp"}}, 
            {"name": "ask","description": "告知用户日程创建完成及具体的提醒时间","required": True,"schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "询问日程时间",
        "name_for_model": "ask_for_time",
        "description_for_model": "创建日程时向用户询问时间的工具. Format the arguments as a JSON object.",
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
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "time", "description": "当前用户希望日程提醒的时间", "required": True, "schema": {"type": "string","format": "timestamp"}},
            {"name": "ask", "descripton": "告知用户日程修改成功", "required": True, "schema": {"type": "string"}}
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