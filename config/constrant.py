# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-20 14:16:41
@desc    :   param template
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


PLAN_MAP = {
            '辅助诊断': ("对于医学相关问题,请遵循以下流程:\n"+
                        "1. 首先明确患者主诉信息，如果没有持续时间、发生时机、诱因或症状发生部位不明确可以一步一步向患者询问。\n"+
                        "2. 得到答案后根据患者症状，推断用户可能患有的疾病，一步一步询问患者疾病初诊、鉴别诊断、确诊需要的其他信息, 如家族史、既往史、检查结果等信息。\n"+
                        "3. 最终给出初步诊断结果，给出可能性最高的几种诊断，并按照可能性排序。\n"),
            '日常对话': "",
            '疾病/症状/体征异常问诊': ""
        }

TEMPLATE_ENV = "为{var}用户提供健康咨询和管理服务"
TEMPLATE_SENCE = "面向{var}"
TEMPLATE_ROLE = "请你扮演一个{var}的角色"
TEMPLATE_PLAN = "遵循以下流程完成任务:\n{var}"


task_schedule_return_demo = [
    {
        "task": "做饭",
        "event": "创建",
        "tmp_time": "2023-10-27 10:45:59",
        "occur_time": "2023-10-27 12:00:00",
        "remind_rule": "提前半小时",
        "remind_time": "2023-10-27 11:30:00",
		"ask": "已为你执行日程操作",
		"cron": "0 30 11 * * *"
    },
	{
        "task": "",
        "event": "search",
        "tmp_time": "2023-10-27 10:45:59",
        "occur_time": "",
        "remind_rule": "",
        "remind_time": "",
		"ask": "",
		"cron": ""
    },
	{
        "task": "做饭",
        "event": "修改",
        "tmp_time": "2023-10-27 10:45:59",
        "occur_time": "2023-10-27 12:00:00",
        "remind_rule": "提前10分钟",
        "remind_time": "2023-10-27 11:20:00",
		"ask": "已为你执行日程操作",
		"cron": ""
    },
	{
        "task": "做饭",
        "event": "取消",
        "tmp_time": "2023-10-27 10:45:59",
        "occur_time": "",
        "remind_rule": "",
        "remind_time": "",
		"ask": "已为你执行日程操作",
		"cron": ""
    }
]

task_schedule_parameter_description = [
    {"name": "task", "description": "任务名称", "required": True, "schema": {"type": "string"}}, 
    {"name": "remind_time", "description": "提醒时间", "required": True, "schema": {"type": "string","format": "timestamp"}}, 
    # {"name": "cron", "description": "在Linux和Unix系统中定期执行任务的时间调度器", "required": True, "schema": {"type": "string[int]"}}, 
    {"name":"ask", "description": "当用户输入信息不全时,通过此字段进一步询问;当输入信息完整,当前任务已完成时,输出: 已为你执行日程操作", "required": True, "schema": {"type":"string"}}
]

task_schedule_parameter_description_for_qwen = [
    {
        "name_for_human": "创建日程提醒",
        "name_for_model": "createPlan",
        "description_for_model": "理解用户原始意图,明确日程和提醒时间后,调用此工具创建日程提醒. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "remind_time", "description": "提醒的时间", "required": True, "schema": {"type": "string","format": "timestamp"}}, 
            {"name": "ask","description": "告知用户日程名称/时间和创建成功的信息","required": True,"schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "询问日程提醒时间",
        "name_for_model": "askForRemindTime",
        "description_for_model": "用户意图为创建日程提醒,但给出的日程时间不明确,调用此工具向用户确认日程提醒的具体时间,不要重复询问. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "ask","description": "向用户询问提醒的时间","required": True,"schema": {"type": "string"}} 
        ]
    },
    {
        "name_for_human": "取消日程",
        "name_for_model": "cancelPlan",
        "description_for_model": "用户想取消当前日程时,调用本工具取消对应日程. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "ask","description": "告知用户日程已取消","required": True,"schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "修改日程",
        "name_for_model": "changePlan",
        "description_for_model": "用户想修改日程时,若已存在日程时,可调用本工具修改对应日程时间. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}},
            {"name": "remind_time", "description": "当前用户希望日程提醒的时间", "required": True, "schema": {"type": "string","format": "timestamp"}},
            {"name": "ask", "descripton": "告知用户日程修改成功", "required": False, "schema": {"type": "string"}}
        ]
    },
    {
        "name_for_human": "查询日程",
        "name_for_model": "searchPlan",
        "description_for_model": "用户想查询特定的日程信息时调用此工具. Format the arguments as a JSON object.",
        "parameters": [
            {"name": "task","description": "日程名称","required": True,"schema": {"type": "string"}}
        ]
    },
]

TEMPLATE_TASK_SCHEDULE_MANAGER = """你是一个严谨的时间管理助手，可以帮助用户定制日程、查询日程、根据给出的规则修改日程发生时间和提醒时间、取消日程提醒,以下是一些指导要求:
- 确定日程需要明确以下关键信息:
    - `task`:日程名称
    - `event`:当前时间
    - `occur_time`:明确指出具体的发生时间,不允许自动联想,补全
    - `remind_rule`:明确给出的提醒规则,必须由用户明确给出,如不清晰,请在`ask`字段中进一步向用户确定
    - `remind_time`:根据`remind_rule`和`curr_time`生成
    - `ask`: 向用户咨询/反馈的内容,字段不允许为空
- 如果以上任何关键字段信息缺失,请在`ask`字段中向用户询问缺失内容
- 如果task已完成,请在`ask`字段中回复: 已为你执行日程操作
- 定制、查询、修改、取消日程的数据格式和要求如下:

# 数据返回格式
{task_schedule_return_demo}

## 参数说明如下:
{task_schedule_parameter_description}

当前日程状态:
{curr_plan}
当前时间: {tmp_time}
"""

class ParamServer:
    @property
    def llm_with_documents(cls):
        return {
            "knowledge_base_name": "samples",
            "local_doc_url": False,
            "model_name": "Baichuan2-13B-Chat-API",
            "query": None,
            "score_threshold": 0.6,
            "stream": False,
            "temperature": 0.7,
            "top_k": 3
        }
    
    @property
    def llm_with_search_engine(cls):
        return {
            "query": "",
            "search_engine_name": "duckduckgo",
            "top_k": 3,
            "stream": False,
            "model_name": "Baichuan2-13B-Chat-API",
            "temperature": 0.7
        }

    @property
    def llm_with_graph(cls):
        return {
            "model": "Baichuan2-13B-Chat",
            "messages":[
                {
                    "role":"user",
                    "content":""
                }
            ]
        }