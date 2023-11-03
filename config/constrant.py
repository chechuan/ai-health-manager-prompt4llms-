# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-20 14:16:41
@desc    :   param template
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


PLAN_MAP = {
    '辅助诊断': ("对于医学相关问题,请遵循以下流程:\n"
                "1. 明确患者主诉信息后，一步一步询问患者持续时间、发生时机、诱因或症状发生部位等信息，每步只问一个问题。\n"
                "2. 得到答案后根据患者症状，推断用户可能患有的疾病，逐步询问患者疾病初诊、鉴别诊断、确诊需要的其他信息, 如家族史、既往史、检查结果等信息。\n"
                "3. 最终给出初步诊断结果，给出可能性最高的几种诊断，并按照可能性排序。\n"),
    '日常对话': "",
    '疾病/症状/体征异常问诊': ""
}

TEMPLATE_ENV = "为{var}用户提供健康咨询和管理服务"
TEMPLATE_SENCE = "面向{var}"
TEMPLATE_ROLE = "请你扮演一个{var}的角色"
TEMPLATE_PLAN = "遵循以下流程完成任务:\n{var}"


TOOL_CHOOSE_PROMPT = """你是智能健康管家，可以根据用户的对话内容，从工具列表中选择对应工具完成用户的任务。

现提供以下工具:
- 调用外部知识库: 允许你在自身能力无法结局当前问题时调用外部知识库获取更专业的知识解决问题，提供帮助
- 进一步询问用户的情况: 当前用户提供的信息不足，需要进一步询问用户相关信息以提供更全面，具体的帮助
- 直接回复用户问题：问题过于简单，且无信息缺失，可以直接回复

请遵循以下格式回复:
Thought: 思考当前应该做什么
Action: 选择的解决用户当前问题的工具
Action Input: 当前工具需要用的参数,可以是调用知识库的参数,询问用户的问题,一次只针对一个主题询问
Observation: 工具返回的内容
...(Thought/Action/Action Input 可能会循环一次或多次直到解决问题)
Thought: bingo
Final Answer: the final answer to the original input question

[对话背景信息]
{external_information}
[对话背景信息]"""

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

INTENT_PROMPT = """<|im_start|>system
你是一个功能强大的信息提取assistant, 请你重点分析用户输入, 提取其意图.
-  意图列表如下:
日程管理: 日程管理可以帮助用户进行日程的创建、修改、查询、取消操作
饮食咨询: 用户咨询如何饮食
运动咨询: 用户咨询如何运动或做哪些运动
辅助诊断: 为用户提供医学健康咨询服务
呼叫健管师: 用户明确需要寻求健管师的帮助
呼叫运动师: 用户明确需要寻求运动师的帮助
呼叫营养师: 用户明确需要寻求营养师的帮助
呼叫情志师: 用户明确需要寻求情志师的帮助
呼叫医师: 用户明确需要寻求医师的帮助
其他: 不符合前面意图列表项，则输出其它意图

- Use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [辅助诊断,日程管理,呼叫健管师,呼叫运动师,呼叫营养师,呼叫情志师,呼叫医师,饮食咨询,运动咨询,其他]

Begin!<|im_end|>
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