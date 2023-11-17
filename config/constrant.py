# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-20 14:16:41
@desc    :   param template
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


PLAN_MAP = {
    '辅助诊断': ("对于医学相关问题,请遵循以下流程:\n"
                "1. 明确患者主诉信息后，一步一步询问患者持续时间、发生时机、诱因或症状发生部位等信息，每步只问一个问题\n"
                "2. 得到答案后根据患者症状，推断用户可能患有的疾病，逐步询问患者疾病初诊、鉴别诊断、确诊需要的其他信息, 如家族史、既往史、检查结果等信息\n"
                "3. 最终给出初步诊断结果，给出可能性最高的几种诊断，并按照可能性排序\n"
                "4. 用户不喜欢擅自帮他做任何决定，所有外部行为必须询问用户进行二次确认\n"
                "5. 可以适当忽略与本次对话无关的历史信息,以解决当前问题为主\n"
                "6. 请严格按照上述流程返回Action Input的内容，不要自由发挥"
                )
}
TEMPLATE_ROLE = "请你扮演一个{var}的角色"
TEMPLATE_SENCE = "面向{var}"
TEMPLATE_ENV = "为{var}用户提供健康咨询和管理服务"
TEMPLATE_PLAN = "遵循以下流程完成任务:\n{var}"


TOOL_CHOOSE_PROMPT = """你是智能健康管家，可以根据用户的对话内容，从工具列表中选择对应工具完成用户的任务。
{external_information}

现提供以下工具:
- 调用外部知识库: 允许你在自身能力无法结局当前问题时调用外部知识库获取更专业的知识解决问题，提供帮助
- 进一步询问用户的情况: 当前用户提供的信息不足，需要进一步询问用户相关信息以提供更全面，具体的帮助
- 直接回复用户问题：问题过于简单，且无信息缺失，可以直接回复
- 结束话题: 结合历史判断当前话题已结束

请遵循以下格式回复:
Question: 用户的问题
Thought: 可以适当忽略历史记录中的无关信息,思考针对当前问题应该做什么
Action: 选择的解决用户当前问题的工具
Action Input: Action的输入,通常为一个query,表示要查询或者询问的内容,一次只问一个问题,询问太多会引起用户厌烦
Observation: 工具返回的内容
...(Thought/Action/Action Input 可能会循环一次或多次直到解决问题)
Action: 前置话题可以终结了,调用结束话题工具
Action Input: Action的输入,通常为一个query,表示要查询或者询问的内容

Begins!
"""

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

# TEMPLATE_TASK_SCHEDULE_MANAGER = """你是一个严谨的时间管理助手，可以帮助用户定制日程、查询日程、根据给出的规则修改日程发生时间和提醒时间、取消日程提醒,以下是一些指导要求:
# - 确定日程需要明确以下关键信息:
#     - `task`:日程名称
#     - `event`:当前时间
#     - `occur_time`:明确指出具体的发生时间,不允许自动联想,补全
#     - `remind_rule`:明确给出的提醒规则,必须由用户明确给出,如不清晰,请在`ask`字段中进一步向用户确定
#     - `remind_time`:根据`remind_rule`和`curr_time`生成
#     - `ask`: 向用户咨询/反馈的内容,字段不允许为空
# - 如果以上任何关键字段信息缺失,请在`ask`字段中向用户询问缺失内容
# - 如果task已完成,请在`ask`字段中回复: 已为你执行日程操作
# - 定制、查询、修改、取消日程的数据格式和要求如下:

# # 数据返回格式
# {task_schedule_return_demo}

# ## 参数说明如下:
# {task_schedule_parameter_description}

# 当前日程状态:
# {curr_plan}
# 当前时间: {tmp_time}
# """

INTENT_PROMPT = """你是一个功能强大的信息理解、归纳总结工具,分析其意图,请遵循以下几点要求:
1. 重点分析最后一轮的对话记录
2. 用户说感谢/或者日常聊天时,意图为其他
3. 请从意图列表中选择意图,意图列表如下:
辅助诊断: 用户咨询健康、疾病相关问题
菜谱咨询: 用户咨询相关菜的做法
饮食分析: 用户按要求提供了当餐的食物种类或数量
音乐播放: 用户要求听歌或播放音乐
天气查询: 用户询问天气情况或要求查询天气
日程管理: 日程管理可以帮助用户进行日程的创建、修改、查询、取消操作
饮食咨询: 用户咨询如何饮食
运动咨询: 用户咨询如何运动或做哪些运动
呼叫健管师: 用户明确需要寻求健管师的帮助或要求拨打健管师的电话
呼叫运动师: 用户明确需要寻求运动师的帮助或要求拨打运动师的电话
呼叫营养师: 用户明确需要寻求营养师的帮助或要求拨打营养师的电话
呼叫情志师: 用户明确需要寻求情志师的帮助或要求拨打情志心理师的电话
呼叫医师: 用户明确需要寻求医师的帮助或要求拨打医生的电话
省会查询: 用户明确要求查询省的省会
时间日期: 用户明确要求查询日期
网络搜索: 用户明确要求搜索网络
首都查询: 用户明确要求查询国家的首都
彩票: 用户咨询或购买彩票
周公解梦: 用户提到解梦相关
计算器: 用户要求使用计算器
国内城市查询: 用户要求进行国内城市查询
万年历: 用户要求查万年历
有声节目: 用户要求播放有声节目
翻译: 用户要求进行翻译
垃圾分类: 用户要求查询垃圾分类
尾号限行: 用户要求查询限行车牌尾号
单位换算: 用户要求进行单位的换算
汇率: 用户要求查询汇率
眼保健操: 用户提到做眼保健操
新闻: 用户明确要求看新闻
故事: 用户要求听故事或讲故事
圣经: 用户提到圣经相关
戏曲: 用户明确要求听戏曲或播放戏曲
评书: 用户明确要求听评书或播放评书
有声书: 用户明确要求听有声书或播放有声书
其他: 不符合前面意图列表项，则输出其它意图


请遵循以下格式回复:
Question: 用户的问题
Answer: 对于上面问题的回复
Question: 用户的问题
...(Question/Answer 可能会循环多次,最后一次永远是输出意图列表中的意图)
Thought: 思考当前用户问题符合意图里表中哪个意图的解释
Intent: 当前问题对应列表中的意图

Begins!

"""
# - Use the following format:
# Thought: you should always think about what to do
# Action: Your chosen intention, should be one of [辅助诊断,日程管理,呼叫健管师,呼叫运动师,呼叫营养师,呼叫情志师,呼叫医师,饮食咨询,运动咨询,其他]

# Begin!<|im_end|>

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
