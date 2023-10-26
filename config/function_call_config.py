# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-26 16:02:28
@desc    :   function call config
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''


function_tools = [
    {
        "name_for_human": "获取处理流程",
        "name_for_model": "get_plan",
        "description_for_model": "使用此工具可获取不同问题对应的处理流程。Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "query",
                "description": "参数值为问题的类别，可选类别如下:'辅助诊断','日常对话','疾病/症状/体征异常问诊'",
                "required": True,
                "schema": {
                    "type": "string"
                }
            }
        ]
    },
    {
        "name_for_human": "查询必要的知识",
        "name_for_model": "callForKnowledge",
        "description_for_model": "",
        "parameters": [
            {
                "name": "query",
                "description": "要查询的内容",
                "required": True,
                "schema": {
                    "type": "string"
                }
            }
        ]
    },
    {
        "name_for_human": "查询必要的信息",
        "name_for_model": "callForMessage",
        "description_for_model": "当需要更多业务数据支持解决当前问题时，使用此工具可以查询更多信息。Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "type",
                "description": "要查询的信息类型",
                "required": True,
                "schema": {
                    "type": "string",
                    "option": [
                        "历史会话记录",
                        "用户个人信息"
                    ]
                }
            }
        ]
    },
    {
        "name_for_human": "寻求真人服务",
        "name_for_model": "seekHumanHelp",
        "description_for_model": "如果用户要求或流程明确要求或者你判断用户情况紧急且自身无法解决用户需求，就调用此工具。Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "type",
                "description": "寻求帮助的类型",
                "required": True,
                "schema": {
                    "type": "string",
                    "option": [
                        "医师",
                        "健管师",
                        "营养师",
                        "运动师",
                        "情志师"
                    ]
                }
            }
        ]
    },
    {
        "name_for_human": "询问用户获取必要信息",
        "name_for_model": "askUser",
        "description_for_model": "与用户日常交流，当前返回的内容需要用户回答，或者用户提供的信息不足，使用此工具可以询问更多信息。Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "query",
                "description": "要询问患者的问题",
                "required": True,
                "schema": {
                    "type": "string"
                }
            }
        ]
    },
    {
        "name_for_human": "搜索引擎",
        "name_for_model": "llm_with_search_engine",
        "description_for_model": "duckduckgo是一个功能强大通用搜索引擎,可访问互联网、查询百科知识、了解时事新闻等,其他工具无法检索问题相关知识时,可以使用搜索引擎。Format the arguments as a JSON object.",
        "parameters": [
            {
                "name": "query",
                "description": "搜索关键词或短语",
                "required": True,
                "schema": {"type": "string"}
            }
        ]
    }
]