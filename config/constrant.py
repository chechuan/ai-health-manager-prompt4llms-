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