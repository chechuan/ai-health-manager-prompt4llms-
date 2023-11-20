# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-26 16:16:37
@desc    :   Prompt生产流
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append('.')
import yaml
from langchain.prompts.prompt import PromptTemplate
from typing_extensions import Dict

from config.constrant import (PLAN_MAP, TEMPLATE_ENV, TEMPLATE_PLAN,
                              TEMPLATE_ROLE, TEMPLATE_SENCE)
from src.utils.Logger import logger
from src.utils.module import MysqlConnector

# class baseVarsForPromptEngine:
#     """定义外部信息的变量
#     """
#     def __init__(self, 
#                  env: Optional[str] = "居家",
#                  scene: Optional[str] = "一般用户",
#                  role: Optional[str] = "健管师",
#                  role_desc: Optional[str] = "协助医生工作",
#                  plan: Optional[str] = "辅助诊断",
#                  prompt_meta_data: Optional[dict] = {}) -> None:
#         # 环境, options: 居家, 机构, 外出, 开车
#         self.env = env
#         # 场景, options: 一般用户, 专业工作人员, 为患者服务的工作人员
#         self.scene = scene
#         # 角色, options: 健管师, 医师, 营养师, 运动师, 情志调理师
#         self.role = role
#         # 角色任务描述
#         self.role_desc = role_desc
#         # 计划, options: 辅助诊断
#         self.plan = plan
#         # 外置prompt优先 prompt_meta_data
#         self.prompt_meta_data = prompt_meta_data

# class promptEngine:
#     """组装先验知识到INSTRUCTION_TEMPLATE中
#     """
#     def __init__(self, prompt_meta_data):
#         self.prompt_meta_data = prompt_meta_data
#         self.tpe_env = PromptTemplate(input_variables=["var"], template=TEMPLATE_ENV)
#         self.tpe_plan = PromptTemplate(input_variables=["var"], template=TEMPLATE_PLAN)
#         self.tpe_scene = PromptTemplate(input_variables=["var"], template=TEMPLATE_SENCE)
#         self.tpe_role = PromptTemplate(input_variables=["var"], template=TEMPLATE_ROLE)

#     def __concat(self, prompt: str, tpe: PromptTemplate, var: str, verbose: bool=False, **kwds) -> str:
#         concat_keyword = kwds.get("concat_keyword", ",") + " "
#         if prompt:
#             prompt += concat_keyword
#         prompt += tpe.format(var=var)
#         if verbose:
#             print(prompt)
#         return prompt

#     def _call(self, *args, **kwds):
#         """拼接知识体系内容
#         """
#         prompt = ""
#         bm = args[0]
#         if bm.role:
#             prompt = self.__concat(prompt, self.tpe_role, bm.role, **kwds)
#         if bm.scene:
#             prompt = self.__concat(prompt, self.tpe_scene, bm.scene, **kwds)
#         if bm.env:
#             prompt = self.__concat(prompt, self.tpe_env, bm.env, **kwds)
#         if bm.plan: 
#             if bm.prompt_meta_data and bm.prompt_meta_data.get("event") and bm.prompt_meta_data["event"].get(bm.plan):
#                 plan = bm.prompt_meta_data["event"][bm.plan]['description']+"\n"+\
#                         args[0].prompt_meta_data['event']['辅助诊断']['process']
#             else:
#                 plan = PLAN_MAP.get(bm.plan, f"当前意图{bm.plan}无对应流程")
#             prompt = self.__concat(prompt, self.tpe_plan, plan, **kwds)
#         return prompt

class baseVarsForPromptEngine:
    """定义外部信息的变量
    """
    def __init__(self, 
                 character: Optional[str] = "医生助手",
                 event: Optional[str] = "_辅助诊断") -> None:
        self.character = character
        self.event = event

class promptEngine:
    """组装先验知识到INSTRUCTION_TEMPLATE中
    """
    def __init__(self, prompt_meta_data=None):
        if prompt_meta_data:
            self.prompt_meta_data = prompt_meta_data
        else:
            self.req_prompt_data_from_mysql()

    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data
        """
        def filter_format(obj, splited=False):
            obj_str = json.dumps(obj, ensure_ascii=False).replace("\\r\\n", "\\n")
            obj_rev = json.loads(obj_str)
            if splited:
                for obj_rev_item in obj_rev:
                    if obj_rev_item.get('event'):
                        obj_rev_item['event'] = obj_rev_item['event'].split("\n")
            return obj_rev
        
        mysql_config = yaml.load(open(Path("config","mysql_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
        self.mysql_conn = MysqlConnector(**mysql_config)

        self.prompt_meta_data = {}
        prompt_character = self.mysql_conn.query("select * from ai_prompt_character")
        prompt_event = self.mysql_conn.query("select * from ai_prompt_event")
        prompt_tool = self.mysql_conn.query("select * from ai_prompt_tool")
        prompt_character = filter_format(prompt_character, splited=True)
        prompt_event = filter_format(prompt_event)
        prompt_tool = filter_format(prompt_tool)
        self.prompt_meta_data['character'] = {i['name']: i for i in prompt_character}
        self.prompt_meta_data['event'] = {i['event']: i for i in prompt_event}
        self.prompt_meta_data['tool'] = {i['name']: i for i in prompt_tool}
        logger.debug("req prompt meta data from mysql.")
    
    def __join_character(self, character: str, **kwds):
        """拼接角色部分的prompt
        
        - Args

            character (str, required)
                角色名, 可选于`ai_prompt_character`表`name`列
            **kwds
                key word arguments
        """
        assert isinstance(character, str)
        assert self.prompt_meta_data['character'].get(character), "character not found."
        c_item = self.prompt_meta_data['character'][character]
        prompt = ""
        if c_item.get('name', None):
            prompt += f"请你扮演一个经验丰富的{c_item['name']},"
        if c_item.get('description', None):
            if prompt:
                prompt += f"{c_item['description']}\n"
            else:
                prompt += f"请你{c_item['description']}\n"
        if c_item.get('duty', None):
            prompt += f"以下是对你职责的要求:\n{c_item['duty']}\n\n"
        if c_item.get('constraint', None):
            prompt += f"请注意:\n{c_item['constraint']}\n"
        return prompt
    
    def __join_event(self, event: str, **kwds):
        """拼接事件部分prompt

        - Args

            event (str, required)
                具体事件, 可选于`ai_prompt_event`表`event`列
            **kwds
                key word arguments
        """
        assert isinstance(event, str)
        assert self.prompt_meta_data['event'].get(event), "event not found."
        e_item = self.prompt_meta_data['event'][event]
        prompt = ""
        if e_item.get('event', None):
            name = e_item['event'].replace('_', '')
            prompt += f"\n对于{name}场景,请你结合以下信息与用户沟通:\n"
        if e_item.get('process', None):
            prompt += f"{e_item['process']}\n"
        if e_item.get('constraint'):
            prompt += f"\n注意事项:\n{e_item['constraint']}"
        return prompt

    def _call(self, *args, **kwds):
        """拼接角色事件知识
        """
        
        sys_prompt = kwds.get("sys_prompt", None)
        if sys_prompt and "角色" in sys_prompt or "医生" in sys_prompt or "患者" in sys_prompt:
            return kwds.get("sys_prompt")
        else:
            default_prompt = ("请你扮演一个经验丰富的医生助手,帮助医生处理日常诊疗和非诊疗的事务性工作,以下是一些对你的要求:\n"
                          "1. 明确患者主诉信息后，一步一步询问患者持续时间、发生时机、诱因或症状发生部位等信息，每步只问一个问题\n"
                          "2. 得到答案后根据患者症状，推断用户可能患有的疾病，逐步询问患者疾病初诊、鉴别诊断、确诊需要的其他信息, 如家族史、既往史、检查结果等信息\n"
                          "3. 最终给出初步诊断结果，给出可能性最高的几种诊断，并按照可能性排序\n"
                          "4. 用户不喜欢擅自帮他做任何决定，所有外部行为必须询问用户进行二次确认\n"
                          "5. 可以适当忽略与本次对话无关的历史信息,以解决当前问题为主\n"
                          "6. 请严格按照上述流程返回Action Input的内容，不要自由发挥\n\n"
                          "现提供以下工具,请你根据用户的对话内容,从工具列表中选择对应工具完成用户的任务:\n"
                          "- 进一步询问用户的情况: 当前用户提供的信息不足，需要进一步询问用户相关信息以提供更全面，具体的帮助\n"
                          "- 调用外部知识库: 允许你在自身能力无法结局当前问题时调用外部知识库获取更专业的知识解决问题，提供帮助\n"
                          "- 直接回复用户问题: 问题过于简单，且无信息缺失，结合历史给出诊断结果\n"
                          "- 结束话题: 当前用户问题不属于辅助诊断场景\n"
                          )
            # 调试阶段未命中给出的辅助诊断的prompt先试用此前给的默认的辅助诊断的prompt保证效果
            return default_prompt
            # 调试出稳定的sys_prompt后补充到数据库中,再读取数据拼接
            prompt = ""
            bm = args[0]
            if bm.character:
                prompt += self.__join_character(bm.character, **kwds)
            if bm.event:
                prompt += self.__join_event(bm.event, **kwds)
            return prompt


if __name__ == "__main__":
    pe = promptEngine()
    args = baseVarsForPromptEngine()
    pe._call(args, concat_keyword=",")