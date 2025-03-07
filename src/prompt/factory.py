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
import argparse

from langchain.prompts.prompt import PromptTemplate

from data.constrant import PLAN_MAP, TEMPLATE_ENV, TEMPLATE_PLAN, TEMPLATE_ROLE, TEMPLATE_SENCE
from src.utils.Logger import logger
from src.utils.resources import InitAllResource


class baseVarsForPromptEngine:
    """定义外部信息的变量
    """
    def __init__(self, 
                 env: Optional[str] = "居家",
                 scene: Optional[str] = "一般用户",
                 role: Optional[str] = "健管师",
                 role_desc: Optional[str] = "协助医生工作",
                 plan: Optional[str] = "_auxiliary_diagnosis",
                 prompt_meta_data: Optional[dict] = {}) -> None:
        # 环境, options: 居家, 机构, 外出, 开车
        self.env = env
        # 场景, options: 一般用户, 专业工作人员, 为患者服务的工作人员
        self.scene = scene
        # 角色, options: 健管师, 医师, 营养师, 运动师, 情志调理师
        self.role = role
        # 角色任务描述
        self.role_desc = role_desc
        # 计划, options: 辅助诊断
        self.plan = plan
        # 外置prompt优先 prompt_meta_data
        self.prompt_meta_data = prompt_meta_data

class promptEngine:
    """组装先验知识到INSTRUCTION_TEMPLATE中
    """
    def __init__(self, prompt_meta_data):
        self.prompt_meta_data = prompt_meta_data
        self.tpe_env = PromptTemplate(input_variables=["var"], template=TEMPLATE_ENV)
        self.tpe_plan = PromptTemplate(input_variables=["var"], template=TEMPLATE_PLAN)
        self.tpe_scene = PromptTemplate(input_variables=["var"], template=TEMPLATE_SENCE)
        self.tpe_role = PromptTemplate(input_variables=["var"], template=TEMPLATE_ROLE)

    def __concat(self, 
                 prompt: str, 
                 tpe: PromptTemplate, 
                 var: str, 
                 verbose: bool=False, 
                #  **kwds
                 ) -> str:
        concat_keyword = ","
        # concat_keyword = kwds.get("concat_keyword", ",") + " "
        if prompt:
            prompt += concat_keyword
        prompt += tpe.format(var=var)
        if verbose:
            print(prompt)
        return prompt

    def _call(self, *args, **kwds):
        """拼接知识体系内容
        """
        prompt = ""
        bm = args[0]
        role, scene, env, plan = bm.role, bm.scene, bm.env, bm.plan
        if role: prompt = self.__concat(prompt, self.tpe_role, bm.role)
        if scene: prompt = self.__concat(prompt, self.tpe_scene, bm.scene)
        if env: prompt = self.__concat(prompt, self.tpe_env, bm.env)
        if (
            plan 
            and self.prompt_meta_data 
            and self.prompt_meta_data.get("event") 
            and self.prompt_meta_data["event"].get(plan)
        ):
            prompt += self.prompt_meta_data["event"][plan]['description'] + "\n"
            prompt += self.prompt_meta_data['event'][plan]['process']
        else:
            plan = PLAN_MAP.get(plan, f"当前意图{plan}未定义对应处理")
            prompt = self.__concat(prompt, self.tpe_plan, plan)
        return prompt

class CustomPromptEngine:
    """组装先验知识到INSTRUCTION_TEMPLATE中
    """
    def __init__(self, global_share_resource):
        if global_share_resource.prompt_meta_data:
            self.prompt_meta_data = global_share_resource.prompt_meta_data
        else:
            parser = argparse.ArgumentParser()
            parser.add_argument('--env', type=str, default="local", help='env: local, dev, fat, prod')
            parser.add_argument('--ip', type=str, default="0.0.0.0", help='ip')
            parser.add_argument('--port', type=int, default=6500, help='port')
            args = parser.parse_args()
            try:
                self.prompt_meta_data = InitAllResource(args).prompt_meta_data
            except Exception as e:
                self.prompt_meta_data = {}
    
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
        lst_prompt = []
        if c_item.get('description', None):
            lst_prompt.append(c_item['description'])
        if c_item.get('duty', None):
            lst_prompt.append(c_item['duty'])
        if c_item.get('constraint', None):
            lst_prompt.append(c_item['constraint'])
        prompt = "\n\n".join(lst_prompt)
        return prompt
    
    def __join_event(self, e_item: str, **kwds):
        """拼接事件部分prompt

        - Args

            event (str, required)
                具体事件, 可选于`ai_prompt_event`表`event`列
            **kwds
                key word arguments
        """
        lst_prompt = []
        prompt = ""
        if e_item.get('description', None):
            lst_prompt.append(e_item['description'])
        if e_item.get('process', None):
            lst_prompt.append(e_item['process'])
        if e_item.get('constraint'):
            lst_prompt.append(e_item['constraint'])
        prompt = "\n\n".join(lst_prompt)
        return prompt

    def _call(self, *args, **kwds):
        """拼接角色事件知识
        """
        default_intent_code = "other"
        intent_code = kwds.get("intentCode", default_intent_code)
        if not self.prompt_meta_data['event'].get(intent_code):
            logger.exception(f"not support current enevt {intent_code}, change to {default_intent_code}")
            intent_code = default_intent_code
        if kwds.get('use_sys_prompt') and kwds.get("sys_prompt") and "schedule" not in intent_code:
            sys_prompt = kwds.get("sys_prompt", "")
        else:
            sys_prompt = ""
            event = self.prompt_meta_data['event'][intent_code]
            character = event['character'] if event.get('character') else None
            if character:
                sys_prompt += self.__join_character(character, **kwds) + "\n\n"
            if event and intent_code != "schedule_qry_up":
                sys_prompt += self.__join_event(event, **kwds)
        functions = []
        tools = self.prompt_meta_data['event'][intent_code]['tool_in_use']
        if tools:
            function_names = tools.split("\n")
            all_funcsets = self.prompt_meta_data['tool']
            if not kwds.get("tool_in_use"):
                functions = [all_funcsets[name] for name in function_names if all_funcsets.get(name)]
            else:
                print(all_funcsets)
                print(kwds.get("tool_in_use"))
                functions = [all_funcsets[name] for name in kwds['tool_in_use'] if all_funcsets.get(name)]
        return sys_prompt, functions


if __name__ == "__main__":
    pe = promptEngine()
    pe._call("intent_code", concat_keyword=",")