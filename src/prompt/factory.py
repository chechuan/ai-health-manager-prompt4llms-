# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-26 16:16:37
@desc    :   Prompt生产流
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import sys
from typing import Optional

from langchain.prompts.prompt import PromptTemplate

sys.path.append(".")
from config.constrant import (PLAN_MAP, TEMPLATE_ENV, TEMPLATE_PLAN,
                              TEMPLATE_ROLE, TEMPLATE_SENCE)


class baseVarsForPromptEngine:
    """定义外部信息的变量
    """
    # 环境, options: 居家, 机构, 外出, 开车
    env: Optional[str] = "居家"
    # 场景, options: 一般用户, 专业工作人员, 为患者服务的工作人员
    scene: Optional[str] = "一般用户"
    # 角色, options: 健管师, 医师, 营养师, 运动师, 情志调理师
    role: Optional[str] = "健管师"
    # 角色任务描述
    role_desc: Optional[str] = "协助医生工作"
    # 计划, options: 辅助诊断
    plan: Optional[str] = "辅助诊断"

class promptEngine:
    """组装先验知识到INSTRUCTION_TEMPLATE中
    """
    def __init__(self) -> None:
        self.tpe_env = PromptTemplate(input_variables=["var"], template=TEMPLATE_ENV)
        self.tpe_plan = PromptTemplate(input_variables=["var"], template=TEMPLATE_PLAN)
        self.tpe_scene = PromptTemplate(input_variables=["var"], template=TEMPLATE_SENCE)
        self.tpe_role = PromptTemplate(input_variables=["var"], template=TEMPLATE_ROLE)

    def __concat(self, prompt: str, tpe: PromptTemplate, var: str, verbose: bool=False, **kwds) -> str:
        concat_keyword = kwds.get("concat_keyword", ",") + " "
        if prompt:
            prompt += concat_keyword
        prompt += tpe.format(var=var)
        if verbose:
            print(prompt)
        return prompt

    def _call(self, *args, **kwds):
        """拼接知识体系内容
        """
        ret = ""
        bm = args[0]
        if bm.role:
            ret = self.__concat(ret, self.tpe_role, bm.role, **kwds)
        if bm.scene:
            ret = self.__concat(ret, self.tpe_scene, bm.scene, **kwds)
        if bm.env:
            ret = self.__concat(ret, self.tpe_env, bm.env, **kwds)
        if bm.plan: 
            plan = PLAN_MAP.get(bm.plan, f"当前意图{bm.plan}无对应流程")
            ret = self.__concat(ret, self.tpe_plan, plan, **kwds)
        return ret

if __name__ == "__main__":
    pe = promptEngine()
    args = baseVarsForPromptEngine()
    pe._call(args, concat_keyword=",")