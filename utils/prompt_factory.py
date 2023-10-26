# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-26 16:16:37
@desc    :   Prompt生产
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain.prompts.prompt import PromptTemplate
from transformers import HfArgumentParser

sys.path.append(".")
from config.constrant import (PLAN_MAP, TEMPLATE_ENV, TEMPLATE_PLAN,
                              TEMPLATE_ROLE, TEMPLATE_SENCE)


@ dataclass
class baseMsg:
    env: Optional[str] = field(
        default=None,
        metadata={"help": "环境, options: 居家, 机构, 外出, 开车"},
    )
    scene: Optional[str] = field(
        default=None,
        metadata={"help": "场景, options: 一般用户, 专业工作人员, 为患者服务的工作人员"},
    )
    role: Optional[str] = field(
        default=None,
        metadata={"help": "角色, options: 健管师, 医师, 营养师, 运动师, 情志调理师"}
    )
    plan: Optional[str] = field(
        default=None,
        metadata={"help": "计划, options: 健管师, 医师, 营养师, 运动师, 情志调理师"}
    )


class promptEngine:
    def __init__(self) -> None:
        self.tpe_env = PromptTemplate(input_variables=["var"], template=TEMPLATE_ENV)
        self.tpe_plan = PromptTemplate(input_variables=["var"], template=TEMPLATE_PLAN)
        self.tpe_scene = PromptTemplate(input_variables=["var"], template=TEMPLATE_SENCE)
        self.tpe_role = PromptTemplate(input_variables=["var"], template=TEMPLATE_ROLE)

    def __concat(self, prompt: str, tpe: PromptTemplate, var: str, verbose: bool=True) -> str:
        if prompt:
            prompt += ", "
        prompt += tpe.format(var=var)
        if verbose:
            print(prompt)
        return prompt

    def _call(self, *args, **kwds):
        """拼接知识体系内容
        """
        ret = ""
        bm = args[0]
        if bm.env:
            ret = self.__concat(ret, self.tpe_env, bm.env)
        if bm.scene:
            ret = self.__concat(ret, self.tpe_scene, bm.scene)
        if bm.role:
            ret = self.__concat(ret, self.tpe_role, bm.role)
        if bm.plan: 
            plan = PLAN_MAP.get(bm.plan, f"当前意图{bm.plan}无对应流程")
            ret = self.__concat(ret, self.tpe_plan, plan)
        return ret

if __name__ == "__main__":
    pe = promptEngine()
    parser = HfArgumentParser((baseMsg))
    args = parser.parse_args()
    pe._call(args)