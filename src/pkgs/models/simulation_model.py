# -*- encoding: utf-8 -*-
"""
@Time    :   2025-03-10
@desc    :   仿真模型实现
"""

import asyncio
import json
from typing import Any, Dict, Union, Generator, List

import httpx

from src.prompt.model_init import acallLLM, callLLM
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.langfuse_prompt_manager import LangfusePromptManager
from src.utils.parameter_fetcher import ParameterFetcher


class SimulationModel:
    """一个简单的仿真模型框架，提供与 LLM 交互的基本能力"""

    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr = gsr
        self.parameter_fetcher = ParameterFetcher(
            api_key=self.gsr.api_key,
            api_secret=self.gsr.api_secret,
            host=self.gsr.host,
            api_endpoints=self.gsr.api_endpoints,
        )
        self.langfuse_prompt_manager = LangfusePromptManager(
            langfuse_client=self.gsr.langfuse_client,
            prompt_meta_data=self.gsr.prompt_meta_data,
        )
        self.regist_aigc_functions()

    async def aaigc_functions_general(
        self,
        _event: str = "",
        prompt_vars: Dict[str, Any] | None = None,
        model_args: Dict[str, Any] | None = None,
        prompt_template: str = "",
        **kwargs,
    ) -> Union[str, Generator]:
        """通用生成函数，用于快速调用模型"""

        prompt_vars = prompt_vars or {}
        model_args = model_args or {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
        }

        event = kwargs.get("intentCode", _event)
        model = self.gsr.get_model(event)

        if prompt_template:
            prompt = prompt_template.format(**prompt_vars)
        else:
            prompt = await self.langfuse_prompt_manager.get_formatted_prompt(
                event, prompt_vars
            )

        history = [{"role": "system", "content": prompt}]
        logger.debug(f"[SimulationModel] LLM input: {prompt}")

        content = await acallLLM(model=model, history=history, **model_args)
        logger.info(f"[SimulationModel] LLM output: {content}")
        return content

    async def call_function(self, **kwargs) -> Union[str, Generator]:
        intent_code = kwargs.get("intentCode")
        intent_code = (
            self.gsr.intent_aigcfunc_map.get(intent_code)
            if self.gsr.intent_aigcfunc_map.get(intent_code)
            else intent_code
        )
        if not self.funcmap.get(intent_code):
            raise RuntimeError(f"Code {intent_code} not supported")

        func = self.funcmap[intent_code]
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            return func(**kwargs)
        except Exception as e:
            logger.exception(f"call_function {intent_code} error: {e}")
            raise e

    def regist_aigc_functions(self) -> None:
        self.funcmap: Dict[str, Any] = {}
        for name in dir(self):
            if name.startswith("aigc_functions_") and callable(getattr(self, name)):
                self.funcmap[name] = getattr(self, name)

    async def aigc_functions_demo(self, **kwargs) -> str:
        """基础函数：简单回显输入内容"""
        text = kwargs.get("text", "你好")
        prompt_vars = {"text": text}
        prompt_template = "请仿真回答以下内容：{text}"
        return await self.aaigc_functions_general(
            _event="simulation_demo",
            prompt_vars=prompt_vars,
            prompt_template=prompt_template,
            **kwargs,
        )

    async def aigc_functions_disease_diagnosis(self, **kwargs) -> List[Dict[str, Any]]:
        """现患疾病诊断：从 kwargs 获取参数并调用规则引擎"""
        rule_url = f"{self.gsr.host}/rule-engine/rule/execute"
        input_data = kwargs
        results: List[Dict[str, Any]] = []

        for rule in self.gsr.simulation_rule_base:
            rule_raw = rule.get("rule")
            if not rule_raw:
                continue

            try:
                # 首先转义 \n, \uXXXX 等字符为原始字符
                rule_unicode = rule_raw.encode().decode("unicode_escape")

                # 然后将乱码编码成 bytes（latin1 不丢数据），再用 utf-8 解码成中文
                rule_content = rule_unicode.encode("latin1").decode("utf-8")
            except Exception as e:
                logger.warning(f"Rule decode failed: {e}, fallback to raw")
                rule_content = rule_raw

            payload = {"input": input_data, "ruleContent": rule_content}

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(rule_url, json=payload, timeout=30)

                if resp.status_code == 200:
                    try:
                        json_data = resp.json()
                        data = json_data.get("data") or {}
                        patient = data.get("patient") or {}
                        result = patient.get("result")
                        if result:
                            results.append(result)
                    except Exception as parse_err:
                        logger.warning(
                            f"Failed to parse rule engine response: {parse_err}, raw: {resp.text}"
                        )
                else:
                    logger.error(
                        f"Rule engine call failed: {resp.status_code} → {resp.text}"
                    )

            except Exception as e:
                logger.exception(f"Exception while calling rule engine: {e}")

        return results

    def aigc_functions_weight_status_evaluation(self, **kwargs) -> str:
        """体重状态评估"""
        age: int = kwargs.get("age")
        gender: str = kwargs.get("gender")
        height: float = kwargs.get("height")
        weight: float = kwargs.get("weight")
        body_fat: float | None = kwargs.get("bodyFatRate")
        if None in (age, gender, height, weight):
            raise ValueError("age, gender, height and weight are required")

        bmi = weight / ((height / 100) ** 2)
        status = ""
        if age >= 18:
            if (age <= 64 and bmi < 18.5) or (age > 64 and bmi < 20):
                status = "体重不足"
            elif (age <= 64 and 18.5 <= bmi < 24) or (age > 64 and 20 <= bmi < 26.9):
                status = "正常"
            elif (age <= 64 and 24 <= bmi < 28) or (age > 64 and 26.9 <= bmi < 28):
                status = "超重"
            elif 28 <= bmi < 30:
                status = "轻度肥胖"
            elif 30 <= bmi < 40:
                status = "中度肥胖"
            else:
                status = "重度肥胖" if bmi >= 40 else status
        else:
            raise ValueError("age must be >=18")

        if status == "正常" and body_fat is not None:
            hidden = False
            if gender == "男":
                if age < 30:
                    hidden = body_fat >= 21
                else:
                    hidden = body_fat >= 22
            elif gender == "女":
                if age < 30:
                    hidden = body_fat >= 24
                else:
                    hidden = body_fat >= 27
            if hidden:
                status = "隐性肥胖可能"
        return status

    def aigc_functions_abdominal_obesity_evaluation(self, **kwargs) -> str:
        """腹型肥胖评估"""
        gender: str = kwargs.get("gender")
        waist: float = kwargs.get("waistline")
        hip: float = kwargs.get("hipline")
        weight_status: str = kwargs.get("weightStatus")
        if None in (gender, waist, hip, weight_status):
            raise ValueError("missing required fields")
        whr = waist / hip
        if gender == "男":
            waist_limit = 90
            whr_limit = 0.90
        else:
            waist_limit = 80
            whr_limit = 0.85
        if whr >= whr_limit or waist >= waist_limit:
            return "中心型肥胖"
        if whr < whr_limit and waist >= waist_limit:
            return "外周型肥胖"
        if weight_status in {"轻度肥胖", "中度肥胖", "重度肥胖"} and waist < waist_limit and whr < whr_limit:
            return "均匀型肥胖"
        return "均匀型肥胖"

    def aigc_functions_bmr_calculation(self, **kwargs) -> Dict[str, Any]:
        """基础代谢计算"""
        age: int = kwargs.get("age")
        gender: str = kwargs.get("gender")
        height: float = kwargs.get("height")
        weight: float = kwargs.get("weight")
        if None in (age, gender, height, weight):
            raise ValueError("age, gender, height and weight are required")
        if gender == "男":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        if age < 30:
            standard = 15.3 * weight + 679 if gender == "男" else 14.7 * weight + 496
        else:
            standard = 11.6 * weight + 879 if gender == "男" else 8.7 * weight + 820
        evaluation = "达标" if bmr >= standard else "未达标"
        return {"bmr": round(bmr, 1), "bmrEvaluation": evaluation}

    def aigc_functions_pal_energy_cost(self, **kwargs) -> float:
        """日常体力活动消耗"""
        bmr: float = kwargs.get("bmr")
        pal_category: str = kwargs.get("palCategory")
        if bmr is None or pal_category is None:
            raise ValueError("bmr and palCategory are required")
        pal_map = {"极轻": 1.2, "轻": 1.45, "中": 1.65, "高": 1.85, "极高": 2.2}
        if pal_category not in pal_map:
            raise ValueError("invalid palCategory")
        pal = pal_map[pal_category]
        return round(bmr * (pal - 1), 1)
