# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-20 17:17:57
@desc    :   健康模块功能实现
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

import inspect
import asyncio
import json
import json5
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Generator, Dict, Union, List, Literal, AsyncGenerator
from src.utils.module import (
    check_aigc_functions_body_fat_weight_management_consultation, check_and_calculate_bmr,
    calculate_standard_weight, convert_meal_plan_to_text,
    curr_time, determine_recent_solar_terms, format_historical_meal_plans_v2,
    generate_daily_schedule, generate_key_indicators, check_consecutive_days, get_festivals_and_other_festivals,
    get_weather_info, parse_generic_content, handle_calories, run_in_executor, log_with_source,
    determine_weight_status, determine_body_fat_status, truncate_to_limit, get_highest_data_per_day,
    filter_user_profile, prepare_question_list, match_health_label, enrich_meal_items_with_images,
    calculate_and_format_diet_plan, format_historical_meal_plans, query_course, call_mem0_search_memory,
    format_mem0_search_result, call_mem0_get_all_memories, convert_dict_to_key_value_section, strip_think_block,
    convert_structured_kv_to_prompt_dict, convert_schedule_fields_to_english, enrich_schedule_with_extras,
    extract_daily_schedule, export_all_lessons_with_actions, format_key_indicators, format_meals_info,
    format_intervention_plan, get_daily_key_bg, map_diet_analysis, format_meals_info_v2, format_warning_indicators,
    get_upcoming_exercise_schedule, parse_generic_content_sync
)
from data.test_param.test import testParam
from src.prompt.model_init import acallLLM, acallLLtrace, callLLM
from src.utils.Logger import logger
from src.utils.api_protocal import *
from src.utils.resources import InitAllResource
from src.utils.langfuse_prompt_manager import LangfusePromptManager
from src.utils.parameter_fetcher import ParameterFetcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from Levenshtein import ratio
from data.expert_knowledge_prompt import *


class HealthExpertModel:
    def __init__(self, gsr: InitAllResource) -> None:
        # 初始化实例属性
        self.gsr = gsr
        self.parameter_fetcher = ParameterFetcher(
            api_key=self.gsr.api_key,
            api_secret=self.gsr.api_secret,
            host=self.gsr.host,
            api_endpoints=self.gsr.api_endpoints
        )
        self.langfuse_prompt_manager = LangfusePromptManager(
            langfuse_client=self.gsr.langfuse_client,
            prompt_meta_data=self.gsr.prompt_meta_data,
        )
        self.regist_aigc_functions()

    # async def aaigc_functions_general(
    #     self,
    #     _event: str = "",
    #     prompt_vars: dict = {},
    #     model_args: Dict = {},
    #     prompt_template: str = "",
    #     **kwargs,
    # ) -> Union[str, Generator]:
    #     """通用生成"""
    #     event = kwargs.get("intentCode")
    #     model = self.gsr.get_model(event)
    #     model_args: dict = (
    #         {
    #             "temperature": 0,
    #             "top_p": 1,
    #             "repetition_penalty": 1.0,
    #         }
    #         if not model_args
    #         else model_args
    #     )
    #     prompt_template: str = (
    #         prompt_template
    #         if prompt_template
    #         else self.gsr.get_event_item(event)["description"]
    #     )
    #     logger.debug(f"Prompt Vars Before Formatting: {repr(prompt_vars)}")
    #
    #     prompt = prompt_template.format(**prompt_vars)
    #     logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")
    #
    #     content: Union[str, Generator] = await a(
    #         model=model,
    #         query=prompt,
    #         **model_args,
    #     )
    #     if isinstance(content, str):
    #         logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
    #     return content

    async def aaigc_functions_general(
            self,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: Dict = {},
            prompt_template: str = "",
            **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""

        extra_params = {
            "name": _event,
            "user_id": kwargs.get("user_id", "default"),
            "session_id": kwargs.get("session_id", "default"),
            "release": "v1.0.0",
            "tags": ["AIGC", "health-module", _event],  # 添加 Tags 便于分类追踪
            "metadata": {
                "environment": kwargs.get("environment", "production"),
                "version": kwargs.get("version", "v1.0.0"),
                "description": f"Processing event {_event}"
            },
            "langfuse": self.gsr.langfuse_client,
            "tokenizer": self.gsr.qwen_tokenizer
        }
        # 获取模型及配置
        event = kwargs.get("intentCode")
        model = self.gsr.get_model(event)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        logger.debug(f"Prompt Vars Before Formatting: {repr(prompt_vars)}")

        # 格式化 prompt
        if prompt_template:
            try:
                prompt = prompt_template.format(**prompt_vars)
            except KeyError as e:
                return f"Error: Missing placeholder for {e} in prompt_vars."
        else:
            # 使用 LangfusePromptManager 获取并格式化
            prompt = await self.langfuse_prompt_manager.get_formatted_prompt(event, prompt_vars)

        logger.debug(f"AIGC Functions {_event} LLM Input: {(prompt)}")
        his = [{
            'role': 'system',
            'content': prompt
        }]
        content: Union[str, Generator] = await acallLLtrace(
            model=model,
            history=his,
            extra_params=extra_params,
            **model_args
        )

        logger.info(f"AIGC Functions {_event} LLM Output: {(content)}")

        return content

    async def __compose_user_msg__(
            self,
            mode: Literal[
                "user_profile",
                "user_profile_new",
                "messages",
                "drug_plan",
                "medical_records",
                "ietary_guidelines",
                "key_indicators",
            ],
            user_profile: UserProfile = None,
            medical_records: MedicalRecords = None,
            ietary_guidelines: DietaryGuidelinesDetails = None,
            messages: List[ChatMessage] = [],
            key_indicators: "List[KeyIndicators]" = "[]",
            drug_plan: "List[DrugPlanItem]" = "[]",
            role_map: Dict = {},
    ) -> str:
        content = ""
        if mode == "user_profile":
            if user_profile:
                for key, value in user_profile.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "user_profile_new":
            if user_profile:
                for key, value in user_profile.items():
                    if value and USER_PROFILE_KEY_MAP_SANJI.get(key):
                        content += f"{USER_PROFILE_KEY_MAP_SANJI[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "messages":
            assert messages is not None, "messages can't be None"
            assert messages is not [], "messages can't be empty list"
            role_map = (
                {"assistant": "医生", "user": "患者"} if not role_map else role_map
            )
            for message in messages:
                if message.get("role", "other") == "other":
                    content += f"other: {message['content']}\n"
                elif role_map.get(message.get("role", "other")):
                    content += f"{role_map[message['role']]}: {message['content']}\n"
                else:
                    content += f"{message['content']}\n"
        elif mode == "drug_plan":
            if drug_plan:
                for item in json5.loads(drug_plan):
                    content += (
                            ", ".join(
                                [
                                    f"{USER_PROFILE_KEY_MAP.get(k)}: {v}"
                                    for k, v in item.items()
                                ]
                            )
                            + "\n"
                    )
                content = content.strip()
        elif mode == "medical_records":
            if medical_records:
                for key, value in medical_records.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "ietary_guidelines":
            if ietary_guidelines:
                for key, value in ietary_guidelines.items():
                    if value and DIETARY_GUIDELINES_KEY_MAP.get(key):
                        content += f"{DIETARY_GUIDELINES_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "key_indicators":
            # 创建一个字典来存储按日期聚合的数据
            aggregated_data = {}

            # 遍历数据并聚合
            for item in key_indicators:
                for entry in item["data"]:
                    date, time = entry["time"].split(" ")
                    value = entry["value"]
                    if date not in aggregated_data:
                        aggregated_data[date] = {}
                    aggregated_data[date][item["key"]] = {"time": time, "value": value}

            # 创建 Markdown 表格
            content = "| 测量日期 | 测量时间 | 体重 | BMI | 体脂率 |\n"
            content += "| ------ | ------ | ---- | ----- | ------ |\n"

            # 填充表格
            for date, measurements in aggregated_data.items():
                time = measurements.get("体重", {}).get("time", "")
                weight = measurements.get("体重", {}).get("value", "")
                bmi = measurements.get("bmi", {}).get("value", "")
                body_fat_rate = measurements.get("体脂率", {}).get("value", "")
                row = f"| {date} | {time} | {weight} | {bmi} | {body_fat_rate} |\n"
                content += row
        else:
            logger.error(f"Compose user profile error: mode {mode} not supported")
        return content

    async def __update_model_args__(self, kwargs, **args) -> Dict:
        if "model_args" in kwargs:
            if kwargs.get("model_args"):
                args = {
                    **args,
                    **kwargs["model_args"],
                }
            del kwargs["model_args"]
        return args

    def aaigc_functions_general_sync(
            self,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: Dict = {},
            prompt_template: str = "",
            **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""

        extra_params = {
            "name": _event,
            "user_id": kwargs.get("user_id", "default"),
            "session_id": kwargs.get("session_id", "default"),
            "release": "v1.0.0",
            "tags": ["AIGC", "health-module", _event],  # 添加 Tags 便于分类追踪
            "metadata": {
                "environment": kwargs.get("environment", "production"),
                "version": kwargs.get("version", "v1.0.0"),
                "description": f"Processing event {_event}"
            },
            "langfuse": self.gsr.langfuse_client,
            "tokenizer": self.gsr.qwen_tokenizer
        }
        # 获取模型及配置
        event = kwargs.get("intentCode")
        model = self.gsr.get_model(event)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        logger.debug(f"Prompt Vars Before Formatting: {repr(prompt_vars)}")

        # 格式化 prompt
        if prompt_template:
            try:
                prompt = prompt_template.format(**prompt_vars)
            except KeyError as e:
                return f"Error: Missing placeholder for {e} in prompt_vars."
        else:
            # 使用 LangfusePromptManager 获取并格式化
            prompt = self.langfuse_prompt_manager.get_formatted_prompt_sync(event, prompt_vars)

        logger.debug(f"AIGC Functions {_event} LLM Input: {(prompt)}")
        his = [{
            'role': 'system',
            'content': prompt
        }]
        content: Union[str, Generator] = callLLM(
            model=model,
            history=his,
            extra_params=extra_params,
            **model_args
        )

        logger.info(f"AIGC Functions {_event} LLM Output: {(content)}")

        return content

    def __compose_user_msg_sync__(
            self,
            mode: Literal[
                "user_profile",
                "user_profile_new",
                "messages",
                "drug_plan",
                "medical_records",
                "ietary_guidelines",
                "key_indicators",
            ],
            user_profile: UserProfile = None,
            medical_records: MedicalRecords = None,
            ietary_guidelines: DietaryGuidelinesDetails = None,
            messages: List[ChatMessage] = [],
            key_indicators: "List[KeyIndicators]" = "[]",
            drug_plan: "List[DrugPlanItem]" = "[]",
            role_map: Dict = {},
    ) -> str:
        content = ""
        if mode == "user_profile":
            if user_profile:
                for key, value in user_profile.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "user_profile_new":
            if user_profile:
                for key, value in user_profile.items():
                    if value and USER_PROFILE_KEY_MAP_SANJI.get(key):
                        content += f"{USER_PROFILE_KEY_MAP_SANJI[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "messages":
            assert messages is not None, "messages can't be None"
            assert messages is not [], "messages can't be empty list"
            role_map = (
                {"assistant": "医生", "user": "患者"} if not role_map else role_map
            )
            for message in messages:
                if message.get("role", "other") == "other":
                    content += f"other: {message['content']}\n"
                elif role_map.get(message.get("role", "other")):
                    content += f"{role_map[message['role']]}: {message['content']}\n"
                else:
                    content += f"{message['content']}\n"
        elif mode == "drug_plan":
            if drug_plan:
                for item in json5.loads(drug_plan):
                    content += (
                            ", ".join(
                                [
                                    f"{USER_PROFILE_KEY_MAP.get(k)}: {v}"
                                    for k, v in item.items()
                                ]
                            )
                            + "\n"
                    )
                content = content.strip()
        elif mode == "medical_records":
            if medical_records:
                for key, value in medical_records.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "ietary_guidelines":
            if ietary_guidelines:
                for key, value in ietary_guidelines.items():
                    if value and DIETARY_GUIDELINES_KEY_MAP.get(key):
                        content += f"{DIETARY_GUIDELINES_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "key_indicators":
            # 创建一个字典来存储按日期聚合的数据
            aggregated_data = {}

            # 遍历数据并聚合
            for item in key_indicators:
                for entry in item["data"]:
                    date, time = entry["time"].split(" ")
                    value = entry["value"]
                    if date not in aggregated_data:
                        aggregated_data[date] = {}
                    aggregated_data[date][item["key"]] = {"time": time, "value": value}

            # 创建 Markdown 表格
            content = "| 测量日期 | 测量时间 | 体重 | BMI | 体脂率 |\n"
            content += "| ------ | ------ | ---- | ----- | ------ |\n"

            # 填充表格
            for date, measurements in aggregated_data.items():
                time = measurements.get("体重", {}).get("time", "")
                weight = measurements.get("体重", {}).get("value", "")
                bmi = measurements.get("bmi", {}).get("value", "")
                body_fat_rate = measurements.get("体脂率", {}).get("value", "")
                row = f"| {date} | {time} | {weight} | {bmi} | {body_fat_rate} |\n"
                content += row
        else:
            logger.error(f"Compose user profile error: mode {mode} not supported")
        return content

    def __update_model_args_sync__(self, kwargs, **args) -> Dict:
        if "model_args" in kwargs:
            if kwargs.get("model_args"):
                args = {
                    **args,
                    **kwargs["model_args"],
                }
            del kwargs["model_args"]
        return args

    async def call_function(self, **kwargs) -> Union[str, Generator]:
        """调用函数
        - Args:
            intentCode (str): 意图代码
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        intent_code = kwargs.get("intentCode")
        # TODO intentCode -> funcCode
        intent_code = (
            self.gsr.intent_aigcfunc_map.get(intent_code)
            if self.gsr.intent_aigcfunc_map.get(intent_code)
            else intent_code
        )
        if not self.funcmap.get(intent_code):
            logger.error(f"intentCode {intent_code} not found in funcmap")
            raise RuntimeError(f"Code not supported.")

        try:
            func = self.funcmap.get(intent_code)
            if asyncio.iscoroutinefunction(func):
                content = await func(**kwargs)
            else:
                content = func(**kwargs)
        except Exception as e:
            logger.exception(f"call_function {intent_code} error: {e}")
            raise e
        return content

    def regist_aigc_functions(self) -> None:
        self.funcmap = {}
        for obj_str in dir(self):
            if obj_str.startswith("aigc_functions_") and not self.funcmap.get(obj_str):
                self.funcmap[obj_str] = getattr(self, obj_str)

    # @param_check(check_params=["messages"])
    async def aigc_functions_diagnosis_generation(self, **kwargs) -> str:
        """西医决策-诊断生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxioksess0fecv6j3aq

        _event = "西医决策-诊断生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["user_profile", "messages", "medical_records"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "medical_records": medical_records,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_chief_complaint_generation(self, **kwargs) -> str:
        """西医决策-主诉生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxiol879po3hki6uxyd

        _event = "西医决策-主诉生成"

        # 验证必填字段
        if not kwargs.get("messages"):
            raise ValueError("参数 'messages' 为必填项")

        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {"messages": messages}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_present_illness(self, **kwargs) -> str:
        """西医决策-现病史生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxiolopcya0m20232q

        _event = "西医决策-现病史生成"

        # 验证必填字段
        if not kwargs.get("messages"):
            raise ValueError("参数 'messages' 为必填项")

        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {"messages": messages}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_past_medical_history(self, **kwargs) -> str:
        """西医决策-既往史生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxiomgvu65nnylw8on6

        _event = "西医决策-既往史生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        if kwargs.get("user_profile"):
            # 检查past_history_of_present_illness是否为空
            if not kwargs.get("user_profile", {}).get("past_history_of_present_illness"):
                raise ValueError("user_profile中必须包含past_history_of_present_illness")

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {"user_profile": user_profile, "messages": messages}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    async def aigc_functions_generate_allergic_history(self, **kwargs) -> str:
        """西医决策-过敏史生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxiomn8kvixfptr1whl

        _event = "西医决策-过敏史生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        if kwargs.get("user_profile"):
            # 检查 allergic_history 是否为空
            if not kwargs.get("user_profile", {}).get("allergic_history"):
                raise ValueError("user_profile 中必须包含 allergic_history")

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {"user_profile": user_profile, "messages": messages}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_medication_plan(self, **kwargs) -> str:
        """西药医嘱生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxjry3ryciljuj4ay4q

        _event = "西药医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "medical_records": medical_records,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_examination_plan(self, **kwargs) -> str:
        """检查检验医嘱生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxvjtw1xzpz73psuirk

        _event = "检查检验医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "medical_records": medical_records,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    async def aigc_functions_physician_consultation_decision_support_v2(self, **kwargs) -> str:
        """
        西医决策-医师问诊决策支持-v2

        需求文档: https://alidocs.dingtalk.com/i/nodes/ZgpG2NdyVXXRGO6gHZ5GLQKjVMwvDqPk?utm_scene=team_space&iframeQuery=anchorId%3Duu_lyfepraswhdngcyvjmp

        能力说明：

        根据用户画像、医师信息、会话记录、预问诊会话记录等信息，生成问诊问题列表。

        参数:
            kwargs (dict): 包含以下键的参数字典：
                - user_profile (dict): 用户画像（非必填）
                - physician_info (dict): 接诊医师信息（非必填）
                - messages (list): 会话记录（必填）
                - pre_consultation_records (list): 预问诊会话记录（非必填）

        返回:
            str: 生成的问诊问题列表
        """

        _event = "西医决策-医师问诊决策支持-v2"

        # 必填字段和至少需要一项的参数列表
        messages = kwargs.get("messages", "")
        if not messages:
            raise ValueError(f"messages不能为空")

        # 处理用户画像信息
        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", "")
        )

        # 处理医师信息
        physician_info_data = kwargs.get("physician_info")
        physician_info = PhysicianInfo(**physician_info_data) if physician_info_data else ""

        # 处理会话记录
        messages = await self.__compose_user_msg__("messages", messages=kwargs.get("messages", ""))

        # 处理预问诊会话记录
        pre_consultation_records = await self.__compose_user_msg__(
            "messages", messages=kwargs.get("pre_consultation_records", ""),
            role_map={"assistant": "医生智伴", "user": "患者"}
        )

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile,
            "physician_info": physician_info,
            "messages": messages,
            "pre_consultation_records": pre_consultation_records
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        # 解析生成的问诊问题
        content = await parse_generic_content(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sjkyn_guideline_generation(self, **kwargs) -> str:
        """
        三济康养方案总则

        需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncvz122LqV3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxllc1eq0ukitru9tjf

        根据用户画像和病历信息生成康养方案总则。

        参数:
            kwargs (dict): 包含用户画像和病历信息的参数字典

        返回:
            str: 生成的康养方案总则内容
        """

        _event = "三济康养方案总则"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age",
                "gender",
                "height",
                "weight",
                "bmi",
                "current_diseases",
            ]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        # 如果提供了 user_profile，则检查 required_fields 是否完整
        if kwargs.get("user_profile"):
            user_profile = kwargs["user_profile"]
            missing_fields = [
                field for field in required_fields["user_profile"] if not user_profile.get(field)
            ]
            if missing_fields:
                raise ValueError(f"user_profile 中缺少以下必需字段: {', '.join(missing_fields)}")

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

        # 组合用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = await self.__compose_user_msg__(
            "messages", messages=kwargs.get("messages", "")
        )

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages_str,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        return content

    async def aigc_functions_test1230(self, **kwargs) -> str:
        _event = "三济康养方案"
        tag = kwargs.get("tag")

        # 构建提示变量
        prompt_vars = {
            "tag": tag
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        return content

    async def aigc_functions_dietary_guidelines_generation(self, **kwargs) -> str:
        """饮食调理原则生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncvz122LqV3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxllcmbkr26x41akllo

        _event = "饮食调理原则生成"

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = await self.__compose_user_msg__(
            "messages", messages=kwargs.get("messages", "")
        )

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages_str,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_dietary_details_generation(self, **kwargs) -> str:
        """饮食调理细则生成"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygukg03i9frbezyb5s

        _event = "饮食调理细则生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "weight_status", "daily_physical_labor_intensity",
                             "bmi", ("current_diseases", "management_goals")]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})

        # 使用集合优化检查
        missing_fields = []
        for key, fields in required_fields.items():
            if key == "user_profile":
                user_profile_keys = set(user_profile.keys())
                for field in fields:
                    if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                        if not any(f in user_profile_keys for f in field):
                            missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
                    elif field not in user_profile_keys:
                        missing_fields.append(f"缺少必填字段: {field}")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        try:
            # 使用工具类方法检查并计算基础代谢率（BMR）
            bmr = await check_and_calculate_bmr(user_profile)
            if bmr:
                user_profile["bmr"] = f"{bmr}kcal"
        except ValueError as e:
            pass

        # 组装用户画像
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 组合会话记录字符串
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "datetime": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
            "food_principle": food_principle,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_meal_plan_generation(self, **kwargs) -> str:
        """带量食谱-生成餐次、食物名称"""

        # 需求文档： https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygts2wqefdpc1f13ob

        kwargs["intentCode"] = "aigc_functions_meal_plan_generation"

        _event = "生成餐次、食物名称"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})

        missing_fields = []  # 存储缺失的字段
        for key, fields in required_fields.items():
            if key == "user_profile":
                user_profile_keys = set(user_profile.keys())
                for field in fields:
                    if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                        if not any(f in user_profile_keys for f in field):
                            missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
                    elif field not in user_profile_keys:
                        missing_fields.append(f"缺少必填字段: {field}")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 初始化变量
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}kcal\n"

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合会话记录字符串
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 饮食调理细则
        ietary_guidelines = await self.__compose_user_msg__(
            "ietary_guidelines", ietary_guidelines=kwargs.get("ietary_guidelines")
        )

        # 获取历史食谱
        historical_diets = await format_historical_meal_plans_v2(kwargs.get("historical_diets"))

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
            "food_principle": food_principle,
            "ietary_guidelines": ietary_guidelines,
            "historical_diets": historical_diets,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        content = await parse_generic_content(content)
        return content

    async def aigc_functions_generate_food_quality_guidance(self, **kwargs) -> str:
        """生成餐次、食物名称的质量指导"""

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncvz122LqV3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxlmm6en673nuggac8

        kwargs["intentCode"] = "aigc_functions_generate_food_quality_guidance"

        _event = "生成餐次、食物名称的质量指导"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                ("current_diseases", "management_goals")
            ],
            "ietary_guidelines": {"basic_nutritional_needs": ""}
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})
        ietary_guidelines = kwargs.get("ietary_guidelines", {})

        missing_fields = []  # 存储缺失的字段
        # 检查 user_profile 字段
        user_profile_keys = set(user_profile.keys())
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查 ietary_guidelines 字段
        for field, subfield in required_fields["ietary_guidelines"].items():
            if field not in ietary_guidelines or ietary_guidelines.get(field) == subfield:
                missing_fields.append(f"缺少必填字段: {field} -> {subfield}")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 初始化变量
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        basic_nutritional_needs = kwargs.get("ietary_guidelines").get(
            "basic_nutritional_needs"
        )

        meal_plan = await convert_meal_plan_to_text(kwargs.get("meal_plan", []))

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "basic_nutritional_needs": basic_nutritional_needs,
            "meal_plan": meal_plan,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        return content

    async def aigc_functions_sanji_plan_exercise_regimen(self, **kwargs) -> str:
        """三济康养方案-运动-运动调理原则

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncvz122LqV3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxllcdhgekhu7atk6lq

        # 能力说明

        根据用户画像如健康状态，管理目标，运动水平等，输出适合用户的运动调理原则，说明运动调理的目标和建议

        ## 参数说明
        - Args
            1. 用户画像（其中必填项: 年龄、性别、身高、体重、BMI、体力劳动强度, 非必填项: 现患疾病/管理目标）
            2. 病历
            3. 体检报告
            4. 检验/检查结果
            5. 关键指标数据

            Note: 上面5个，必须有一项

        - Result
            - 运动调理原则: String
        """
        _event = "三济康养方案-运动-运动调理原则"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                ("current_diseases", "management_goals")
            ]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})

        missing_fields = []  # 存储缺失的字段
        user_profile_keys = set(user_profile.keys())

        # 检查 user_profile 字段
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查至少一个字段
        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 初始化变量
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=medical_records
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_sanji_plan_exercise_plan(
            self, **kwargs
    ) -> Union[str, Generator]:
        """三济康养方案-运动-运动计划

        # 需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncvz122LqV3kdP0wQ?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxlld06f540bbnm6thv

        # 能力说明

        根据用户画像如健康状态，管理目标，运动水平等，输出适合用户的运动调理原则，说明运动调理的目标和建议

        ## 参数说明
        - Args
            1. 用户画像（其中必填项：年龄、性别、身高、体重、BMI、体力劳动强度、现患疾病或管理目标）
            2. 病历
            3. 体检报告
            4. 检验/检查结果
            5. 关键指标数据

            Note: 上面5个，必须有一项

        - Result
            - 运动计划: Dict[Dict]
        """
        _event = "三济康养方案-运动-运动计划"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                ("current_diseases", "management_goals")
            ]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})

        missing_fields = []  # 存储缺失的字段
        user_profile_keys = set(user_profile.keys())

        # 检查 user_profile 字段
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查至少有一个字段
        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 初始化变量
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=medical_records
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
            "sport_principle": kwargs.get("sport_principle", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # 输出格式是```json{}```, 需要正则提取其中的json数据
        content = await parse_generic_content(content)

        # content = query_course(self.gsr.exercise_data, "有氧热身")
        # export_all_lessons_with_actions(self.gsr.exercise_data, "exported_lessons_with_all_actions", "output.xlsx")
        return content

    async def aigc_functions_body_fat_weight_management_consultation(
            self, **kwargs
    ) -> Union[str, Generator]:
        """体脂体重管理-问诊

        需求文档: https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYpZo0qYtaj01POMVakx1Z5N?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxwurbdeo3dgi5ppwl

        # 能力说明

        对于存在体脂体重管理需求的用户，识别其体脂体重变化趋势，通过问诊能力获取更多信息。

        - Args
            1. 画像
                - 年龄（必填）
                - 性别（必填）
                - 身高（必填）
                - 疾病史（非必填）
            2. 当前日期
            3. 体重体脂记录数据:测量日期、测量时间、体重数据、体脂数据、bmi（体重、bmi必填，体脂不必填）
            4. 对话历史（非必填）
        - Return
            问题: str
        """
        _event, kwargs = "体脂体重管理-问诊", deepcopy(kwargs)
        # 参数检查
        await check_aigc_functions_body_fat_weight_management_consultation(
            kwargs
        )

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        if "messages" in kwargs and kwargs["messages"] and len(kwargs["messages"]) >= 6:
            messages = await self.__compose_user_msg__("messages", messages=kwargs["messages"],
                                                       role_map={"assistant": "健康管理师", "user": "客户"})
            kwargs["intentCode"] = "aigc_functions_body_fat_weight_management_consultation_suggestions"
            _event = "体脂体重管理-问诊-建议"
        else:
            messages = (
                await self.__compose_user_msg__("messages", messages=kwargs["messages"])
                if kwargs.get("messages")
                else ""
            )
        key_indicators = await self.__compose_user_msg__(
            "key_indicators", key_indicators=kwargs["key_indicators"]
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "key_indicators": key_indicators,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_weight_data_analysis(self, **kwargs) -> Union[str, Generator]:
        """体重数据分析

        需求文档: https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYpZo0qYtaj01POMVakx1Z5N?utm_scene=team_space&iframeQuery=anchorId%3Duu_lxyc5o9umjk7jocgfy

        分析用户上传的体重数据，提供合理建议。支持以下几种类型的能力：
        - 1日数据分析
        - 2日数据分析
        - 多日数据分析

        - Args
            1. 画像
                - 年龄（必填）
                - 性别（必填）
                - 身高（必填）
                - 疾病史（非必填）
                - 用户目标体重（非必填）
            2. 当前日期
            3. 体重记录数据:测量日期、测量时间、体重数据、bmi（体重、bmi必填）
        - Return
            建议: str
        """
        _event, kwargs = "体脂体重管理-体重数据分析", deepcopy(kwargs)
        intent_code_map = {
            1: "aigc_functions_weight_data_analysis_1day",
            2: "aigc_functions_weight_data_analysis_2day",
            3: "aigc_functions_weight_data_analysis_multiday"
        }

        # 参数检查
        await check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        # 获取并排序体重数据
        weight_data = next((item["data"] for item in kwargs.get("key_indicators", []) if item["key"] == "体重"), [])
        if not weight_data:
            raise ValueError("体重数据缺失")
        weight_data.sort(key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        # 确定事件编码
        days_count = len(weight_data)
        event_key = min(days_count, 3)
        kwargs["intentCode"] = intent_code_map[event_key]

        user_profile = kwargs["user_profile"]
        current_weight = user_profile.get("weight")
        current_bmi = user_profile.get("bmi")

        # 计算标准体重
        standard_weight = await calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 组装体重状态和目标
        weight_status, bmi_status, weight_goal = await determine_weight_status(user_profile, current_bmi)
        weight_status_goal_msg = f"当前体重为{current_weight}千克，{weight_status}，BMI{bmi_status}，需要{weight_goal}。"

        # 处理两天数据比较逻辑
        weight_change_message = ""
        if days_count == 2:
            latest_weight = float(weight_data[-1]["value"])
            previous_weight = float(weight_data[-2]["value"])
            weight_change = latest_weight - previous_weight
            weight_change_message = (
                f"与上次测量相比，最近的体重增加了{weight_change:.2f}kg。"
                if weight_change > 0
                else f"与上次测量相比，最近的体重减轻了{abs(weight_change):.2f}kg。"
                if weight_change < 0
                else "最近一次测量的体重与上次相比没有变化，保持在相同的数值。"
            )

        # 组装用户信息和关键指标字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)
        key_indicators_str = await self.__compose_user_msg__("key_indicators",
                                                             key_indicators=kwargs.get("key_indicators", None))

        # 组装提示变量并包含体重状态和目标消息
        prompt_vars = {
            "user_profile": user_profile_str,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "key_indicators": key_indicators_str,
            "weight_status_goal_msg": weight_status_goal_msg + weight_change_message
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0)

        # 调用分析函数
        content: Union[str, Generator] = await self.aaigc_functions_general(_event=_event, prompt_vars=prompt_vars,
                                                                            model_args=model_args, **kwargs)
        return content

    async def aigc_functions_body_fat_weight_data_analysis(self, **kwargs) -> Union[str, Generator]:
        """体重及体脂数据分析

        https://alidocs.dingtalk.com/i/nodes/dQPGYqjpJYpZo0qYtaj01POMVakx1Z5N?utm_scene=team_space&iframeQuery=anchorId%3Duu_lydsnxos2xr640le5ak

        分析用户上传的体重及体脂数据，提供合理建议。

        Args:
            1. 画像
                - 年龄（非必填）
                - 性别（必填）
                - 身高（非必填）
                - 疾病史（非必填）
                - 用户目标体重（非必填）
            2. 当前日期
            3. 体重及体脂记录数据:测量日期、测量时间、体重数据、体脂数据、bmi（体脂必填，体重、bmi不必填）

        Returns:
            建议: str
        """
        # 深拷贝参数以避免修改原始数据
        _event, kwargs = "体脂体重管理-体重及体脂数据分析", deepcopy(kwargs)

        # 事件代码映射
        _intentCode_map = {
            1: "aigc_functions_body_fat_weight_data_analysis_1day",
            2: "aigc_functions_body_fat_weight_data_analysis_2day",
            3: "aigc_functions_body_fat_weight_data_analysis_multiday"
        }

        key_indicators = kwargs.get("key_indicators", [])

        # 参数检查
        await check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        body_fat_data = next((item["data"] for item in key_indicators if item["key"] == "体脂率"), [])
        days_count = len(body_fat_data)
        if days_count == 0:
            raise ValueError("体脂率数据缺失")

        # 根据日期排序体脂数据
        body_fat_data.sort(key=lambda x: datetime.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        event_key = min(days_count, 3)
        kwargs["intentCode"] = _intentCode_map[event_key]

        user_profile = kwargs["user_profile"]
        current_body_fat_rate = float(body_fat_data[-1]["value"].replace('%', ''))

        # 计算标准体重
        standard_weight = await calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 计算标准体脂率
        standard_body_fat_rate = "10%-20%" if user_profile["gender"] == "男" else "15%-25%"
        user_profile["standard_body_fat_rate"] = standard_body_fat_rate

        # 组装体脂率状态和目标
        body_fat_status, body_fat_goal = await determine_body_fat_status(user_profile["gender"], current_body_fat_rate)
        body_fat_status_goal_msg = f"当前体脂率为{current_body_fat_rate}%，属于{body_fat_status}，需要{body_fat_goal}。"

        # 处理两天数据比较逻辑
        body_fat_change_message = ""
        if days_count == 2:
            latest_body_fat = float(body_fat_data[-1]["value"].replace('%', ''))
            previous_body_fat = float(body_fat_data[-2]["value"].replace('%', ''))
            body_fat_change = latest_body_fat - previous_body_fat
            body_fat_change_message = (f"最近一次的体脂率比上次测量升高{body_fat_change:.2f}%。"
                                       if body_fat_change > 0
                                       else f"最近一次的体脂率比上次测量降低{abs(body_fat_change):.2f}%。"
            if body_fat_change < 0
            else "最近一次测量的体脂率与上次相比没有变化，保持在相同的数值。")

        # 组装用户和指标信息
        user_profile_str = self.__update_model_args__("user_profile", user_profile=user_profile)
        key_indicators_str = self.__update_model_args__("key_indicators", key_indicators=kwargs["key_indicators"])

        # 准备提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "key_indicators": key_indicators_str,
            "body_fat_status_goal_msg": body_fat_status_goal_msg + body_fat_change_message
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0)

        # 调用通用函数生成内容
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        return content

    async def aigc_functions_recommended_daily_calorie_intake(self, **kwargs) -> str:
        """
        推荐每日饮食摄入热量值（B端）
        """

        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})
        knowledge_system = kwargs.get("knowledge_system")

        _event = "推荐每日饮食摄入热量值"
        kwargs["intentCode"] = "aigc_functions_recommended_daily_calorie_intake"

        # 字段校验
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                "recommended_caloric_intake", ("current_diseases", "management_goals")
            ]
        }

        missing_fields = []
        user_profile_keys = set(user_profile.keys())

        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # ✅ 分支 1：knowledge_system = "yaoshukun"
        if knowledge_system == "yaoshukun":
            _event = "姚院专项_推荐每日饮食摄入热量值"
            kwargs["intentCode"] = "aigc_functions_recommended_daily_calorie_intake_yaoshukun"

            prompt_vars = {
                "user_profile": await self.__compose_user_msg__("user_profile", user_profile=user_profile),
                "messages": await self.__compose_user_msg__(
                    "messages",
                    messages=kwargs.get("messages", ""),
                    role_map={"assistant": "assistant", "user": "user"}
                ),
                "group": kwargs.get("group", ""),
                "glucose_data": kwargs.get("glucose_data", ""),
                "current_date": kwargs.get("current_date", "")
            }

            model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

            content: str = await self.aaigc_functions_general(
                _event=_event,
                prompt_vars=prompt_vars,
                model_args=model_args,
                **kwargs
            )

            # 处理 deepseek 输出
            content = await strip_think_block(content)
            return await parse_generic_content(content)

        # ❌ 分支 2：knowledge_system = "laikang"（直接跳过）
        elif knowledge_system == "laikang":
            return None

        # ✅ 分支 3：默认逻辑（knowledge_system 未传或为其他）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile["bmr"] = f"{bmr}kcal"

        prompt_vars = {
            "user_profile": await self.__compose_user_msg__("user_profile", user_profile=user_profile),
            "medical_records": await self.__compose_user_msg__("medical_records", medical_records=medical_records),
            "messages": await self.__compose_user_msg__(
                "messages",
                messages=kwargs.get("messages", ""),
                role_map={"assistant": "assistant", "user": "user"}
            ),
            "food_principle": kwargs.get("food_principle", "")
        }

        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)
        content: str = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        return await parse_generic_content(content)

    async def aigc_functions_recommended_macro_nutrient_ratios(self, **kwargs) -> Dict[str, List[Dict[str, float]]]:
        """
        推荐三大产能营养素能量占比（B端）

        需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygopy2yluso0n0bbce

        根据用户画像如健康状态、管理目标等信息，推荐用户每日三大产能的供能占比，包含碳水化合物、蛋白质、脂肪，供B端营养师参考和调整，方便营养师指导用户的饮食方案。

        参数:
            kwargs (dict): 包含用户画像和病历信息的参数字典

        返回:
            Dict[str, List[Dict[str, float]]]: 三大产能营养素供能占比推荐值
        """

        _event = "推荐三大产能营养素能量占比"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                "weight_status", ("current_diseases", "management_goals")
            ]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})

        missing_fields = []  # 存储缺失的字段
        user_profile_keys = set(user_profile.keys())

        # 检查 user_profile 字段
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查至少有一个字段
        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile["bmr"] = f"{bmr}kcal"

        # 组合用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__("medical_records",
                                                              medical_records=kwargs.get("medical_records", ""))

        # 组合消息字符串
        messages_str = await self.__compose_user_msg__("messages", messages=kwargs.get("messages", ""))

        # 检查并获取饮食调理原则
        food_principle = kwargs.get("food_principle", "")

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages_str,
            "medical_records": medical_records_str,
            "food_principle": food_principle
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        content = await parse_generic_content(content)
        return content

    async def aigc_functions_recommended_meal_plan(self, **kwargs) -> Dict[str, List[Dict[str, float]]]:
        """
        推荐餐次及每餐能量占比（B端）

        需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygoqgmqzah57kz9t6

        根据用户画像如健康状态、管理目标等信息，推荐用户一天餐次安排及每餐能量占比，供B端营养师参考和调整，方便营养师指导用户的饮食方案。

        参数:
            kwargs (dict): 包含用户画像和病历信息的参数字典

        返回:
            Dict[str, List[Dict[str, float]]]: 餐次及每餐能量占比推荐值
        """

        _event = "推荐餐次及每餐能量占比"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "bmi", "daily_physical_labor_intensity",
                "weight_status", ("current_diseases", "management_goals")
            ]
        }

        # 检查必填字段
        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})

        missing_fields = []  # 存储缺失的字段
        user_profile_keys = set(user_profile.keys())

        # 检查 user_profile 字段
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查至少有一个字段
        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        try:
            # 使用工具类方法检查并计算基础代谢率（BMR）
            bmr = await check_and_calculate_bmr(user_profile)
            if bmr:
                user_profile["bmr"] = f"{bmr}kcal"
        except ValueError as e:
            pass

        # 组合用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__("medical_records",
                                                              medical_records=kwargs.get("medical_records", ""))

        # 组合消息字符串
        messages_str = await self.__compose_user_msg__("messages", messages=kwargs.get("messages", ""))

        # 检查并获取饮食调理原则
        food_principle = kwargs.get("food_principle", "")

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages_str,
            "medical_records": medical_records_str,
            "food_principle": food_principle
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        content = await parse_generic_content(content)

        return content

    async def aigc_functions_recommended_meal_plan_with_recipes(self, **kwargs) -> List[
        Dict[str, List[Dict[str, float]]]]:
        """
        推荐餐次及每餐能量占比和食谱-带量食谱

        需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygts2wqefdpc1f13ob

        根据用户健康标签、饮食调理原则，结合当前节气等信息，为用户生成合理的食谱计划，包含餐次、食物名称指导。

        参数:
            kwargs (dict): 包含用户画像和病历信息的参数字典

        返回:
            List[Dict[str, List[Dict[str, float]]]]: 一天的餐次及每餐食谱推荐值
        """

        _event = "推荐餐次及每餐能量占比和食谱"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "weight_status", ("current_diseases", "management_goals"), "age", "gender"
            ],
            "diet_plan_standards": {}
        }

        user_profile = kwargs.get("user_profile", {})
        medical_records = kwargs.get("medical_records", {})
        key_indicators = kwargs.get("key_indicators", {})

        missing_fields = []  # 存储缺失的字段
        user_profile_keys = set(user_profile.keys())

        # 检查 user_profile 字段
        for field in required_fields["user_profile"]:
            if isinstance(field, tuple):  # 如果是元组，表示其中至少需要一个字段
                if not any(f in user_profile_keys for f in field):
                    missing_fields.append(f"必须提供 {field[0]} 或 {field[1]} 中的至少一个字段")
            elif field not in user_profile_keys:
                missing_fields.append(f"缺少必填字段: {field}")

        # 检查至少有一个字段
        if not any([user_profile, medical_records, key_indicators]):
            missing_fields.append("至少提供 user_profile、medical_records 或 key_indicators 中的一个")

        # 如果有缺失字段，抛出错误
        if missing_fields:
            raise ValueError(" ".join(missing_fields))

        # 拼接用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 获取其他相关信息
        messages = kwargs.get("messages", "")
        medical_records = kwargs.get("medical_records", "")
        food_principle = kwargs.get("food_principle", "")
        ietary_guidelines = kwargs.get("ietary_guidelines", "")
        historical_diets = kwargs.get("historical_diets", "")
        diet_plan_standards = kwargs.get("diet_plan_standards", {})

        # 拼接病历信息字符串
        medical_records_str = await self.__compose_user_msg__("medical_records", medical_records=medical_records)

        # 拼接其他信息字符串
        ietary_guidelines_str = await self.__compose_user_msg__(
            "ietary_guidelines", ietary_guidelines=ietary_guidelines
        )
        historical_diets_str = await format_historical_meal_plans(historical_diets)

        diet_plan_standards_str = await calculate_and_format_diet_plan(diet_plan_standards)

        # 组合消息字符串
        messages_str = await self.__compose_user_msg__("messages", messages=messages,
                                                       role_map={"assistant": "assistant", "user": "user"})

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages_str,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "medical_records": medical_records_str,
            "food_principle": food_principle,
            "ietary_guidelines": ietary_guidelines_str,
            "historical_diets": historical_diets_str,
            "diet_plan_standards": diet_plan_standards_str
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        content = await parse_generic_content(content)
        items = await run_in_executor(
            lambda: enrich_meal_items_with_images(content, self.gsr.dishes_data, 0.5, MEAL_TIME_MAPPING))
        if not items:
            items = DEFAULT_MEAL_PLAN
        return items

    async def aigc_functions_generate_related_questions(self, **kwargs) -> List[str]:
        """
        猜你想问

        需求文档：<https://alidocs.dingtalk.com/i/nodes/amweZ92PV6BD4ZlzHvbOD3xzVxEKBD6p?utm_scene=team_space&iframeQuery=anchorId%3Duu_lyy3y0qzqkiszfgupfe>

        根据用户的提问和用户画像，生成三个用户还可能提问的相关问题。

        参数:
            kwargs (dict): 包含用户提问和用户画像的参数字典

        返回:
            List[str]: 用户可能提问的三个相关问题
        """
        _event = "猜你想问"
        kwargs = deepcopy(kwargs)

        # 获取用户提问信息
        user_question = kwargs.get("user_question", "")
        user_messages = kwargs.get("messages", "")

        prompt_parts = []

        if user_question:
            prompt_parts.append(f"## 用户问题\n{user_question}")

        if user_messages:
            # 处理会话记录
            messages = await self.__compose_user_msg__("messages", messages=user_messages,
                                                       role_map={"assistant": "Assistant", "user": "User"})
            prompt_parts.append(f"## 用户会话记录\n{messages}")

        prompt_message = "\n".join(prompt_parts) if prompt_parts else ""

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", "")

        # 拼接用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        prompt_vars = {
            "messages": prompt_message,
            "user_profile": user_profile_str,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)

        # 新增逻辑：检查并调整问号数量
        def format_question(question: str) -> str:
            # 替换句中非句尾的问号（包括中文问号和英文问号）为逗号，只保留句尾问号
            question = re.sub(r"[？?](?!$)", "，", question)  # 替换句中问号为逗号
            if not question.endswith("？") and not question.endswith("?"):
                question += "？"  # 如果句末没有问号（中文或英文），添加一个中文问号
            return question

        # 格式化每个问题，确保符合要求
        formatted_content = [format_question(question) for question in content]

        return formatted_content

    async def aigc_functions_generate_greeting(self, **kwargs) -> str:
        """
        生成每日问候开场白

        需求文档：<https://alidocs.dingtalk.com/i/nodes/amweZ92PV6BD4ZlzHvbOD3xzVxEKBD6p?utm_scene=team_space&iframeQuery=anchorId%3Duu_lyz717o1dw4tnoeoiqa>

        根据用户画像、当日剩余日程、关键指标、当日相关信息等生成每日问候开场白。

        参数:
            kwargs (dict): 包含用户画像、当日剩余日程、关键指标、当日相关信息的参数字典

        返回:
            str: 每日问候开场白文本
        """

        _event = "生成每日问候开场白"

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})
        city = user_profile.get("city", "")

        # 获取当日剩余日程信息
        daily_schedule = kwargs.get("daily_schedule", [])
        daily_schedule_str = await generate_daily_schedule(daily_schedule)
        daily_schedule_section = f"## 当日剩余日程\n{daily_schedule_str}" if daily_schedule_str else ""

        # 获取关键指标信息
        key_indicators = kwargs.get("key_indicators", [])
        key_indicators_str = await generate_key_indicators(key_indicators)
        key_indicators_section = f"## 关键指标\n{key_indicators_str}" if key_indicators_str else ""

        # 异步获取当天天气信息
        today_weather = await run_in_executor(lambda: get_weather_info(self.gsr.weather_api_config, city)
                                              )

        # 获取最近节气
        recent_solar_terms = await determine_recent_solar_terms()

        # 获取当日节日
        today_festivals = await get_festivals_and_other_festivals()

        # 构建当日相关信息
        daily_info = [f"### 当前日期和时间\n{curr_time()}"]
        if today_weather:
            daily_info.append(f"### 当日天气\n{today_weather}")
        if recent_solar_terms:
            daily_info.append(f"### 最近节气\n{recent_solar_terms}")
        if today_festivals:
            daily_info.append(f"### 当日节日\n{today_festivals}")

        daily_info_str = "\n".join(daily_info).strip()

        # 拼接用户画像信息字符串
        # user_profile_str = self.__compose_user_msg__("user_profile", user_profile=user_profile)
        # user_profile_section = f"## 用户画像\n{user_profile_str}" if user_profile_str else ""

        # 构建提示变量
        prompt_vars = {
            "daily_schedule": daily_schedule_section,
            "key_indicators": key_indicators_section,
            "daily_info": daily_info_str
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_generate_greeting_new(self, **kwargs) -> str:
        """
        生成每日问候开场白

        需求文档：<https://alidocs.dingtalk.com/i/nodes/amweZ92PV6BD4ZlzHvbOD3xzVxEKBD6p?utm_scene=team_space&iframeQuery=anchorId%3Duu_lyz717o1dw4tnoeoiqa>

        根据用户画像、当日剩余日程、关键指标、当日相关信息等生成每日问候开场白。

        参数:
            kwargs (dict): 包含用户画像、当日剩余日程、关键指标、当日相关信息的参数字典

        返回:
            str: 每日问候开场白文本
        """

        _event = "生成每日问候开场白新版"

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})
        city = user_profile.get("city", "")

        # 获取当日剩余日程信息
        daily_schedule = kwargs.get("daily_schedule", [])
        daily_schedule_str = await generate_daily_schedule(daily_schedule)
        daily_schedule_section = f"## 当日剩余日程\n{daily_schedule_str}" if daily_schedule_str else ""

        # 获取关键指标信息
        key_indicators = kwargs.get("key_indicators", [])
        key_indicators_str = await generate_key_indicators(key_indicators)
        key_indicators_section = f"## 关键指标\n{key_indicators_str}" if key_indicators_str else ""

        # 异步获取当天天气信息
        today_weather = await run_in_executor(lambda: get_weather_info(self.gsr.weather_api_config, city)
                                              ) if city else None

        # 获取最近节气
        recent_solar_terms = await determine_recent_solar_terms()

        # 获取当日节日
        today_festivals = await get_festivals_and_other_festivals()

        # 构建当日相关信息
        daily_info = [f"### 当前日期和时间\n{curr_time()}"]
        if today_weather:
            daily_info.append(f"### 当日天气\n{today_weather}")
        if recent_solar_terms:
            daily_info.append(f"### 最近节气\n{recent_solar_terms}")
        if today_festivals:
            daily_info.append(f"### 当日节日\n{today_festivals}")

        daily_info_str = "\n".join(daily_info).strip()

        groupSceneTag = kwargs.get("groupSceneTag", '')
        manageDays = kwargs.get("manageDays", '')
        dietStatus = kwargs.get("dietStatus", '')

        if dietStatus:
            if groupSceneTag and manageDays:
                cr = f"今天是你参与{groupSceneTag}管理服务的第{manageDays}天,昨日的饮食状态{dietStatus}"
            else:
                cr = f"你昨日的饮食状态{dietStatus}"
        elif groupSceneTag and manageDays:
            cr = f"今天是你参与{groupSceneTag}管理服务的第{manageDays}天."
        else:
            cr = ""

        if not cr:
            kwargs["intentCode"] = "aigc_functions_generate_greeting"
            return await self.aigc_functions_generate_greeting(**kwargs)

        if today_weather:
            template = f"早上好/下午好/晚上好！{cr}今天{city}天气xxx，请注意xxx。"
        else:
            template = f"早上好/下午好/晚上好！{cr}请注意xxx。"
        # 拼接用户画像信息字符串
        # user_profile_str = self.__compose_user_msg__("user_profile", user_profile=user_profile)
        # user_profile_section = f"## 用户画像\n{user_profile_str}" if user_profile_str else ""

        # 构建提示变量
        prompt_vars = {
            "daily_schedule": daily_schedule_section,
            "key_indicators": key_indicators_section,
            "daily_info": daily_info_str,
            "cr": cr,
            "template": template
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_guide_user_back_to_consultation(self, **kwargs) -> str:
        """
        问诊意图返回引导

        根据主线和支线会话记录，生成引导用户返回主线问诊的引导话术。
        需求文档：https://alidocs.dingtalk.com/i/nodes/YndMj49yWjwlLv9jfrYkQQOBV3pmz5aA?cid=56272080423&utm_source=im&utm_scene=team_space&iframeQuery=utm_medium%3Dim_card%26utm_source%3Dim&utm_medium=im_group_card&corpId=ding5aaad5806ea95bd7ee0f45d8e4f7c288

        参数:
            kwargs (dict): 包含以下键的参数字典：
                - main_conversation_history (list): 主线会话记录（必填）
                - branch_conversation_history (list): 支线会话记录（必填）

        返回:
            str: 生成的引导话术文本
        """

        _event = "问诊意图返回引导"

        main_conversation_history = kwargs.get("main_conversation_history")
        branch_conversation_history = kwargs.get("branch_conversation_history")

        if not main_conversation_history or not branch_conversation_history:
            raise ValueError("主线会话记录和支线会话记录均不能为空")

        # 组装会话历史
        main_conversation_history = await self.__compose_user_msg__("messages", messages=main_conversation_history)
        branch_conversation_history = await self.__compose_user_msg__("messages", messages=branch_conversation_history)

        # 构建提示变量
        prompt_vars = {
            "main_conversation_history": main_conversation_history,
            "branch_conversation_history": branch_conversation_history
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_generate_food_calories(self, **kwargs) -> dict:
        """
        智能生成对应质量的食物热量值，单位为kcal。
        需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygukmscz3ob4lgrij7

        参数:
            kwargs (dict): 包含食物名称和食物质量的参数字典

        返回:
            dict: 对应质量的食物热量值
        """
        _event = "生成对应质量的食物热量值"

        # 获取食物名称和质量
        food_name = kwargs.get("food_name", "")
        food_quantity = kwargs.get("food_quantity", "")

        if not food_name or not food_quantity:
            raise ValueError("food_name和food_quantity不能为空")

        # 构建提示变量
        prompt_vars = {
            "food_name": food_name,
            "food_quantity": food_quantity
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        content = await parse_generic_content(content)
        content = await handle_calories(content, **kwargs)
        return content

    async def aigc_functions_bp_warning_generation(self, **kwargs) -> str:
        """血压医师端预警

        分析患者的血压数据，提供合理建议。支持以下几种类型的能力：
        - 连续5天血压数据分析
        - 非连续当前血压数据分析
        需求文档：https://alidocs.dingtalk.com/i/nodes/6LeBq413JA19EGl2t6vmyYDlJDOnGvpb?utm_medium=wiki_feed_notification&utm_source=im_SYSTEM

        - Args
            1. 患者基本情况
                - 年龄（必填）
                - 性别（必填）
                - 现用药处方（非必填）
            2. 当前血压情况
            3. 近5天血压数据（仅连续数据）
        - Return
            建议: str
        """

        _event = "血压预警生成"

        # 获取用户画像
        user_profile = kwargs.get("user_profile", {})
        med_prescription = kwargs.get("med_prescription", {})

        # 必填字段检查
        if "age" not in user_profile or "gender" not in user_profile:
            raise ValueError("缺少必填字段：年龄或性别")

        # 获取当前血压和近5天血压数据
        current_bp_data = kwargs.get("current_bp", {})
        recent_bp_data = kwargs.get("recent_bp_data", [])

        # 检查 current_bp_data 和 recent_bp_data 是否为空
        if not current_bp_data:
            raise ValueError("缺少必填字段：当前血压数据")
        if not recent_bp_data:
            raise ValueError("缺少必填字段：近5天血压数据")

        # 初始化当前血压模型
        current_bp = CurrentBloodPressure(**current_bp_data)

        # 获取最近5天每天的血压最高的一条数据
        recent_bp_data = await get_highest_data_per_day(recent_bp_data)

        # 更新每条记录的血压等级
        for record in recent_bp_data:
            record['level'] = BloodPressureRecord(**record).determine_level()
            record['sbp'] = BloodPressureRecord(**record).formatted_sbp()
            record['dbp'] = BloodPressureRecord(**record).formatted_dbp()

        # 判断5天数据是否连续
        is_consecutive = await check_consecutive_days(recent_bp_data)

        # 拼接用户画像信息字符串
        user_profile = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 拼接当前血压情况字符串
        current_bp = "|当前时间|收缩压|舒张压|单位|血压等级|\n|{date}|{sbp}|{dbp}|{unit}|{level}".format(
            date=current_bp.date, sbp=current_bp.formatted_sbp(), dbp=current_bp.formatted_dbp(),
            unit="mmHg", level=current_bp.determine_level()
        )

        # 拼接近5天连续血压数据
        recent_bp_data = "|测量时间|收缩压|舒张压|单位|血压等级|\n" + "\n".join([
            "|{date}|{sbp}|{dbp}|{unit}|{level}|".format(
                date=record['date'], sbp=record['sbp'], dbp=record['dbp'], unit="mmHg", level=record['level']
            ) for record in recent_bp_data
        ])

        # 根据连续性更新 intent_code
        if is_consecutive:
            kwargs['intentCode'] = "aigc_functions_blood_pressure_alert_continuous"
            prompt_vars_format = {
                "user_profile": user_profile,
                "med_prescription": MedPrescription(**med_prescription) if med_prescription else "",
                "current_bp": current_bp,
                "recent_bp_data": recent_bp_data
            }
        else:
            kwargs['intentCode'] = "aigc_functions_blood_pressure_alert_non_continuous"
            prompt_vars_format = {
                "user_profile": user_profile,
                "med_prescription": MedPrescription(**med_prescription) if med_prescription else "",
                "current_bp": current_bp
            }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars_format, model_args=model_args, **kwargs
        )

        return content

    async def aigc_functions_diet_recommendation_summary(
            self, **kwargs
    ) -> Union[str, Generator]:
        """生成每日饮食建议总结

        需求文档: https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aRZLvqyHLjY06mQWgN7R35y?utm_medium=wiki_feed_notification&utm_source=im_SYSTEM
        """
        kwargs = deepcopy(kwargs)

        # 提取并校验饮食分析内容
        diet_analysis = kwargs.get("diet_analysis", "")
        if not diet_analysis:
            raise ValueError("Diet analysis is required.")

        # 调用模型生成饮食分析结果的精炼总结
        diet_summary_model_output = await self.__call_model_summary__(**kwargs)

        # 截取模型输出内容并拼接最终输出结果
        truncated_summary = await truncate_to_limit(diet_summary_model_output, limit=20)
        final_summary = f"{truncated_summary}"

        return final_summary

    async def __call_model_summary__(self, **kwargs) -> str:
        """调用模型生成饮食分析结果的精炼总结"""
        diet_analysis = kwargs.get("diet_analysis", "")
        prompt_vars = {"diet_analysis": diet_analysis}
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0)

        diet_summary_output = await self.aaigc_functions_general(
            _event="饮食分析结果生成", prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return diet_summary_output

    async def aigc_functions_tcm_consultation_decision_support(self, **kwargs) -> str:
        """
        中医决策-医师问诊决策支持

        需求文档: https://alidocs.dingtalk.com/i/nodes/G1DKw2zgV26KNXoktzxNweoMVB5r9YAn?cid=2713422242%3A4589735565&utm_source=im&utm_scene=team_space&iframeQuery=utm_medium%3Dim_card%26utm_source%3Dim&utm_medium=im_single_card&corpId=ding5aaad5806ea95bd7ee0f45d8e4f7c288

        能力说明：

        根据患者提供的中医四诊信息（包括脉象采集、面象采集、舌象采集）以及其他相关信息，生成基于中医问诊的诊疗问题列表。

        参数:
            kwargs (dict): 包含以下键的参数字典：
                - user_profile (dict): 用户画像（非必填）
                - messages (list): 会话记录（必填）
                - tcm_four_diagnoses (dict): 中医四诊信息（可选填），包含脉象采集、面象采集、舌象采集信息

        返回:
            str: 生成的中医问诊问题列表
        """

        _event = "中医决策-医师问诊决策支持"

        # 获取会话记录（必填项）
        messages = kwargs.get("messages", "")
        if not messages:
            raise ValueError("messages不能为空")

        # 处理用户画像信息
        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", "")
        )
        # 拼接用户画像信息字符串
        user_profile_str = f"## 用户画像\n{user_profile}" if user_profile else ""

        # 处理会话记录
        messages = await self.__compose_user_msg__("messages", messages=kwargs.get("messages", ""))

        # 处理中医四诊信息
        tcm_four_diagnoses_data = kwargs.get("tcm_four_diagnoses", {})

        # 检查字典中是否至少有一个非空列表。如果至少有一个列表非空，则格式化输出。如果所有列表均为空，则输出为空字符串
        tcm_four_diagnoses = f"## 中医四诊\n{TcmFourDiagnoses(**tcm_four_diagnoses_data).format_tcm_diagnosis()}" if any(
            tcm_four_diagnoses_data.values()) else ""

        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "tcm_four_diagnoses": tcm_four_diagnoses,
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        # 解析生成的问诊问题
        content = await parse_generic_content(content)
        return content

    async def aigc_functions_sjkyn_guideline_generation_new(self, **kwargs) -> str:
        """
        三济康养方案总则生成与能量调理话术生成

        功能描述：
        根据用户的画像信息和病历数据生成三济康养方案总则，并附加个性化的能量调理话术。

        参数:
            kwargs (dict): 包含用户基本信息、病历信息、健康指标等的字典

        返回:
            Union[str, AsyncGenerator]: 返回字符串或异步生成器
        """

        _event = "三济康养方案总则新版"
        kwargs = deepcopy(kwargs)

        model_args = kwargs.get("model_args", {})

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", "current_diseases"],
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        if not any(kwargs.get(param) for param in at_least_one):
            raise ValueError(f"至少需要提供其中一个参数: {', '.join(at_least_one)}")

        # 获取用户基本信息
        user_profile = kwargs.get("user_profile", {})

        # 检查并确保必填字段完整
        missing_fields = [
            field for field in required_fields["user_profile"] if not user_profile.get(field)
        ]
        if missing_fields:
            raise ValueError(f"user_profile 中缺少以下必需字段: {', '.join(missing_fields)}")

        # 组合用户基本信息字符串
        user_profile_str = await self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 计算基础代谢率
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合附加信息字符串（原messages部分不变）
        additional_info_str = await self.__compose_user_msg__(
            "messages", messages=kwargs.get("messages", "")
        )

        # 构建总则生成提示变量
        prompt_vars_for_total_guideline = {
            "user_profile": user_profile_str,
            "messages": additional_info_str,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records_str,
        }

        # 更新模型参数
        generation_params_for_total_guideline = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用通用AIGC函数生成总则
        total_guideline: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars_for_total_guideline,
            model_args=generation_params_for_total_guideline, **kwargs
        )

        # 生成能量调理话术
        kwargs["intentCode"] = "aigc_functions_energy_treatment_guideline_generation"
        kwargs["model_args"] = model_args

        # 用户基本信息处理
        filtered_user_profile = await filter_user_profile(user_profile)

        # 组合过滤后的用户基本信息字符串
        filtered_user_profile_str = await self.__compose_user_msg__(
            "user_profile_new", user_profile=filtered_user_profile
        )

        # 构建能量调理话术提示变量
        prompt_vars_for_energy_guideline = {
            "user_profile": filtered_user_profile_str
        }

        # 更新模型参数
        generation_params_for_energy_guideline = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用AIGC函数生成能量调理话术
        energy_treatment_guideline: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars_for_energy_guideline,
            model_args=generation_params_for_energy_guideline, **kwargs
        )
        logger.info(f"total_guideline + energy_treatment_guideline: {total_guideline, energy_treatment_guideline}")

        kwargs["model_args"] = model_args
        # 根据是否需要流式响应返回不同类型
        if kwargs.get("model_args") and kwargs["model_args"].get("stream") is True:
            async def combined_stream():
                async for chunk in total_guideline:
                    yield chunk
                async for chunk in energy_treatment_guideline:
                    yield chunk

            return combined_stream()  # 返回异步生成器
        else:
            # 组合总则和能量调理话术为一个字符串
            combined_response = f"{total_guideline}{energy_treatment_guideline}"
            return combined_response  # 返回字符串

    async def aigc_functions_energy_treatment_detailed_generation(self, **kwargs) -> str:
        """
        生成中医调理细则

        功能描述：
        根据用户画像信息生成个性化的中医调理细则，用于精准调理身体的能量平衡。

        参数:
            kwargs (dict): 包含用户基本信息的字典

        返回:
            str: 生成的中医调理细则
        """

        _event = "中医调理细则生成"

        # 必填字段检查
        required_fields = {
            "user_profile": ["gender", "current_diseases"],
        }

        # 获取用户基本信息
        user_profile = kwargs.get("user_profile", {})
        user_profile_new = await filter_user_profile(user_profile)

        # 检查并确保必填字段完整
        missing_fields = [
            field for field in required_fields["user_profile"] if not user_profile.get(field)
        ]
        if missing_fields:
            raise ValueError(f"user_profile 中缺少以下必需字段: {', '.join(missing_fields)}")

        # 组合用户基本信息字符串
        user_profile_str = await self.__compose_user_msg__(
            "user_profile_new", user_profile=user_profile_new
        )

        # 构建中医调理细则生成提示变量
        prompt_vars_for_energy_treatment = {
            "user_profile": user_profile_str
        }

        # 更新模型参数
        generation_params_for_energy_treatment = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用AIGC函数生成中医调理细则
        energy_treatment_detailed_guideline: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars_for_energy_treatment,
            model_args=generation_params_for_energy_treatment, **kwargs
        )

        return energy_treatment_detailed_guideline

    def retrieve_answer(self, user_question: str, threshold: float = 0.8) -> Dict[
        str, Union[str, List[str], bool]]:
        """
        使用多种匹配方式（正则、编辑距离、TF-IDF）检索答案，动态调整权重和阈值

        参数:
            user_question (str): 用户的问题
            threshold (float): 匹配的最低相似度阈值（默认 0.7）

        返回:
            dict: 包含答案及猜你想问的结构
        """
        best_match = None
        highest_score = 0

        question_list = []
        question_map = {}

        # 构建问题列表及映射
        for item in self.gsr.jia_kang_bao_data:
            main_question = item.get("question", "")
            question_list.append(main_question)
            question_map[main_question] = item
            for similar_question in item.get("similar_questions", []):
                question_list.append(similar_question)
                question_map[similar_question] = item

        # 优先尝试正则匹配
        for question in question_list:
            if re.search(re.escape(user_question), question, re.IGNORECASE):
                return {
                    "answer": question_map[question].get("answer", "未找到答案"),
                    "output_guess": question_map[question].get("tags", {}).get("output_guess", False),
                    "guess_you_want": question_map[question].get("guess_you_want", []),
                }

        # 尝试编辑距离匹配
        for question in question_list:
            score = ratio(user_question.lower(), question.lower())
            if score > highest_score:
                highest_score = score
                best_match = question_map[question]

        # 如果编辑距离匹配分数达到阈值，返回结果
        if highest_score >= threshold:
            return {
                "answer": best_match.get("answer", "未找到答案"),
                "output_guess": best_match.get("tags", {}).get("output_guess", False),
                "guess_you_want": best_match.get("guess_you_want", []),
            }

        # 最后尝试 TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(question_list + [user_question])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        tfidf_highest_index = cosine_similarities.argmax()
        tfidf_highest_score = cosine_similarities[tfidf_highest_index]

        if tfidf_highest_score >= threshold:
            best_match_question = question_list[tfidf_highest_index]
            best_match_item = question_map[best_match_question]
            return {
                "answer": best_match_item.get("answer", "未找到答案"),
                "output_guess": best_match_item.get("tags", {}).get("output_guess", False),
                "guess_you_want": best_match_item.get("guess_you_want", []),
            }

        # 如果未匹配到，返回默认结果
        return None

    async def aigc_functions_jia_kang_bao_support(self, **kwargs) -> Dict:
        """
        家康宝问题检索支持（带有猜你想问的逻辑）

        参数:
            kwargs: 包含以下键的参数字典：
                - user_question (str): 用户的问题
                其他可选参数

        返回:
            dict: 包含答案和猜你想问的结构
        """
        # 获取用户问题
        user_question = kwargs.get("user_question", "")
        if not user_question:
            raise ValueError("参数 user_question 不能为空")

        # 首先尝试使用正则匹配
        answer = self.retrieve_answer(user_question)
        if answer:
            return answer

        # 如果正则未匹配，调用模型生成
        messages = await self.__compose_user_msg__(
            "messages", messages=kwargs.get("messages", ""), role_map={"assistant": "助手", "user": "客户"}
        )

        question_list = prepare_question_list(self.gsr.jia_kang_bao_data)

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        prompt_vars = {
            "question_list": question_list,
            "user_question": user_question,
            "messages": messages,
        }

        # 使用通用生成方法调用模型
        result = await self.aaigc_functions_general(
            _event="家康宝服务咨询",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )

        # 模型返回结果处理
        try:
            # 使用增强的解析方法提取 `Matched Question` 和 `Thought`
            parsed_result = self.parse_model_output(result)
            matched_question = parsed_result.get("matched_question")
            thought = parsed_result.get("thought")

            if matched_question:
                # 查找匹配问题的 ID 或直接返回答案
                return self.retrieve_answer(matched_question)

        except Exception as e:
            # 记录错误日志
            logger.error(f"解析模型输出失败: {e}")

        # 如果未匹配，返回默认结构
        return {
            "answer": None,
            "output_guess": False,
            "guess_you_want": [],
            "thought": thought
        }

    def parse_model_output(self, output: str) -> Dict[str, str]:
        """
        从模型输出中提取 `Matched Question` 和 `Thought`

        参数:
            output (str): 模型输出的完整文本

        返回:
            dict: 包含提取结果的字典，结构如下：
                {
                    "matched_question": 提取的匹配问题或 None,
                    "thought": 提取的模型思路或 None,
                }
        """
        import re

        # 定义正则模式
        matched_question_pattern = r"Matched Question: (.+)"
        thought_pattern = r"Thought: (.+)"

        # 初始化返回字典
        result = {
            "matched_question": None,
            "thought": None,
        }

        # 匹配 `Matched Question`
        matched_question_match = re.search(matched_question_pattern, output)
        if matched_question_match:
            result["matched_question"] = matched_question_match.group(1).strip()

        # 匹配 `Thought`
        thought_match = re.search(thought_pattern, output)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        return result

    async def aigc_functions_health_user_active_label(self, **kwargs) -> List[Dict[str, Union[str, int]]]:
        """
        主动模式下，从所有用户的记忆中提取健康标签（支持并发处理，含耗时日志）。

        - 自动调用 mem0 的 get_all_memories 接口。
        - 每个用户的记忆独立生成标签，并发执行，支持并发上限。
        - 每条标签结果包含 user_id。
        """
        import time
        import asyncio
        from collections import defaultdict
        from asyncio import Semaphore

        _event = "健康用户主动标签提取"
        mem0_url = self.gsr.mem0_url
        all_labels = []

        start_time = time.time()
        logger.info("[打点] 接口开始执行")

        # 1. 获取所有用户的记忆数据
        t1 = time.time()
        mem0_data = await call_mem0_get_all_memories(mem0_url)
        logger.info(f"[打点] 获取所有记忆耗时: {time.time() - t1:.2f}s")

        if not mem0_data or not mem0_data.get("all_memories"):
            logger.warning("未获取到任何用户记忆数据。")
            return []

        # 2. 按 user_id 分组
        t2 = time.time()
        user_memories = defaultdict(list)
        for item in mem0_data["all_memories"]:
            user_id = item.get("user_id")
            memory = item.get("memory")
            if user_id and memory:
                user_memories[user_id].append(memory)
        logger.info(f"[打点] 用户记忆分组耗时: {time.time() - t2:.2f}s，共有用户数: {len(user_memories)}")

        # 3. 定义并发处理函数（带并发限制）
        sem = Semaphore(10)  # 控制最多10个并发

        async def process_user_limited(user_id: str, memory_list: List[str]) -> List[Dict]:
            async with sem:
                memory_text = "\n".join(memory_list)
                prompt_vars = {"memory": memory_text}
                model_args = await self.__update_model_args__(kwargs, temperature=0.7,
                                                              top_p=1)  # 注意去掉 repetition_penalty
                try:
                    content: str = await self.aaigc_functions_general(
                        _event=_event, prompt_vars=prompt_vars, model_args=model_args, user_id=user_id, **kwargs
                    )
                    parsed = await parse_generic_content(content)
                    return [
                        {**matched_data, "user_id": user_id}
                        for item in parsed
                        for tag in item.get("tag_value", [])
                        if (matched_data := match_health_label(self.gsr.health_labels_data, item["label_name"], tag))
                    ]
                except Exception as e:
                    logger.error(f"[主动标签提取] 处理用户 {user_id} 标签失败: {e}")
                    return []

        # 4. 并发执行
        t3 = time.time()
        tasks = [process_user_limited(user_id, memories) for user_id, memories in user_memories.items()]
        results = await asyncio.gather(*tasks)
        logger.info(f"[打点] 模型调用并发耗时: {time.time() - t3:.2f}s")

        # 5. 聚合结果
        t4 = time.time()
        for label_list in results:
            all_labels.extend(label_list)
        logger.info(f"[打点] 标签聚合耗时: {time.time() - t4:.2f}s")

        logger.info(f"[打点] 接口总耗时: {time.time() - start_time:.2f}s，最终标签数: {len(all_labels)}")
        return all_labels

    async def aigc_functions_clear_inflammatory_diet(self, **kwargs) -> Union[str, List[Dict]]:
        """
        清除致炎饮食建议生成
        """
        _event = "清除致炎饮食建议生成"

        user_profile = kwargs.get("user_profile", {})

        prompt_vars = {
            "user_profile": await self.__compose_user_msg__("user_profile", user_profile=user_profile),
            "group": kwargs.get("group", ""),
            "messages": await self.__compose_user_msg__("messages", messages=kwargs.get("messages") or []),
            "glucose_data": kwargs.get("glucose_data", None),
            "questionnaire": kwargs.get("questionnaire"),
            "current_date": datetime.today().strftime("%Y-%m-%d"),
        }

        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        content = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )

        content = await parse_generic_content(content)

        image_url = IMAGE_MAP.get("清除致炎饮食", {}).get("url", "")

        content = await convert_dict_to_key_value_section(content, image=image_url)

        return content

    async def aigc_functions_emotion_mind_adjustment(self, **kwargs) -> Union[str, Dict]:
        """
        情志调理心态建议生成

        需求文档：https://alidocs.dingtalk.com/i/nodes/93NwLYZXWyqxvl7zuj4yGpabWkyEqBQm?corpId=ding5aaad5806ea95bd7ee0f45d8e4f7c288&doc_type=wiki_doc&utm_medium=search_main&utm_source=search

        根据姚树坤院长理念，为慢病用户生成情志调理建议，符合中老年表达习惯。

        参数:
            kwargs (dict): 包含用户画像、管理群组、历史信息等

        返回:
            dict: 情志调理建议（英文字段）
        """
        _event = "情志调理心态建议生成"

        user_profile = kwargs.get("user_profile", {})

        prompt_vars = {
            "user_profile": await self.__compose_user_msg__("user_profile", user_profile=user_profile),
            "group": kwargs.get("group", ""),
            "messages": await self.__compose_user_msg__("messages", messages=kwargs.get("messages") or []),
            "questionnaire": kwargs.get("questionnaire", None),
            "current_date": datetime.today().strftime("%Y-%m-%d"),
        }

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        content = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )

        content = await strip_think_block(content)

        content = await parse_generic_content(content)

        image_url = IMAGE_MAP.get("情志调理心态", {}).get("url", "")

        content = await convert_dict_to_key_value_section(content, image=image_url)

        return content

    async def aigc_functions_smoking_alcohol_control(self, **kwargs) -> Dict:
        """
        戒烟限酒建议生成

        需求文档：https://alidocs.dingtalk.com/i/nodes/93NwLYZXWyqxvl7zuj4yGpabWkyEqBQm?corpId=ding5aaad5806ea95bd7ee0f45d8e4f7c288&doc_type=wiki_doc&utm_medium=search_main&utm_source=search

        根据固定规则，结合性别和疾病情况，输出戒烟限酒建议（统一结构）。

        参数:
            kwargs (dict): 包含 user_profile 和 group 字段

        返回:
            dict: 标准结构 {
                "title": "戒烟限酒",
                "image": "...",
                "data": [
                    {"name": ..., "value": ...}
                ]
            }
        """
        _event = "戒烟限酒"

        user_profile = kwargs.get("user_profile", {})
        group = kwargs.get("group", "")

        gender = user_profile.get("gender")
        diseases = user_profile.get("current_diseases") or []

        has_high_risk_disease = (
                any(d in ["糖尿病", "脂肪肝", "肝功能异常"] for d in diseases)
                or any(risk in group for risk in ["糖尿病", "脂肪肝", "肝功能异常"])
        )

        if has_high_risk_disease:
            raw = {
                "戒烟限酒": {
                    "戒烟": "吸烟会增加血管紧张度，可以逐步减量，直至完全戒掉。",
                    "限酒与降血压显著相关": "酒精摄入量平均减少67%, 收缩压下降3.31 mmHg，舒张压下降2.04 mmHg。",
                    "严格戒酒": "任何含酒精的饮品（如红酒、白酒、啤酒、黄酒等）对人体健康都无益。糖尿病、脂肪肝或肝功能异常的人群应该严格戒酒。"
                }
            }
        else:
            if gender == "男":
                raw = {
                    "戒烟限酒": {
                        "戒烟": "吸烟会增加血管紧张度，可以逐步减量，直至完全戒掉。",
                        "限酒与降血压显著相关": "酒精摄入量平均减少67%, 收缩压下降3.31 mmHg，舒张压下降2.04 mmHg。",
                        "限酒": "男性每次不超过50g酒精，2两酒=80g酒精。3钱的酒杯，建议每次男性不超3杯（1两）。每周不超2次。",
                        "注意": "任何含酒精的饮品（如红酒、白酒、啤酒、黄酒等）对人体健康都无益。糖尿病、脂肪肝或肝功能异常的人群应该严格戒酒。"
                    }
                }
            else:
                raw = {
                    "戒烟限酒": {
                        "戒烟": "吸烟会增加血管紧张度，可以逐步减量，直至完全戒掉。",
                        "限酒与降血压显著相关": "酒精摄入量平均减少67%, 收缩压下降3.31 mmHg，舒张压下降2.04 mmHg。",
                        "限酒": "女性每次不超过30g酒精，2两酒=80g酒精。3钱的酒杯，建议每次女性不超2杯，每周不超2次。",
                        "注意": "任何含酒精的饮品（如红酒、白酒、啤酒、黄酒等）对人体健康都无益。糖尿病、脂肪肝或肝功能异常的人群应该严格戒酒。"
                    }
                }

        image_url = IMAGE_MAP.get("戒烟限酒", {}).get("url", "")

        content = await convert_dict_to_key_value_section(raw, image=image_url)

        return content

    async def aigc_functions_daily_schedule_push(self, **kwargs) -> Union[str, Dict]:
        """
        日程打卡推送内容生成

        需求文档：https://alidocs.dingtalk.com/i/nodes/93NwLYZXWyqxvl7zuj4yGpabWkyEqBQm?corpId=ding5aaad5806ea95bd7ee0f45d8e4f7c288&doc_type=wiki_doc&utm_medium=search_main&utm_source=search

        根据姚树坤理论及用户画像，生成早餐、午餐、晚餐、14点/19点运动打卡文案。

        参数:
            kwargs (dict): 包含用户画像、管理群组、健康管理参数

        返回:
            dict: 包含日程推送列表结构（含视频/图片等）
        """
        _event = "日程推送"

        user_profile = kwargs.get("user_profile", {})

        # 从结构化字段中提取各干预内容到提示词
        prompt_vars = {
            "user_profile": await self.__compose_user_msg__("user_profile", user_profile=user_profile),
            "group": kwargs.get("group"),
            "current_date": kwargs.get("current_date"),
            "diet_weight_control": await convert_structured_kv_to_prompt_dict(kwargs.get("diet_weight_control")),
            "clear_inflammatory_diet": await convert_structured_kv_to_prompt_dict(
                kwargs.get("clear_inflammatory_diet")),
            "salt_intake_control": await convert_structured_kv_to_prompt_dict(kwargs.get("salt_intake_control")),
            "moderate_exercise": await convert_structured_kv_to_prompt_dict(kwargs.get("moderate_exercise")),
        }

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        content = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )

        content = await strip_think_block(content)

        parsed = await parse_generic_content(content)

        daily_list = await extract_daily_schedule(parsed)

        daily_schedule_list = []
        for raw_item in daily_list:
            item = await convert_schedule_fields_to_english(raw_item)
            item = await enrich_schedule_with_extras(item)
            daily_schedule_list.append(item)
        if not daily_schedule_list:
            daily_schedule_list = DAILV_SCHEDULE_PUSH
        return daily_schedule_list

    async def aigc_functions_diet_eval(self, **kwargs):
        """
        饮食评估功能

        需求文档：https://alidocs.dingtalk.com/i/nodes/YQBnd5ExVEBjOmlqHMNqNnYZJyeZqMmz

        根据专家体系（laikang/yaoshukun）、管理群组和指标类型，分发到对应的饮食评估场景函数，统一处理饮食分析与饮食状态建议。

        参数:
            kwargs (dict): 请求参数，包含用户画像、饮食记录、指标数据等

        返回:
            dict: 返回包含饮食分析和饮食状态的结构
        """
        expert_system = kwargs.get("expert_system")
        manage_group = kwargs.get("manage_group")
        indicator_type = kwargs.get("key_indicators", {}).get("type")

        if expert_system == "laikang":
            if manage_group == "血糖管理":
                return await self._diet_eval_blood_sugar_laikang(**kwargs)
            elif manage_group == "血压管理":
                return await self._diet_eval_blood_pressure_laikang(**kwargs)
            elif manage_group == "减脂减重管理":
                return await self._diet_eval_weight_management_laikang(**kwargs)
        elif expert_system == "yaoshukun":
            if manage_group == "血糖管理" and indicator_type == "blood_sugar":
                return await self._diet_eval_blood_sugar_yaoshukun(**kwargs)
            elif manage_group == "血糖管理" and indicator_type == "dynamic_blood_sugar":
                return await self._diet_eval_dynamic_blood_sugar_yaoshukun(**kwargs)
            elif manage_group == "血压管理":
                return await self._diet_eval_blood_pressure_yaoshukun(**kwargs)
            elif manage_group == "减脂减重管理":
                return await self._diet_eval_weight_management_yaoshukun(**kwargs)

        raise ValueError("不支持的专家体系或管理群组/指标类型组合")

    async def _diet_eval_blood_sugar_laikang(self, **kwargs):
        """
        处理三疗体系下的血糖管理（指尖血）饮食评估。

        根据传入的用户画像、饮食记录、血糖数据等，评估当前饮食对血糖控制的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、血糖数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_blood_sugar_laikang"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))
        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_blood_pressure_laikang(self, **kwargs):
        """
        处理三疗体系下的血压管理饮食评估。

        根据传入的用户画像、饮食记录、血压数据等，评估当前饮食对血压控制的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、血压数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_blood_pressure_laikang"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))
        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_weight_management_laikang(self, **kwargs):
        """
        处理三疗体系下的减脂减重饮食评估。

        根据传入的用户画像、饮食记录、体重数据等，评估当前饮食对体重管理的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、体重数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_weight_management_laikang"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))

        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_blood_sugar_yaoshukun(self, **kwargs):
        """
        处理姚树坤体系下的血糖管理（指尖血）饮食评估。

        根据传入的用户画像、饮食记录、血糖数据等，评估当前饮食对血糖控制的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、血糖数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_blood_sugar_yaoshukun"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))

        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_dynamic_blood_sugar_yaoshukun(self, **kwargs):
        """
        处理姚树坤体系下的血糖管理（动态血糖）饮食评估。

        根据传入的用户画像、饮食记录、动态血糖数据等，评估当前饮食对动态血糖的控制影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、动态血糖数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_dynamic_blood_sugar_yaoshukun"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))

        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))
        intervention_plan = await format_intervention_plan(kwargs.get("intervention_plan"))

        glucose_analyses = await get_daily_key_bg(kwargs.get("key_indicators"), kwargs.get("meals_info"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators,
            "glucose_analyses_summary": glucose_analyses.rstrip("\n").rstrip(),
            "intervention_plan": intervention_plan
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_blood_pressure_yaoshukun(self, **kwargs):
        """
        处理姚树坤体系下的血压管理饮食评估。

        根据传入的用户画像、饮食记录、血压数据等，评估当前饮食对血压控制的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、血压数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_blood_pressure_yaoshukun"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))

        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))
        intervention_plan = await format_intervention_plan(kwargs.get("intervention_plan"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators,
            "glucose_analyses_summary": kwargs.get("glucose_analyses_summary"),
            "intervention_plan": intervention_plan
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def _diet_eval_weight_management_yaoshukun(self, **kwargs):
        """
        处理姚树坤体系下的减脂减重饮食评估。

        根据传入的用户画像、饮食记录、体重数据等，评估当前饮食对体重管理的影响，并提供相应的建议。

        参数:
            kwargs (dict): 包含用户画像、饮食记录、体重数据等信息。

        返回:
            dict: 返回饮食分析和饮食状态的建议。
        """
        kwargs["intentCode"] = "aigc_functions_diet_eval_weight_management_yaoshukun"
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=kwargs.get("user_profile", {}))

        meeals_info = await format_meals_info(kwargs.get("meals_info"))
        key_indicators = await format_key_indicators(kwargs.get("key_indicators"))
        intervention_plan = await format_intervention_plan(kwargs.get("intervention_plan"))

        # 准备提示词变量
        prompt_vars = {
            "manage_group": kwargs.get("manage_group"),
            "user_profile": user_profile_str.rstrip("\n").rstrip(),
            "current_date": f"## 当前日期：{kwargs.get('current_date')}",
            "meals_info": meeals_info,
            "key_indicators": key_indicators,
            "glucose_analyses_summary": kwargs.get("glucose_analyses_summary"),
            "intervention_plan": intervention_plan
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

        # 获取饮食评估内容
        content = await self.aaigc_functions_general(
            _event="饮食评估",
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs
        )
        content = await parse_generic_content(content)
        content = map_diet_analysis(content)
        return content

    async def aigc_functions_diet_recommendation(self, **kwargs):
        user_profile = kwargs.get("user_profile")
        height = user_profile.get("height")
        weight = user_profile.get("weight")
        disease = user_profile.get("disease")
        exercise_intensity = user_profile.get("intensity")  # 这里传数字了，比如1，2，3，4，5

        # 运动强度映射（如果后面需要用的话）
        intensity_mapping = {
            "1": "极低活动水平（久坐、很少运动）",
            "2": "轻度活动水平（周1~3次低强度运动，如散步）",
            "3": "中度活动水平（周3~5次中强度运动，如慢跑）",
            "4": "高度活动水平（周3~5次高强度运动，如健身）",
            "5": "极高活动水平（高强度体力劳动、运动员）"
        }

        # 计算 BMI
        bmi = float(weight) / (float(height) / 100) ** 2

        # 生成一句BMI描述
        if bmi < 18.5:
            bmi_desc = f"您的身体质量指数(BMI)为{bmi:.1f}。您的BMI低于正常区间，属于过轻范围。"
        elif 18.5 <= bmi < 24.9:
            bmi_desc = f"您的身体质量指数(BMI)为{bmi:.1f}。您的BMI在正常区间，属于健康范围。"
        else:
            bmi_desc = f"您的身体质量指数(BMI)为{bmi:.1f}。您的BMI高于正常区间，属于过重范围。"

        # 固定分类
        cate_list = [
            {"name": "大豆类", "code": "001"},
            {"name": "动物性食物", "code": "002"},
            {"name": "豆菜类", "code": "003"},
            {"name": "根菜类", "code": "004"},
            {"name": "菇类", "code": "005"},
            {"name": "谷物", "code": "006"},
            {"name": "果菜瓜菜类", "code": "007"},
            {"name": "果脯类", "code": "008"},
            {"name": "花菜类", "code": "009"},
            {"name": "坚果类", "code": "010"},
            {"name": "茎菜类", "code": "011"},
            {"name": "薯类", "code": "012"},
            {"name": "水果类", "code": "013"},
            {"name": "叶菜类", "code": "014"}
        ]

        # 给每个分类加推荐克数
        # for item in cate_list:
        #     item["weight"] = 100 + int(item["code"]) * 10  # 假数据，可以后面自己调整

        return {
            "bmi_desc": bmi_desc,
            "cate_list": cate_list
        }

    async def aigc_functions_generate_meal_plan(self, **kwargs):
        ingredients = kwargs.get("ingredients", [])

        # 参数检查
        if not ingredients:
            return {"message": "食材列表不能为空"}

        meal_plans = []

        for item in ingredients:
            name = item.get("name")
            num = item.get("num")

            if not name or num is None:
                continue  # 如果name或者num缺失，跳过

            meal_plans.append({
                "recipe_name": f"{name}料理",
                "method": f"准备{num}份{name}，清洗、切块，快速翻炒至熟，加调味料即可食用。",
                "image": ""  # 图片字段空着
            })

        return {
            "meal_plans": meal_plans
        }

        def get_parameters(self, **kwargs):
            """
            根据 user_id 获取用户相关信息（包括用户画像、饮食记录、群日程等），
            适应不同的场景需求，如血糖预警、更新运动日程、日程查询等。
            """
            user_id = kwargs.get("user_id")
            group_id = kwargs.get("group_id")
            start_time = kwargs.get("start_time")
            end_time = kwargs.get("end_time")
            is_new = kwargs.get("is_new", True)  # 默认为 True，按需可调整
            task_type = kwargs.get("task_type")  # 默认任务类型为血糖预警

            # 获取用户画像、饮食记录、群日程等信息
            profile_data = self.parameter_fetcher.get_user_profile(user_id)
            meals_info = self.parameter_fetcher.get_meals_info(group_id, start_time, end_time)
            group_schedule = self.parameter_fetcher.get_group_schedule(group_id, is_new)

            # 提取 expert_system（例如，血糖预警中用到）
            expert_system = profile_data.get("expert_system")

            # 动态选择 intent_code 和提示词逻辑
            if task_type == "blood_sugar_warning":
                # 血糖预警场景下的逻辑
                if expert_system == "1":
                    intent_code = "aigc_functions_blood_sugar_warning_sanliao"
                    prompt = """
    # 已知信息
    ## 用户画像
    {user_profile}
    {warning_indicators}
    {meals_info}

    {diet_comment}

    # 任务描述
    你扮演一位慢病管理专家，你负责管理的客户血糖波动超过正常值，你对该客户出具即时性的处理意见。
    # 背景描述
    - 你所在的血糖管理项目，利用慢病管理平台为入组患者提供21天血糖闭环管理服务，对患者进行线上血糖管理。
    - 患者可以在群聊中向你和你的团队同步他的饮食和运动情况，你也会给予点评和意见。
    - 患者佩戴有动态血糖仪，你可以通过慢病管理平台实时监测患者动态血糖值，患者血糖超过正常值时你会收到预警通知。

    # 处理流程
    1.如果存在3小时内的营养师点评信息，需要提及上次营养师点评的内容，肯定营养师之前的评价，对患者做健康教育。例如，可以使用“看到营养师3小时内对您饮食做出过点评，”
    2.根据上一餐饮食情况和运动情况，帮助推测患者血糖波动原因，以帮助患者纠正错误的生活习惯，建立正确的血糖管理知识。
    3.如果不存在上一餐饮食情况，则需要询问用户上一餐饮食情况。

    # 输出要求
    1.输出包含3个字段，`分析预警原因``下一餐饮食建议``运动建议`，按json格式输出。
    2.语气和态度：无需打招呼，直接输出，保持口语化、专业而友好的语气，避免使用过于医学化的术语，让患者感觉舒适和被支持。例如，可以使用“看到你X点X分血糖值有所升高，达到XXmmol/L，我们分析一下情况”这样的表达。
    3.提供明确、具体的行动建议，帮助患者立即采取措施。例如，参考当前时间，时间允许情况下可以建议患者进行15-30分钟的散步，以及在下一次餐食中具体建议，例如明天下午加餐可以把芒果换成低GI水果。
    4.安全提示：提醒患者如果感觉不适或血糖持续升高，应及时联系你。
    5.后续支持：强调患者可以随时联系你，提供持续的支持和帮助，增加患者的信任感和安全感。
    5.记录和反馈：鼓励患者充分利用动态血糖仪，感受血糖对用餐情况和运动情况的反应，以便更好地了解哪些措施有效，这有助于未来的管理和调整。
    6.如何不建议运动，`运动建议`可明确指出。
    7.总字数不超过300字。
                            """
                elif expert_system == "2":
                    intent_code = "aigc_functions_blood_sugar_warning_yaoshukun"
                    prompt = """
    # 已知信息
    ## 用户画像
    {user_profile}
    {warning_indicators}
    {meals_info}

    {diet_comment}

    # 任务描述
    你扮演一位慢病管理专家，你负责管理的客户血糖波动超过正常值，你对该客户出具即时性的处理意见。
    # 背景描述
    - 你所在的血糖管理项目，利用慢病管理平台为入组患者提供21天血糖闭环管理服务，对患者进行线上血糖管理。
    - 患者可以在群聊中向你和你的团队同步他的饮食和运动情况，你也会给予点评和意见。
    - 患者佩戴有动态血糖仪，你可以通过慢病管理平台实时监测患者动态血糖值，患者血糖超过正常值时你会收到预警通知。

    # 处理流程
    1.如果存在3小时内的营养师点评信息，需要提及上次营养师点评的内容，肯定营养师之前的评价，对患者做健康教育。例如，可以使用“看到营养师3小时内对您饮食做出过点评，”
    2.根据上一餐饮食情况和运动情况，帮助推测患者血糖波动原因，以帮助患者纠正错误的生活习惯，建立正确的血糖管理知识。
    3.如果不存在上一餐饮食情况，则需要询问用户上一餐饮食情况。

    # 输出要求
    1.输出包含3个字段，`分析预警原因``下一餐饮食建议``运动建议`，按json格式输出。
    2.语气和态度：无需打招呼，直接输出，保持口语化、专业而友好的语气，避免使用过于医学化的术语，让患者感觉舒适和被支持。例如，可以使用“看到你X点X分血糖值有所升高，达到XXmmol/L，我们分析一下情况”这样的表达。
    3.提供明确、具体的行动建议，帮助患者立即采取措施。例如，参考当前时间，时间允许情况下可以建议患者进行15-30分钟的散步，以及在下一次餐食中具体建议，例如明天下午加餐可以把芒果换成低GI水果。
    4.安全提示：提醒患者如果感觉不适或血糖持续升高，应及时联系你。
    5.后续支持：强调患者可以随时联系你，提供持续的支持和帮助，增加患者的信任感和安全感。
    5.记录和反馈：鼓励患者充分利用动态血糖仪，感受血糖对用餐情况和运动情况的反应，以便更好地了解哪些措施有效，这有助于未来的管理和调整。
    6.如何不建议运动，`运动建议`可明确指出。
    7.总字数不超过300字。
                            """
                else:
                    return {"error": "expert_system 未指定有效的值"}
            elif task_type == "update_exercise_schedule":
                # 更新运动日程的逻辑
                intent_code = "aigc_functions_update_exercise_schedule"
                prompt = """
    # 已知信息
    ## 用户画像
    {user_profile}
    {warning_indicators}
    {exercise_suggestions}
    {group_schedule}
    # 任务描述
    请你扮演一位专业的健康管理师，针对慢病用户的指标预警情况，修改用户的运动日程内容。
    # 处理流程
    1.你首先需要判断该用户的运动日程在指标预警后是否需要调整，如果不需要，返回`false`,如果你认为用户原运动方案已经不合适，需要调整，则需要思考如何调整。
    # 输出要求
    1.你需要根据用户指标预警的具体情况，健康管理师之前给出的运动建议，优化运动日程。
    2.新的运动日程的字段应该和原日程字段内容匹配，优化对应的日程时间、日程推送内容。另外返回修改原因。按json格式返回。
    3.如果你认为需要取消之前对应的早间运动或午间运动或晚间运动项目，则对应的日程时间、日程推送内容返回`false`。
    4.修改原因可以说明为什么之前的不合适。
                        """
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")

            # 获取当天当前时间后的运动日程
            exercise_schedule = get_upcoming_exercise_schedule(group_schedule)

            # 返回所有相关参数
            return {
                "user_profile": profile_data.get("user_profile"),
                "meals_info": meals_info,
                "intent_code": intent_code,
                "group_schedule": exercise_schedule,
                "prompt": prompt
            }

        def aigc_functions_blood_sugar_warning(self, return_text: bool = True, **kwargs):
            """
            血糖预警评估功能，根据用户画像中的专家体系动态选择意图编码。
            需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncrwjP7DqV3kdP0wQ
            """
            # 自动从用户画像中获取 expert_system 和 intent_code
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=3)

            params = self.get_parameters(
                user_id=kwargs.get("user_id"),
                group_id=kwargs.get("group_id"),
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                task_type="blood_sugar_warning"
            )

            user_profile = params.get("user_profile")
            intent_code = params.get("intent_code")
            meals_info = params.get("meals_info")
            diet_comment = params.get("diet_comment")
            prompt = params.get("prompt")

            # 更新 intent_code 到 kwargs 以确保下游使用一致
            kwargs["intentCode"] = intent_code

            # 补充其它字段
            user_profile_str = self.__compose_user_msg_sync__("user_profile", user_profile)

            meals_info_str = format_meals_info_v2(meals_info) if meals_info else None
            warning_indicators = None

            prompt_vars = {
                "user_profile": user_profile_str.rstrip("\n").rstrip(),
                "meals_info": f"## 3小时内的饮食信息\n{meals_info_str}" if meals_info_str else None,
                "warning_indicators": f"## 预警指标\n{warning_indicators}" if warning_indicators else None,
                "diet_comment": f"## 3小时内营养师点评内容\n{diet_comment}" if diet_comment else None
            }

            model_args = self.__update_model_args_sync__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

            content = self.aaigc_functions_general_sync(
                _event="血糖预警分析",
                prompt_vars=prompt_vars,
                model_args=model_args,
                prompt_template=prompt,
                **kwargs
            )

            content = parse_generic_content_sync(content)
            if return_text:
                return "".join([v for k, v in content.items()])
            else:
                return content

        def aigc_functions_update_exercise_schedule(self, **kwargs):
            """
            通过模型生成更新后的运动日程内容
            需求文档：https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89ojDqlncrwjP7DqV3kdP0wQ
            """
            start_time = datetime.now()

            # 获取参数
            params = self.get_parameters(
                user_id=kwargs.get("user_id"),
                group_id=kwargs.get("group_id"),
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                is_new=True,
                task_type="update_exercise_schedule"
            )

            user_profile = params.get("user_profile")
            group_schedule = params.get("group_schedule")
            warning_indicators = params.get("warning_indicators")
            intent_code = kwargs.get("intentCode")
            prompt = params.get("prompt")

            # 格式化内容
            user_profile_str = self.__compose_user_msg_sync__("user_profile", user_profile)
            warning_indicators_str = None
            result = self.aigc_functions_blood_sugar_warning(return_text=False, **kwargs)
            exercise_suggestions = result.get("运动建议")

            # 构建 prompt_vars
            prompt_vars = {
                "user_profile": user_profile_str.rstrip("\n").rstrip(),
                "warning_indicators": f"## 预警指标\n{warning_indicators_str}" if warning_indicators_str else None,
                "exercise_suggestions": f"## 运动建议\n{exercise_suggestions}" if exercise_suggestions else None,
                "group_schedule": f"## 当前运动日程\n{group_schedule}" if group_schedule else None
            }

            # 构建模型参数
            model_args = self.__update_model_args_sync__(kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0)

            # 调用模型
            content = self.aaigc_functions_general_sync(
                _event="更新运动日程",
                prompt_vars=prompt_vars,
                model_args=model_args,
                prompt_template=prompt,
                **kwargs
            )

            # 模型返回结构统一解析
            return parse_generic_content_sync(content)

        def get_nutritionist_feedback_from_conversation(self, **kwargs):
            """从会话记录中获取最像营养师点评的内容"""

            prompt = """
            你是一名专业的营养师，请从以下多轮聊天记录中，找出最像“你对用户饮食行为进行点评”的一句话。

            你的任务是：仅返回一句最符合“营养师点评”风格的回复，例如包含膳食建议、饮食行为反馈、摄入结构分析、提醒注意事项等内容。

            如果无法从聊天记录中找到类似内容，请返回空。

            【多轮聊天记录】
            {{messages}}
            """

            _event = "营养师点评提取"

            messages = kwargs.get("messages", [])

            messages = [i for i in messages if i["role"] == "assistant"]

            messages = self.__compose_user_msg__("messages", messages)

            prompt_vars = {
                "messages": messages,
            }

            # 调用通用的模型生成接口
            model_args = self.__update_model_args__(
                kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
            )
            content = self.aaigc_functions_general(
                _event=_event, prompt_vars=prompt_vars, model_args=model_args, prompt_template=prompt, **kwargs
            )

            # 解析并返回结果
            content = parse_generic_content(content)
            return content


if __name__ == "__main__":
    gsr = InitAllResource()
    agents = HealthExpertModel(gsr)
    param = testParam.param_dev_report_interpretation
    agents.call_function(**param)

    # # 示例 params 参数
    # params_blood_sugar_warning = {
    #     "intentCode": "aigc_functions_blood_sugar_warning",
    #     "user_id": "16191",  # 用户ID
    #     "group_id": "hb_dev@group_EaeRFM1JVf1yh4o4"  # 群组ID
    # }
    #
    # params_update_exercise_schedule = {
    #     "intentCode": "aigc_functions_update_exercise_schedule",
    #     "user_id": "16191",  # 用户ID
    #     "group_id": "hb_dev@group_EaeRFM1JVf1yh4o4"  # 群组ID
    # }
    # # 调用血糖预警服务
    # result_blood_sugar_warning = agents.aigc_functions_blood_sugar_warning(**params_blood_sugar_warning)
    # print(f"Result of blood sugar warning: {result_blood_sugar_warning}")
    #
    # # 调用更新运动日程服务
    # result_update_exercise_schedule = agents.aigc_functions_update_exercise_schedule(**params_update_exercise_schedule)
    # print(f"Result of update exercise schedule: {result_update_exercise_schedule}")