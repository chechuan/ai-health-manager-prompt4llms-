# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-20 17:17:57
@desc    :   健康模块功能实现
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

import asyncio
import json
from copy import deepcopy
from datetime import datetime
from typing import Generator
import json5
import re
from data.test_param.test import testParam
from src.utils.module import (
    check_required_fields, check_aigc_functions_body_fat_weight_management_consultation,
    check_and_calculate_bmr, get_highest_data_per_day, check_consecutive_days,
    calculate_and_format_diet_plan, calculate_standard_weight, convert_meal_plan_to_text,
    curr_time, determine_recent_solar_terms, format_historical_meal_plans,
    format_historical_meal_plans_v2, generate_daily_schedule, generate_key_indicators,
    get_festivals_and_other_festivals, get_weather_info, parse_generic_content,
    remove_empty_dicts, handle_calories, run_in_executor, log_with_source
)
from src.prompt.model_init import acallLLM
from src.utils.Logger import logger
from src.utils.Logger import logger as global_logger
from src.utils.api_protocal import *
from src.utils.resources import InitAllResource


class HealthExpertModel:

    def __init__(self, gsr: InitAllResource) -> None:
        # 初始化实例属性
        self.gsr = gsr
        self.regist_aigc_functions()

    @log_with_source
    async def aaigc_functions_general(
        self,
        _event: str = "",
        prompt_vars: dict = {},
        model_args: Dict = {},
        prompt_template: str = "",
        **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""
        logger = kwargs.get("logger", global_logger)  # 确保使用注入的 logger
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
        prompt_template: str = (
            prompt_template
            if prompt_template
            else self.gsr.get_event_item(event)["description"]
        )
        logger.debug(f"Prompt Vars Before Formatting: {repr(prompt_vars)}")

        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")

        content: Union[str, Generator] = await acallLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def __compose_user_msg__(
        self,
        mode: Literal[
            "user_profile",
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

    async def aigc_functions_dietary_guidelines_generation(self, **kwargs) -> str:
        """饮食调理原则生成"""

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

        _event = "饮食调理细则生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "weight_status", "daily_physical_labor_intensity",
                             "bmi", ("current_diseases", "management_goals")]
        }

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

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

        kwargs["intentCode"] = "aigc_functions_meal_plan_generation"

        _event = "生成餐次、食物名称"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

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

        kwargs["intentCode"] = "aigc_functions_generate_food_quality_guidance"

        _event = "生成餐次、食物名称的质量指导"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age",
                "gender",
                "height",
                "weight",
                "bmi",
                "daily_physical_labor_intensity",
                ("current_diseases", "management_goals"),
            ],
            "ietary_guidelines": {"basic_nutritional_needs": ""},
        }

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

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
                "age",
                "gender",
                "height",
                "weight",
                "bmi",
                "daily_physical_labor_intensity",
                ("current_diseases", "management_goals"),
            ]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records,
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
                "age",
                "gender",
                "height",
                "weight",
                "bmi",
                "daily_physical_labor_intensity",
                ("current_diseases", "management_goals"),
            ]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = await self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = await self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            await self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "date": datetime.today().strftime("%Y-%m-%d"),
            "medical_records": medical_records,
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
            messages = await self.__compose_user_msg__("messages", messages=kwargs["messages"], role_map={"assistant": "健康管理师", "user": "客户"})
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
        weight_status, bmi_status, weight_goal = await self.__determine_weight_status(user_profile, current_bmi)
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
        key_indicators_str = await self.__compose_user_msg__("key_indicators", key_indicators=kwargs.get("key_indicators", None))

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
        body_fat_status, body_fat_goal = await self._determine_body_fat_status(user_profile["gender"], current_body_fat_rate)
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

    async def __determine_weight_status(self, user_profile, bmi_value):
        age = user_profile["age"]
        if 18 <= age < 65:
            if bmi_value < 18.5:
                return "身材偏瘦", "偏低", "增肌"
            elif 18.5 <= bmi_value < 24:
                return "属于标准体重", "正常", "保持体重"
            elif 24 <= bmi_value < 28:
                return "体重超重", "偏高", "减脂"
            else:
                return "属于肥胖状态", "偏高", "减脂"
        else:
            if bmi_value < 20:
                return "身材偏瘦", "偏低", "增肌"
            elif 20 <= bmi_value < 26.9:
                return "属于标准体重", "正常", "保持体重"
            elif 26.9 <= bmi_value < 28:
                return "体重超重", "偏高", "减脂"
            else:
                return "属于肥胖状态", "偏高", "减脂"

    async def _determine_body_fat_status(self, gender: str, body_fat_rate: float):
        """确定体脂率状态和目标"""
        if gender == "男":
            if body_fat_rate < 10:
                return "偏低状态", "增重"
            elif 10 <= body_fat_rate < 20:
                return "正常范围", "保持体重"
            elif 20 <= body_fat_rate < 25:
                return "偏高状态", "减脂"
            else:
                return "肥胖状态", "减脂"
        elif gender == "女":
            if body_fat_rate < 15:
                return "偏低状态", "增重"
            elif 15 <= body_fat_rate < 25:
                return "正常范围", "保持体重"
            elif 25 <= body_fat_rate < 30:
                return "偏高状态", "减脂"
            else:
                return "肥胖状态", "减脂"

    async def aigc_functions_recommended_daily_calorie_intake(self, **kwargs) -> str:
        """
        推荐每日饮食摄入热量值（B端）

        需求文档：https://alidocs.dingtalk.com/i/nodes/Gl6Pm2Db8D1M7mlatXQ9O6B2WxLq0Ee4?utm_scene=team_space&iframeQuery=anchorId%3Duu_lygopp0xcx3sb0lz5si

        根据用户画像如健康状态、管理目标等信息，推荐用户每日饮食应该摄入的热量值，
        供B端营养师参考和调整，方便营养师指导用户的饮食方案。

        参数:
            kwargs (dict): 包含用户画像和病历信息的参数字典

        返回:
            str: 生成的每日饮食推荐摄入热量值（单位：kcal）
        """

        _event = "推荐每日饮食摄入热量值"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": [
                "age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                "recommended_caloric_intake", ("current_diseases", "management_goals")
            ]
        }

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await check_and_calculate_bmr(user_profile)
        user_profile["bmr"] = f"{bmr}kcal"

        # 组合用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 组合病历信息字符串
        medical_records_str = await self.__compose_user_msg__("medical_records", medical_records=kwargs.get("medical_records", ""))

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

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

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

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

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

    async def aigc_functions_recommended_meal_plan_with_recipes(self, **kwargs) -> List[Dict[str, List[Dict[str, float]]]]:
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
                "weight_status", ("current_diseases", "management_goals")
            ],
            "diet_plan_standards": {}
        }

        # 验证必填字段
        await check_required_fields(kwargs, required_fields)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

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
        messages_str = await self.__compose_user_msg__("messages", messages=messages)

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

        # 调用通用的 AIGC 函数并返回内容
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        content = await parse_generic_content(content)
        content = await run_in_executor(lambda: remove_empty_dicts(content))
        return content

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

        # 获取用户提问信息
        user_question = kwargs.get("user_question", "")
        if not user_question:
            raise ValueError(f"user_question不能为空")

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", "")

        # # 获取健康关键指标
        # health_key_indicators = kwargs.get("health_key_indicators", {}).get("data", [])
        #
        # # 根据管理目标提取近7日的关键指标数据
        # management_goals = user_profile.get("management_goals", "")
        # if management_goals:
        #     recent_indicators = [indicator for indicator in health_key_indicators if
        #                          indicator['date'] >= (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')]
        # else:
        #     recent_indicators = []

        # 拼接用户画像信息字符串
        user_profile_str = await self.__compose_user_msg__("user_profile", user_profile=user_profile)

        # 构建提示变量
        prompt_vars = {
            "user_question": user_question,
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
            # 替换句中非句尾的问号为逗号，只保留句尾问号
            question = re.sub(r"？(?!$)", "，", question)  # 替换句中问号为逗号
            if not question.endswith("？"):
                question += "？"  # 如果句末没有问号，添加一个问号
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

        # 移除性别和年龄信息
        user_profile.pop("gender", None)
        user_profile.pop("age", None)
        user_profile.pop("city", None)

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

        if not today_weather:
            # 如果没有天气信息，删除城市信息
            user_profile.pop("city", None)

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
        truncated_summary = await self.__truncate_to_limit(diet_summary_model_output, limit=20)
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

    async def __truncate_to_limit(self, text: str, limit: int) -> str:
        """
        截取文本至指定字符限制，若字符数超出限制则保留最后一个标点符号，或在句尾加上句号。
        :param text: 原始文本
        :param limit: 字符限制，默认为20个字符
        :return: 处理后的文本
        """
        # 去掉换行符
        text = text.replace("\n", "")

        # 如果文本长度小于等于限制，直接返回
        if len(text) <= limit:
            return text if text.endswith("。") else text + "。"

        # 截取前limit个字符
        truncated = text[:limit]

        # 在截断文本中寻找最后一个标点符号
        last_punctuation = max(truncated.rfind(p) for p in "。，！？")

        # 如果找到标点符号，将该标点转换为句号
        if last_punctuation != -1:
            truncated = truncated[:last_punctuation + 1]
            # 如果标点不是句号，替换成句号
            if truncated[-1] not in "。":
                truncated = truncated[:-1] + "。"
        else:
            # 如果没有标点符号，直接在末尾加句号
            truncated = truncated.rstrip() + "。"

        return truncated

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
        tcm_four_diagnoses = f"## 中医四诊\n{TcmFourDiagnoses(**tcm_four_diagnoses_data).format_tcm_diagnosis()}" if any(tcm_four_diagnoses_data.values()) else ""

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


if __name__ == "__main__":
    gsr = InitAllResource()
    agents = HealthExpertModel(gsr)
    param = testParam.param_dev_report_interpretation
    agents.call_function(**param)