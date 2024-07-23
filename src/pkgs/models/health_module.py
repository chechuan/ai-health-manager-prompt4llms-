# -*- encoding: utf-8 -*-
"""
@Time    :   2024-07-15 13:34:00
@desc    :   健康模块功能实现
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

from src.pkgs.models.utils import ParamTools
from .small_expert_model import expertModel

class HealthModule:
    def __init__(self):
        # 初始化实例属性
        self.some_attribute = None

    async def aigc_functions_diagnosis_generation(self, **kwargs) -> str:
        """西医决策-诊断生成"""

        _event = "西医决策-诊断生成"

        # 必填字段和至少需要一项的参数列表

        at_least_one = ["user_profile", "messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = expertModel.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_chief_complaint_generation(self, **kwargs) -> str:
        """西医决策-主诉生成"""

        _event = "西医决策-主诉生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "messages": []
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

    async def aigc_functions_generate_present_illness(self, **kwargs) -> str:
        """西医决策-现病史生成"""

        _event = "西医决策-现病史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "messages": []
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

    async def aigc_functions_generate_past_medical_history(self, **kwargs) -> str:
        """西医决策-既往史生成"""

        _event = "西医决策-既往史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["past_history_of_present_illness"]
        }
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_allergic_history(self, **kwargs) -> str:
        """西医决策-过敏史生成"""

        _event = "西医决策-过敏史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["allergic_history"]
        }
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_medication_plan(self, **kwargs) -> str:
        """西药医嘱生成"""

        _event = "西药医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_examination_plan(self, **kwargs) -> str:
        """检查检验医嘱生成"""

        _event = "检查检验医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "current_diseases"]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

        # 组合用户画像信息字符串
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = self.__compose_user_msg__(
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
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = self.__compose_user_msg__(
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

    async def aigc_functions_dietary_details_generation(self, **kwargs) -> str:
        """饮食调理细则生成"""

        _event = "饮食调理细则生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", ("current_diseases", "management_goals")]
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 组合会话记录字符串
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
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
        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                content = re.findall("```json(.*?)```", content, re.DOTALL)[0]
                content = dumpJS(json5.loads(content))
            except Exception as e:
                logger.error(f"AIGC Functions {_event} json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)

        return content

    async def aigc_functions_meal_plan_generation(self, **kwargs) -> str:
        """带量食谱-生成餐次、食物名称"""

        _event = "生成餐次、食物名称"

        # 必填字段和至少需要一项的参数列表
        # required_fields = {
        #     "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
        #                      ("current_diseases", "management_goals")]
        # }
        #
        # # 验证必填字段
        # await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        # bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        bmr = 20
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合会话记录字符串
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 饮食调理细则
        ietary_guidelines = self.__compose_user_msg__(
            "ietary_guidelines", ietary_guidelines=kwargs.get("ietary_guidelines")
        )

        # 获取历史食谱
        historical_diets = parse_historical_diets(kwargs.get("historical_diets"))

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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_food_quality_guidance(self, **kwargs) -> str:
        """生成餐次、食物名称的质量指导"""

        _event = "生成餐次、食物名称的质量指导"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")],
            "ietary_guidelines": {
                "basic_nutritional_needs": ""
            }
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        basic_nutritional_needs = kwargs.get("ietary_guidelines").get("basic_nutritional_needs")

        meal_plan = convert_meal_plan_to_text(kwargs.get("meal_plan"))

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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        try:
            content = re.search(r"```json(.*?)```", content, re.S).group(1)
            data = json.loads(content)
        except Exception as err:
            logger.error(f"{_event} json解析失败, {err}")
            data = []
        return data

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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        if "messages" in kwargs and kwargs["messages"] and len(kwargs["messages"]) >= 6:
            messages = self.__compose_user_msg__("messages", messages=kwargs["messages"], role_map={"assistant": "健康管理师", "user": "客户"})
            kwargs["intentCode"] = "aigc_functions_body_fat_weight_management_consultation_suggestions"
            _event = "体脂体重管理-问诊-建议"
        else:
            messages = (
                self.__compose_user_msg__("messages", messages=kwargs["messages"])
                if kwargs.get("messages")
                else ""
            )
        key_indicators = self.__compose_user_msg__(
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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        # 获取并排序体重数据
        weight_data = next((item["data"] for item in kwargs.get("key_indicators", []) if item["key"] == "体重"), [])
        if not weight_data:
            raise ValueError("体重数据缺失")
        weight_data.sort(key=lambda x: dt.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        # 确定事件编码
        days_count = len(weight_data)
        event_key = min(days_count, 3)
        kwargs["intentCode"] = intent_code_map[event_key]

        user_profile = kwargs["user_profile"]
        current_weight = user_profile.get("weight")
        current_bmi = user_profile.get("bmi")

        # 计算标准体重
        standard_weight = calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 组装体重状态和目标
        weight_status, bmi_status, weight_goal = self.__determine_weight_status(user_profile, current_bmi)
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
        user_profile_str = self.__compose_user_msg__("user_profile", user_profile=user_profile)
        key_indicators_str = self.__compose_user_msg__("key_indicators", key_indicators=kwargs["key_indicators"])

        # 组装提示变量并包含体重状态和目标消息
        prompt_vars = {
            "user_profile": user_profile_str,
            "datetime": dt.today().strftime("%Y-%m-%d %H:%M:%S"),
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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        body_fat_data = next((item["data"] for item in key_indicators if item["key"] == "体脂率"), [])
        days_count = len(body_fat_data)
        if days_count == 0:
            raise ValueError("体脂率数据缺失")

        # 根据日期排序体脂数据
        body_fat_data.sort(key=lambda x: dt.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        event_key = min(days_count, 3)
        kwargs["intentCode"] = _intentCode_map[event_key]

        user_profile = kwargs["user_profile"]
        current_body_fat_rate = float(body_fat_data[-1]["value"].replace('%', ''))

        # 计算标准体重
        standard_weight = calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 计算标准体脂率
        standard_body_fat_rate = "10%-20%" if user_profile["gender"] == "男" else "15%-25%"
        user_profile["standard_body_fat_rate"] = standard_body_fat_rate

        # 组装体脂率状态和目标
        body_fat_status, body_fat_goal = self._determine_body_fat_status(user_profile["gender"], current_body_fat_rate)
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
            "datetime": dt.today().strftime("%Y-%m-%d %H:%M:%S"),
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

    def __determine_weight_status(self, user_profile, bmi_value):
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

    def _determine_body_fat_status(self, gender: str, body_fat_rate: float):
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

    async def aigc_functions_diagnosis_generation(self, **kwargs) -> str:
        """西医决策-诊断生成"""

        _event = "西医决策-诊断生成"

        # 必填字段和至少需要一项的参数列表

        at_least_one = ["user_profile", "messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_chief_complaint_generation(self, **kwargs) -> str:
        """西医决策-主诉生成"""

        _event = "西医决策-主诉生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "messages": []
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

    async def aigc_functions_generate_present_illness(self, **kwargs) -> str:
        """西医决策-现病史生成"""

        _event = "西医决策-现病史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "messages": []
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

    async def aigc_functions_generate_past_medical_history(self, **kwargs) -> str:
        """西医决策-既往史生成"""

        _event = "西医决策-既往史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["past_history_of_present_illness"]
        }
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_allergic_history(self, **kwargs) -> str:
        """西医决策-过敏史生成"""

        _event = "西医决策-过敏史生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["allergic_history"]
        }
        at_least_one = ["user_profile", "messages"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_medication_plan(self, **kwargs) -> str:
        """西药医嘱生成"""

        _event = "西药医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_examination_plan(self, **kwargs) -> str:
        """检查检验医嘱生成"""

        _event = "检查检验医嘱生成"

        # 必填字段和至少需要一项的参数列表
        at_least_one = ["messages", "medical_records"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, {}, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile")
        )
        medical_records: str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "current_diseases"]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        # 获取用户画像信息
        user_profile = kwargs.get("user_profile", {})

        # 组合用户画像信息字符串
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = self.__compose_user_msg__(
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
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合消息字符串
        messages_str = self.__compose_user_msg__(
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

    async def aigc_functions_dietary_details_generation(self, **kwargs) -> str:
        """饮食调理细则生成"""

        _event = "饮食调理细则生成"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", ("current_diseases", "management_goals")]
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 组合会话记录字符串
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        # 构建提示变量
        prompt_vars = {
            "user_profile": user_profile_str,
            "messages": messages,
            "current_date": datetime.today().strftime("%Y-%m-%d"),
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
        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                content = re.findall("```json(.*?)```", content, re.DOTALL)[0]
                content = dumpJS(json5.loads(content))
            except Exception as e:
                logger.error(f"AIGC Functions {_event} json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)

        return content

    async def aigc_functions_meal_plan_generation(self, **kwargs) -> str:
        """带量食谱-生成餐次、食物名称"""

        _event = "生成餐次、食物名称"

        # 必填字段和至少需要一项的参数列表
        # required_fields = {
        #     "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
        #                      ("current_diseases", "management_goals")]
        # }
        #
        # # 验证必填字段
        # await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        # bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        bmr = 20
        user_profile_str += f"基础代谢:\n{bmr}\n"

        # 组合病历信息字符串
        medical_records_str = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records")
        )

        # 组合会话记录字符串
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )

        # 饮食调理原则获取
        food_principle = kwargs.get("food_principle")

        # 饮食调理细则
        ietary_guidelines = self.__compose_user_msg__(
            "ietary_guidelines", ietary_guidelines=kwargs.get("ietary_guidelines")
        )

        # 获取历史食谱
        historical_diets = parse_historical_diets(kwargs.get("historical_diets"))

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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
        return content

    async def aigc_functions_generate_food_quality_guidance(self, **kwargs) -> str:
        """生成餐次、食物名称的质量指导"""

        _event = "生成餐次、食物名称的质量指导"

        # 必填字段和至少需要一项的参数列表
        required_fields = {
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")],
            "ietary_guidelines": {
                "basic_nutritional_needs": ""
            }
        }

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields)

        user_profile = kwargs.get("user_profile", {})

        # 初始化变量
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )

        # 使用工具类方法检查并计算基础代谢率（BMR）
        bmr = await ParamTools.check_and_calculate_bmr(user_profile)
        user_profile_str += f"基础代谢:\n{bmr}\n"

        basic_nutritional_needs = kwargs.get("ietary_guidelines").get("basic_nutritional_needs")

        meal_plan = convert_meal_plan_to_text(kwargs.get("meal_plan"))

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

        if isinstance(content, openai.AsyncStream):
            return content
        try:
            content = json5.loads(content)
        except Exception as e:
            try:
                # 处理JSON代码块
                content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
                if content_json:
                    content = dumpJS(json5.loads(content_json[0]))
                else:
                    # 处理Python代码块
                    content_python = re.findall(
                        r"```python(.*?)```", content, re.DOTALL
                    )
                    if content_python:
                        content = content_python[0].strip()
                    else:
                        raise ValueError("No matching code block found")
            except Exception as e:
                logger.error(f"AIGC Functions process_content json5.loads error: {e}")
                content = dumpJS([])
        content = await parse_examination_plan(content)
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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
            "user_profile": ["age", "gender", "height", "weight", "bmi", "daily_physical_labor_intensity",
                             ("current_diseases", "management_goals")]
        }
        at_least_one = ["user_profile", "medical_records", "key_indicators"]

        # 验证必填字段
        await ParamTools.check_required_fields(kwargs, required_fields, at_least_one)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs.get("user_profile", {})
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", {})
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
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
        try:
            content = re.search(r"```json(.*?)```", content, re.S).group(1)
            data = json.loads(content)
        except Exception as err:
            logger.error(f"{_event} json解析失败, {err}")
            data = []
        return data

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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        if "messages" in kwargs and kwargs["messages"] and len(kwargs["messages"]) >= 6:
            messages = self.__compose_user_msg__("messages", messages=kwargs["messages"],
                                                 role_map={"assistant": "健康管理师", "user": "客户"})
            kwargs["intentCode"] = "aigc_functions_body_fat_weight_management_consultation_suggestions"
            _event = "体脂体重管理-问诊-建议"
        else:
            messages = (
                self.__compose_user_msg__("messages", messages=kwargs["messages"])
                if kwargs.get("messages")
                else ""
            )
        key_indicators = self.__compose_user_msg__(
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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        # 获取并排序体重数据
        weight_data = next((item["data"] for item in kwargs.get("key_indicators", []) if item["key"] == "体重"), [])
        if not weight_data:
            raise ValueError("体重数据缺失")
        weight_data.sort(key=lambda x: dt.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        # 确定事件编码
        days_count = len(weight_data)
        event_key = min(days_count, 3)
        kwargs["intentCode"] = intent_code_map[event_key]

        user_profile = kwargs["user_profile"]
        current_weight = user_profile.get("weight")
        current_bmi = user_profile.get("bmi")

        # 计算标准体重
        standard_weight = calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 组装体重状态和目标
        weight_status, bmi_status, weight_goal = self.__determine_weight_status(user_profile, current_bmi)
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
        user_profile_str = self.__compose_user_msg__("user_profile", user_profile=user_profile)
        key_indicators_str = self.__compose_user_msg__("key_indicators", key_indicators=kwargs["key_indicators"])

        # 组装提示变量并包含体重状态和目标消息
        prompt_vars = {
            "user_profile": user_profile_str,
            "datetime": dt.today().strftime("%Y-%m-%d %H:%M:%S"),
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
        await ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        body_fat_data = next((item["data"] for item in key_indicators if item["key"] == "体脂率"), [])
        days_count = len(body_fat_data)
        if days_count == 0:
            raise ValueError("体脂率数据缺失")

        # 根据日期排序体脂数据
        body_fat_data.sort(key=lambda x: dt.strptime(x['time'], "%Y-%m-%d %H:%M:%S"))

        event_key = min(days_count, 3)
        kwargs["intentCode"] = _intentCode_map[event_key]

        user_profile = kwargs["user_profile"]
        current_body_fat_rate = float(body_fat_data[-1]["value"].replace('%', ''))

        # 计算标准体重
        standard_weight = calculate_standard_weight(user_profile["height"], user_profile["gender"])
        user_profile["standard_weight"] = f"{round(standard_weight)}kg"

        target_weight = user_profile.get("target_weight", "未知")
        user_profile["target_weight"] = target_weight

        # 计算标准体脂率
        standard_body_fat_rate = "10%-20%" if user_profile["gender"] == "男" else "15%-25%"
        user_profile["standard_body_fat_rate"] = standard_body_fat_rate

        # 组装体脂率状态和目标
        body_fat_status, body_fat_goal = self._determine_body_fat_status(user_profile["gender"], current_body_fat_rate)
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
            "datetime": dt.today().strftime("%Y-%m-%d %H:%M:%S"),
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

    def __determine_weight_status(self, user_profile, bmi_value):
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

    def _determine_body_fat_status(self, gender: str, body_fat_rate: float):
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