# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-30 10:08:10
@desc    :   自定义对话模型
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
from typing import Any, AnyStr, Dict, List

from requests import Session

from data.constrant import (
    CUSTOM_CHAT_REPOR_TINTERPRETATION_ANSWER_SYS_PROMPT,
    CUSTOM_CHAT_REPOR_TINTERPRETATION_SYS_PROMPT_END_SUMMARY
)
from src.pkgs.models.expert_model import expertModel
from src.prompt.model_init import ChatMessage, DeltaMessage, acallLLM, callLLM
from src.test.exp.data.prompts import (
    _auxiliary_diagnosis_judgment_repetition_prompt,
    _auxiliary_diagnosis_system_prompt_v7,
    _chat_start_with_weather_v2
)
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    accept_stream_response,
    determine_recent_solar_terms_sanji,
    dumpJS,
    update_mid_vars,
    get_weather_info,
    determine_recent_solar_terms
)


class CustomChatModel:
    def __init__(self, gsr: InitAllResource):
        self.gsr = gsr
        self.code_func_map = {
            "blood_meas": expertModel.tool_rules_blood_pressure_level_2,
            "weight_meas": expertModel.fat_reduction,
            "pressure_meas": expertModel.emotions,
            # "blood_meas_with_doctor_recommend": expertModel.tool_rules_blood_pressure_level_doctor_rec,
        }

    def __parameter_check__(self, **kwargs):
        """参数检查"""
        if "intentCode" not in kwargs:
            raise ValueError("intentCode is not in kwargs")
        if kwargs["intentCode"] not in self.code_func_map:
            raise ValueError("intentCode is not in CustomChatModel.code_func_map")
        # if not kwargs.get("history", []):
        #     raise ValueError("history is empty")

    def __extract_event_from_gsr__(
            self, gsr: InitAllResource, code: str
    ) -> Dict[str, Any]:
        """从global_share_resource中提取事件数据"""
        event = {}
        if gsr.prompt_meta_data["event"].get(code):
            event = gsr.prompt_meta_data["event"][code]
            return event
        else:
            raise ValueError(
                f"event code {code} not found in gsr.prompt_meta_data['event']"
            )

    def chat(self, **kwargs):
        """自定义对话"""
        # self.__parameter_check__(**kwargs)
        out = self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatAuxiliary(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["chat_start_with_weather"] = self.__chat_start_with_weather__
        self.code_func_map["glucose_consultation"] = self.__chat_glucose_consultation__
        self.code_func_map["sanji_glucose_diagnosis"] = self.__chat_glucose_diagnosis__
        self.code_func_map["blood_interact"] = self.__chat_blood_interact__
        self.code_func_map["3d_interact"] = self.__chat_3d_interact__
        self.code_func_map["auxiliary_diagnosis"] = self.__chat_auxiliary_diagnosis__
        self.code_func_map["auxiliary_diagnosis_with_doctor_recommend"] = (
            self.__chat_auxiliary_diagnosis__
        )

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        def find_second_occurrence(s, char):
            first_index = s.find(char)
            if first_index == -1:
                return -1
            second_index = s.find(char, first_index + 1)
            return second_index

        try:
            text = text.replace("：", ":")
            text = text.replace("doctor", "Doctor")
            text = text.replace("Question:", "")
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            thought_index2 = find_second_occurrence(text, "\nThought:")
            if thought_index != -1 and doctor_index == -1:
                return "None", text[thought_index + 8: doctor_index].strip()
            if thought_index == -1 and doctor_index != -1:
                return "None", text[doctor_index + 8:].strip()
            if thought_index == -1 and doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8: doctor_index].strip()
            doctor = text[doctor_index + 8: thought_index2].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __parse_diff_response__(self, text, s1, s2):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        def find_second_occurrence(s, char):
            first_index = s.find(char)
            if first_index == -1:
                return -1
            second_index = s.find(char, first_index + 1)
            return second_index

        try:
            l1 = len(s1)
            l2 = len(s2)
            text = text.replace("：", ":")
            text = text.replace("doctor", "Doctor")
            text = text.replace("Question:", "")
            thought_index = text.find(s1)
            if thought_index == -1:
                doctor_index = text.find(s2)
                if doctor_index != -1:
                    return "None", text[doctor_index + l2:].strip()
                else:
                    return "None", text
            doctor_index = text.find("\n" + s2)
            if doctor_index == -1:
                return "None", text[thought_index + l1:].strip()
            thought = text[thought_index + l1: doctor_index].strip()
            thought_index2 = find_second_occurrence(text, "\n" + s1)
            if thought_index2 != -1:
                doctor = text[doctor_index + l2 + 1: thought_index2].strip()
            else:
                doctor = text[doctor_index + l2 + 1:].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __parse_jr_response__(self, text):
        try:
            idx = text.find("\nResult:")
            if idx == -1:
                return ""
            out = text[idx:].split("\n")[0].strip()
            return out
        except Exception as err:
            logger.error(text)
            return "None"

    def __compose_auxiliary_diagnosis_message__(
            self, **kwargs
    ) -> List[DeltaMessage]:
        """组装辅助诊断消息"""
        # event = self.__extract_event_from_gsr__(self.gsr, "auxiliary_diagnosis")
        # sys_prompt = event["description"] + "\n" + event["process"]
        history = [
            i for i in kwargs["history"] if i.get("intentCode") == "auxiliary_diagnosis"
        ]
        sys_prompt = _auxiliary_diagnosis_system_prompt_v7
        pro = kwargs.get("promptParam", {})

        user_profile = pro.get('user_profile', '')
        prompt_vars = {
            "user_profile": user_profile}

        sys_prompt = sys_prompt.format(**prompt_vars)
        system_message = DeltaMessage(role="system", content=sys_prompt)
        messages = []
        for idx in range(len(history)):
            i = history[idx]
            if i["role"] == "assistant":
                if i.get("function_call"):
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"Doctor: {i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
            if i["role"] == "user":
                if idx == 0:
                    content = f"Question: {i['content']}"
                else:
                    content = f"Observation: {i['content']}"
                messages.append(DeltaMessage(role="user", content=content))
            if i["role"] == "doctor":
                if i.get("function_call"):
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"{i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
        messages = [system_message] + messages
        for idx, n in enumerate(messages):
            messages[idx] = n.dict()
        return messages

    def __compose_glucose_consultation_message__(self, **kwargs) -> List[DeltaMessage]:
        """组装辅助诊断消息"""
        slot_dict = {
            "空腹": "fasting",
            "早餐后2h": "breakfast2h",
            "午餐后2h": "lunch2h",
            "晚餐后2h": "dinner2h",
        }
        history = [
            i
            for i in kwargs["history"]
            if i.get("intentCode") == "glucose_consultation"
        ]
        prompt_template = (
            "# 已知信息\n"
            "## 我的画像标签\n"
            "年龄：{age}\n"
            "性别：{gender}\n"
            "既往史：{disease}\n"
            "糖尿病类型：{glucose_t}\n"
            "## 我的血糖情况\n"
            "{glucose_message}\n"
            "## 当前日期、当前时段及当前血糖\n"
            "当前日期：{recent_day}\n"
            "当前时段以及血糖：{recent_time}:{gl}\n\n"
            "# 任务描述\n"
            "1.你是一个经验丰富的医生,请你协助我进行糖尿病慢病病人测量血糖后的情况进行问诊，以下是对问诊流程的描述。\n"
            "2.我会提供近一周血糖数据, 请你根据自身经验分析,针对我的个人情况提出相应的问题，每次最多提出两个问题。\n"
            "3.问题关键点可以包括: 是否出现不适症状，症状的持续时间及严重程度，用药情况，饮食情况，血压波动的诱因等,同类问题可以总结在一起问。\n"
            "4.最后请你结合获取到的信息给出结论分析，并解释患者症状可能的原因，如果患者存在饮食和运动不规律、治疗依从性差、情绪应激、睡眠障碍、酗酒、感染、胰岛素不规范注射等情况，给出合理建议。多轮问诊最多3轮，不要打招呼，不要输出列表，只输出问题，不要输出其他内容，直接开始问诊。每轮输出不超过250字。\n"
            "5.输出结论和建议时，不要带小标题，直接输出结论和建议内容。\n"
        )

        pro = kwargs.get("promptParam", {})
        data = pro.get("glucose", {})
        result = "|血糖测量时段|"
        for date in data.keys():
            result += date + "|"
        result += "\n"
        for time in ["空腹", "早餐后2h", "午餐后2h", "晚餐后2h"]:
            result += "|" + time + "|"
            for date in data.keys():
                t_e = slot_dict[time]
                if t_e in data[date]:
                    result += data[date][t_e] + "|"
            result += "\n"

        prompt_vars = {
            "age": pro.get("askAge", ""),
            "gender": pro.get("askSix", ""),
            "disease": pro.get("disease", []),
            "glucose_t": pro.get("glucose_t", ""),
            "glucose_message": result,
            "recent_day": pro.get("currentDate", ""),
            "recent_time": pro.get("current_gl_solt", ""),
            "gl": pro.get("gl", ""),
        }

        sys_prompt = prompt_template.format(**prompt_vars)
        system_message = DeltaMessage(role="system", content=sys_prompt)
        messages = []
        for idx in range(len(history)):
            i = history[idx]
            if i["role"] == "assistant":
                if i.get("function_call"):
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"Doctor: {i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
            if i["role"] == "user":
                if idx == 0:
                    content = f"Question: {i['content']}"
                else:
                    content = f"Observation: {i['content']}"
                messages.append(DeltaMessage(role="user", content=content))
        messages = [system_message] + messages
        for idx, n in enumerate(messages):
            messages[idx] = n.dict()
        return messages

    def __compose_glucose_diagnosis_message__(self, **kwargs) -> List[DeltaMessage]:
        """组装血糖波动问诊消息"""
        history = [
            i
            for i in kwargs["history"]
            if i.get("intentCode") == "sanji_glucose_diagnosis"
        ]
        prompt_template = (
            "# 已知信息\n"
            "## 用户画像\n"
            "年龄：{age}\n"
            "性别：{gender}\n"
            "身高：{height}\n"
            "体重：{weight}\n"
            "现患病：{disease}\n"
            "生活习惯：{habits}\n"
            "用药情况：{medication}\n"
            "## 7天血糖情况\n"
            "{glucose_message}\n"
            "## 当前日期、当前时段及当前血糖\n"
            "当前日期：{recent_day}\n"
            "当前时段以及血糖：{recent_gl}\n\n"
            "# 任务描述\n"
            "你扮演一个经验丰富的慢病管理专家以及家庭医生，你负责管理的客户血糖异常波动，近3天血糖高于控制目标，请你充分利用已知信息，遵守输出要求，对他血糖测量后的情况进行处理。\n"
            "# 输出要求\n"
            "1. 思考根据当前情况你需要做什么，例如可以收集信息，回答问题，出具建议等。\n"
            "2. 如果你需要收集更多信息，则输出问诊问题，确保获取的信息尽可能详细和准确，这样能更准确地判断血糖波动的原因。每次只提出1-2个问题。\n"
            "3. 如果客户向你发问，则解答用户疑问，保持专业性。\n"
            "4. 如果你收集够足够的信息，则输出对客户的处理建议，包括向客户解释他当前情况的可能原因，可能需要调整高血压药物的剂量或类型，以控制血压。如果用户近7天血糖测量不够频繁，建议客户频繁监测血糖，尤其是在采取任何生活方式调整或药物调整后，以评估其对血糖的影响。教育患者如何识别和应对高血糖和低血糖，以及如何自我监测血糖。\n"
            "5. 血糖波动可能给患者带来心理压力，提供情绪支持和鼓励非常重要。口语化表达，便于客户理解，每轮输出顺畅衔接历史会话记录，可以使用“请你先不要着急”、“我知道了”、“我非常理解你的心情”等。\n"
            "6.每次输出不超过200字，不要输出列表。\n"
            "# 输出格式要求：\n"
            "请遵循以下格式回复：\n"
            "Thought: 思考针对当前问题应该做什么\n"
            "Doctor: 你作为一个慢病管理专家以及家庭医生,分析思考的内容,在此情况下你会对客户说：\n"
        )

        pro = kwargs.get("promptParam", {})
        upro = kwargs.get("user_profile", {})
        data = pro.get("glucose", {})

        prompt_vars = {
            "age": upro.get("age", ""),
            "gender": upro.get("sex", ""),
            "height": upro.get("height", ""),
            "weight": upro.get("weight", ""),
            "disease": upro.get("disease", ""),
            "habits": upro.get("habits", ""),
            "medication": upro.get("medication", ""),
            "glucose_message": data,
            "recent_day": pro.get("currentDate", ""),
            "recent_gl": pro.get("recent_gl", {}),
        }

        sys_prompt = prompt_template.format(**prompt_vars)
        system_message = DeltaMessage(role="system", content=sys_prompt)
        messages = []
        for idx in range(len(history)):
            i = history[idx]
            if i["role"] == "assistant":
                if i.get("function_call"):
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"Doctor: {i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
            if i["role"] == "user":
                if idx == 0:
                    content = f"Question: {i['content']}"
                else:
                    content = f"Observation: {i['content']}"
                messages.append(DeltaMessage(role="user", content=content))
        messages = [system_message] + messages
        for idx, n in enumerate(messages):
            messages[idx] = n.dict()
        return messages

    def __compose_blood_interact_message__(self, **kwargs):
        intentCode = kwargs.get("intentCode")
        history = [
            i for i in kwargs["history"] if i.get("intentCode") == "blood_interact"
        ]

        pro = kwargs.get("promptParam", {})
        prompt_vars = {
            "age": pro.get("askAge", ""),
            "gender": pro.get("askSix", ""),
            "disease": pro.get("disease", []),
            "goal": pro.get("goal", ""),
            "sbp": pro.get("sbp", ""),
            "dbp": pro.get("dbp", ""),
        }

        prompt_template = self.gsr.get_event_item(intentCode)["description"]
        sys_prompt = prompt_template.format(**prompt_vars)
        system_message = DeltaMessage(role="system", content=sys_prompt)
        messages = []
        for idx in range(len(history)):
            i = history[idx]
            if i["role"] == "assistant":
                if i["function_call"]:
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"Doctor: {i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
            if i["role"] == "user":
                if idx == 0:
                    content = f"Question: {i['content']}"
                else:
                    content = f"Observation: {i['content']}"
                messages.append(DeltaMessage(role="user", content=content))
        messages = [system_message] + messages
        for idx, n in enumerate(messages):
            messages[idx] = n.dict()

        return messages

    def __chat_sanji_glucose_diagnosis___(
            self, **kwargs
    ) -> AnyStr:
        """辅助诊断总结、饮食建议

        Args:
            history (List[Dict]): 历史消息列表

        Returns:
            AnyStr: 辅助诊断总结、饮食建议
        """
        history = [
            i for i in kwargs["history"] if i.get("intentCode") == "sanji_glucose_diagnosis"
        ]
        event = self.__extract_event_from_gsr__(
            self.gsr, "sanji_glucose_diagnosis"
        )
        prompt_template_str = event["process"]
        pro = kwargs.get("promptParam", {})

        user_profile = kwargs.get('user_profile', {})
        recent_gl = pro.get('recent_gl', {})
        if_glucose = pro.get('if_glucose', '')
        glucose_level = pro.get('glucose_level', '')

        compose_message = ""
        for i in history:
            role, content = i["role"], i["content"]
            if role == "assistant":
                compose_message += f"你: {content}\n"
            elif role == "user":
                compose_message += f"用户: {content}\n"
        prompt_vars = {
            "user_profile": user_profile,
            "recent_gl": recent_gl,
            "if_glucose": if_glucose,
            "glucose_level": glucose_level,
            "message": compose_message
        }

        prompt = prompt_template_str.format(**prompt_vars)

        messages = [{"role": "user", "content": prompt}]
        chat_response = callLLM(
            model=self.gsr.model_config.get(
                "custom_chat_auxiliary_diagnosis_summary_diet_rec", "Qwen-14B-Chat"
            ),
            history=messages,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 三济康养方案 LLM Output: \n{content}")
        if "患者" in content:
            logger.warning("Prohibited vocab: 患者 generated.")
            content = content.replace("患者", "用户")
        return content

    def __chat_auxiliary_diagnosis_summary_diet_rec__(
            self, **kwargs
    ) -> AnyStr:
        """辅助诊断总结、饮食建议

        Args:
            history (List[Dict]): 历史消息列表

        Returns:
            AnyStr: 辅助诊断总结、饮食建议
        """
        history = [
            i for i in kwargs["history"] if i.get("intentCode") == "auxiliary_diagnosis"
        ]
        event = self.__extract_event_from_gsr__(
            self.gsr, "auxiliary_diagnosis_summary_diet_rec"
        )
        prompt_template_str = event["process"]
        compose_message = ""
        for i in history:
            role, content = i["role"], i["content"]
            if role == "assistant":
                compose_message += f"你: {content}\n"
            elif role == "user":
                compose_message += f"用户: {content}\n"
        prompt = prompt_template_str.replace("{MESSAGE}", compose_message)
        pro = kwargs.get("promptParam", {})
        symptom = pro.get('symptom', '')
        user_profile = pro.get('user_profile', '')
        prompt = prompt.replace("{symptom}", symptom)
        prompt = prompt.replace("{user_profile}", user_profile)

        messages = [{"role": "user", "content": prompt}]
        chat_response = callLLM(
            model=self.gsr.model_config.get(
                "custom_chat_auxiliary_diagnosis_summary_diet_rec", "Qwen-14B-Chat"
            ),
            history=messages,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 辅助诊断总结、饮食建议 LLM Output: \n{content}")
        if "患者" in content:
            logger.warning("Prohibited vocab: 患者 generated.")
            content = content.replace("患者", "用户")
        return content

    async def __chat_auxiliary_diagnosis__(self, **kwargs) -> ChatMessage:
        """辅助问诊"""
        # 过滤掉辅助诊断之外的历史消息
        # model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]
        model = "Qwen1.5-72B-Chat"

        messages = self.__compose_auxiliary_diagnosis_message__(**kwargs)
        logger.info(f"Custom Chat 辅助诊断 LLM Input: {dumpJS(messages)}")
        valid = True
        # for _ in range(2):
        content = callLLM(
            model=model,
            history=messages,
            temperature=0.9,
            max_tokens=8192,
            top_p=0.5,
            n=1,
            presence_penalty=0,
            frequency_penalty=0,
            # stop=["\nObservation:", "问诊Finished!\n\n", "问诊Finished!\n"],
            stream=False,
        )

        logger.info(f"Custom Chat 辅助诊断 LLM Output: \n{content}")
        thought, doctor = self.__parse_response__(content)
        if doctor.startswith("Doctor: "):
            doctor = doctor[len("Doctor: "):]
        # is_repeat = self.judge_repeat(history, doctor, model)
        # logger.debug(f"辅助问诊 重复判断 结果: {is_repeat}")
        # if is_repeat:
        #     valid = False
        #     continue
        # else:
        #     valid = True
        #     break
        conts = []
        if doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请你稍后再试。"
        # elif not doctor or "问诊Finished" in doctor:
        # elif "?" not in doctor and "？" not in doctor and "有没有" not in doctor and '吗' not in doctor:
        #     # doctor = self.__chat_auxiliary_diagnosis_summary_diet_rec__(history)
        #     sug = self.__chat_auxiliary_diagnosis_summary_diet_rec__(**kwargs)
        #     conts = [
        #         sug,
        #         "我建议接入家庭医生对你进行后续健康服务，是否邀请家庭医生加入群聊？",
        #     ]
        else:
            ...
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )
        return mid_vars, conts, (thought, doctor)

    async def chat_general(
            self,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: Dict = {},
            prompt_template: str = "",
            **kwargs,
    ):
        """通用生成"""
        event = kwargs.get("intentCode")
        model = "Qwen1.5-32B-Chat"
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 0.7,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        des = self.gsr.prompt_meta_data["event"][event]["description"]
        prompt_template: str = prompt_template if prompt_template else des
        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")
        content = await acallLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def __chat_start_with_weather__(self, **kwargs) -> ChatMessage:
        model = 'Qwen1.5-72B-Chat'
        pro = kwargs.get("promptParam", {})
        if_entropy = pro.get("withEntropy", '')
        conts = []

        city = "廊坊"

        today_weather = get_weather_info(self.gsr.weather_api_config, city)
        logger.info(f"获取天气: {dumpJS(today_weather)}")
        if if_entropy == "2" or if_entropy == "3":
            prompt_vars = {"today_weather": today_weather}
            prompt_template = _chat_start_with_weather_v2.replace('today_weather', today_weather)
            content: str = await self.chat_general(
                _event="节气问询",
                prompt_template=prompt_template,
                prompt_vars=prompt_vars,
                **kwargs)
            # if if_entropy=='2':
            #     result = content+'小孩和老人，免疫力相对较低，容易受到天气变化的影响。你是否想了解一下家人最近的身体状况，以便提前做好预防呢？'
            # else:
            # from datetime import datetime
            # now = datetime.now()

            # # 格式化当前时间，仅保留小时和分钟
            # current_time_str = now.strftime("%H:%M")

            # # 提取小时部分
            # hour = int(current_time_str.split(':')[0])

            # # 根据小时判断时段
            # if hour < 11:
            #     t="上午"
            # elif 11 <= hour < 13:
            #     t="中午"
            # else:
            #     t="下午"
            # result = '张叔叔，'+t+'好呀。来跟你说下今天的天气哦。'+content+'可以根据天气安排一下今天的活动哟。'
            result = content
            # conts=['天气的变化往往和我们的健康状态紧密相关，可能会对我们的身体产生一些潜在的影响，需要为你播报昨晚的睡眠情况吗？']

        # 用if_entropy字段来控制不同的场景
        else:
            recent_solar_terms = determine_recent_solar_terms_sanji()
            prompt_vars = {"recent_solar_terms": recent_solar_terms}
            content: str = await self.chat_general(
                _event="节气问询",
                prompt_vars=prompt_vars,
                **kwargs)
            if if_entropy == "0":
                result = (
                        content + today_weather
                        + "基于实时监测的物联数据，结合当前节气及天气情况，综合考虑你与家人的生活习惯，生成了三济健康报告:"
                )
            elif if_entropy == "1":
                if "；" in content:
                    content = content.split("；", 1)[0]
                entropy = pro.get("askEntropy", "")
                result = (
                        "叔叔，你的生命熵为"
                        + entropy
                        + "，主要问题是血压不稳定，需要重点控制血压。"
                        + content
                        + "。血压出现一定程度的波动是正常的生理现象，你不必紧张。你可以通过听音乐、阅读、散步等方式来放松心情以保证血压的稳定。"
                )
            else:
                result = "基于实时监测的物联数据，结合当前节气及天气情况，综合考虑你与家人的生活习惯，生成了三济健康报告:"

        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text="",
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )
        return mid_vars, conts, ('', result)

    async def __chat_blood_interact__(self, **kwargs) -> ChatMessage:
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]
        intentCode = kwargs.get("intentCode")
        pro = kwargs.get("promptParam", {})
        messages = self.__compose_blood_interact_message__(**kwargs)
        logger.info(f"Custom Chat 血压初问诊 LLM Input: {dumpJS(messages)}")
        valid = True
        content = callLLM(
            model=model,
            history=messages,
            temperature=0,
            max_tokens=1024,
            top_p=0.5,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            stream=False,
        )

        logger.info(f"Custom Chat 血压初问诊 LLM Output: \n{content}")
        thought, doctor = self.__parse_response__(content)

        add_mess = ""

        prompt_vars = {
            "age": pro.get("askAge", ""),
            "gender": pro.get("askSix", ""),
            "disease": pro.get("disease", []),
            "goal": pro.get("goal", ""),
            "sbp": pro.get("sbp", ""),
            "dbp": pro.get("dbp", ""),
        }
        conts = []
        if "？" not in content and "?" not in content:
            prompt_template = self.gsr.get_event_item(intentCode)["process"]
            sys_prompt = prompt_template.format(**prompt_vars)
            add_mess = [{"role": "system", "content": sys_prompt}]
            content = callLLM(
                model=model,
                history=add_mess,
                temperature=0,
                max_tokens=1024,
                top_p=0.8,
                n=1,
                presence_penalty=0,
                frequency_penalty=0.5,
                stream=False,
            )
            add_thought, add_content = self.__parse_response__(content)
            add_str = "我给你匹配一个降压小妙招，你可以试一下。"
            conts = [add_content, add_str]

        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )

        return mid_vars, conts, (thought, doctor)

    async def __chat_3d_interact__(self, **kwargs) -> ChatMessage:
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]
        intentCode = kwargs.get("intentCode")
        pro = kwargs.get("promptParam", {})
        messages = self.__compose_blood_interact_message__(**kwargs)
        logger.info(f"Custom Chat 血压初问诊 LLM Input: {dumpJS(messages)}")

        valid = True

        content = callLLM(
            model=model,
            history=messages,
            temperature=0,
            max_tokens=1024,
            top_p=0.5,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            stream=False,
        )

        logger.info(f"Custom Chat 血压初问诊 LLM Output: \n{content}")
        thought, doctor = self.__parse_response__(content)

        add_mess = ""

        prompt_vars = {
            "age": pro.get("askAge", ""),
            "gender": pro.get("askSix", ""),
            "disease": pro.get("disease", []),
            "goal": pro.get("goal", ""),
            "sbp": pro.get("sbp", ""),
            "dbp": pro.get("dbp", ""),
        }
        conts = []
        if "？" not in content and "?" not in content:
            prompt_template = self.gsr.get_event_item(intentCode)["process"]
            sys_prompt = prompt_template.format(**prompt_vars)
            add_mess = [{"role": "system", "content": sys_prompt}]
            content = callLLM(
                model=model,
                history=add_mess,
                temperature=0,
                max_tokens=1024,
                top_p=0.8,
                n=1,
                presence_penalty=0,
                frequency_penalty=0.5,
                stream=False,
            )
            add_thought, add_content = self.__parse_response__(content)
            add_str = "我给你匹配一个降压小妙招，你可以试一下。"
            conts = [add_content, add_str]

        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )

        return mid_vars, conts, (thought, doctor)

    async def __chat_glucose_consultation__(self, **kwargs) -> ChatMessage:
        """辅助问诊"""
        # 过滤掉辅助诊断之外的历史消息
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]

        messages = self.__compose_glucose_consultation_message__(**kwargs)
        logger.info(f"Custom Chat 辅助诊断 LLM Input: {dumpJS(messages)}")
        valid = True

        content = callLLM(
            model=model,
            history=messages,
            temperature=0,
            max_tokens=1024,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            stream=False,
        )

        logger.info(f"Custom Chat 辅助诊断 LLM Output: \n{content}")
        thought, doctor = self.__parse_diff_response__(content, "Thought:", "Doctor:")

        conts = []

        if thought == "None" or doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请你稍后再试。"
        if "？" not in content and "?" not in content and '有没有' not in content and "吗" not in content:
            conts = ["血糖问诊结束"]

        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )
        return mid_vars, conts, (thought, doctor)

    async def __chat_glucose_diagnosis__(self, **kwargs) -> ChatMessage:
        """血糖波动问诊"""
        # 过滤掉辅助诊断之外的历史消息
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]

        messages = self.__compose_glucose_diagnosis_message__(**kwargs)
        logger.info(f"glucose_diagnosis 血糖波动问诊 LLM Input: {dumpJS(messages)}")
        valid = True

        content = callLLM(
            model=model,
            history=messages,
            temperature=0,
            max_tokens=1024,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            stream=False,
        )

        logger.info(f"glucose_diagnosis 血糖波动问诊 LLM Output: \n{content}")
        thought, doctor = self.__parse_diff_response__(content, "Thought:", "Doctor:")

        conts = []

        if thought == "None" or doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请你稍后再试。"
        elif "?" not in doctor and "？" not in doctor and "有没有" not in doctor and '吗' not in doctor:
            # doctor = self.__chat_auxiliary_diagnosis_summary_diet_rec__(history)
            sug = self.__chat_sanji_glucose_diagnosis___(**kwargs)
            conts = [sug]
        else:
            ...
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )
        return mid_vars, conts, (thought, doctor)

    def judge_repeat(self, history, content, model):
        his = [f"{i['role']}:{i['content']}" for i in history]
        his = "[" + "  ".join(his) + "]"
        judge_p = _auxiliary_diagnosis_judgment_repetition_prompt.format(his, content)
        logger.debug(f"问诊重复判断LLM输入：{judge_p}")
        h = [{"role": "user", "content": judge_p}]
        model = self.gsr.model_config["auxiliary_diagnosis_judgment_repetition"]
        content = callLLM(
            model=model,
            history=h,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            repetition_penalty=1,
            stop=["Observation", "问诊Finished!"],
            stream=False,
        )
        logger.debug(f"辅助问诊 重复判断 Output: \n{content}")
        output = self.__parse_jr_response__(content)
        if "yes" in output.lower():
            return True
        elif "no" in output.lower():
            return False
        elif (
                "没有回答" in content
                or "没有被回答" in content
                or "未回答" in content
                or "未被回答" in content
        ):
            return False
        elif "回答过" in content or "回答了" in content:
            return True
        return False

    async def chat(self, **kwargs):
        """自定义对话"""
        self.__parameter_check__(**kwargs)
        out = await self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatReportInterpretationAsk(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["report_interpretation_chat"] = (
            self.__chat_report_interpretation__
        )
        self.code_func_map["report_interpretation_chat_with_doctor_recommend"] = (
            self.__chat_report_interpretation__
        )

    def __compose_message__(
            self,
            history: List[Dict[str, str]],
            intentCode: str = "report_interpretation_chat",
            **kwargs,
    ):
        """组装消息"""
        messages = []
        if not history:
            content = kwargs["promptParam"]["report_ocr_result"]
            sysprompt = self.gsr.get_event_item("report_interpretation_chat")[
                "description"
            ]
            messages.append(
                {
                    "role": "system",
                    "content": sysprompt,
                    "intentCode": intentCode,
                }
            )
            messages.append(
                {"role": "user", "content": content, "intentCode": intentCode}
            )
        else:
            # 出现两次user的信息 == 传入报告一次 + 用户回答一次问题
            # 通过此处替换system_prompt控制问诊轮数
            if len([i for i in history if i["role"] == "user"]) >= 3:
                system_prompt = CUSTOM_CHAT_REPOR_TINTERPRETATION_SYS_PROMPT_END_SUMMARY
                history[0]["content"] = (
                    system_prompt if history[0]["role"] == "system" else ...
                )
                for idx in range(len(history)):
                    msg = history[idx]
                    if msg["role"] != "assistant":
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                    messages[-1]["intentCode"] = intentCode
            else:
                for idx in range(len(history)):
                    msg = history[idx]
                    if msg["role"] == "assistant" and msg.get("function_call"):
                        content = f"Thought: {msg['content']}\nDoctor: {msg['function_call']['arguments']}"
                        messages.append({"role": "assistant", "content": content})
                    elif msg["role"] == "assistant":
                        messages.append(
                            {"role": "assistant", "content": msg["content"]}
                        )
                    else:
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                    messages[-1]["intentCode"] = intentCode
        return messages

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        try:
            if text.count("Thought:") > 1:
                second_thought_index = text.find("Thought", text.find("Thought") + 1)
                text = text[second_thought_index:]
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            if thought_index == -1 or doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8: doctor_index].strip()
            doctor = text[doctor_index + 8:].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __chat_report_interpretation__(self, tool: str = "AskHuman", **kwargs):
        """报告解读"""
        model = self.gsr.model_config["report_interpretation_chat"]
        # model = "Qwen-72B-Chat"
        messages = self.__compose_message__(**kwargs)
        logger.info(f"Custom Chat 报告解读 LLM Input: {dumpJS(messages)}")
        content = callLLM(
            model=model,
            history=messages,
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
            # top_k=-1,
            # n=1,
            # presence_penalty=1.15,
            # frequency_penalty=2,
            # repetition_penalty=1,
            # length_penalty=1.2,
            stream=False,
        )
        # content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 报告解读 LLM Output: \n{content}")
        thought, content = self.__parse_response__(content)
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义报告解读对话",
        )
        _contents = []
        sch = -1
        if (
                kwargs["intentCode"] == "report_interpretation_chat"
                and "?" not in content
                and "？" not in content
        ):
            tool = "convComplete"
            sch = 1
            if kwargs["promptParam"]["report_type"] == "口腔报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。别担心，我已经帮你智能匹配到奉华林社区卫生服务中心口腔科的滑波医生，他可是廊坊最好的齿科医生了，并告诉了你妈妈，让她尽快带你去看医生。我还为你智能匹配了一个非常适合你的口腔保健服务包，里面有全套的牙齿问诊和保健服务。你近期一定要认真刷牙，我每天早晚会给你按时播放一个专业的刷牙视频，超级专业有趣的，我陪你一起保护牙齿！"
                ]
            elif kwargs["promptParam"]["report_type"] == "胸部报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。我已经帮你智能匹配到廊坊市人民医院呼吸内科汪医生，并告诉了你的家人，让她尽快带你去看医生。根据你的情况，我为你智能匹配了一个适合你的健康保险计划，里面包含门诊和住院绿通服务、陪诊服务。可针对常见病如肺炎、中耳炎和20种传染病可以报销。帮助守护你的健康。"
                ]
            elif kwargs["promptParam"]["report_type"] == "腹部报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。我已经帮你智能匹配到河北中石油中心医院肝胆内科赵医生，请你尽快去看医生。根据你的情况，我为你智能匹配了一个健康体检保险计划，其中包含全面体检服务、门诊挂号和陪诊服务，可针对规定的12个项目内的检查化验项目进行门诊报销。"
                ]
        return mid_vars, messages, _contents, sch, (thought, content, tool)


class CustomChatReportInterpretationAnswer(CustomChatModel):
    session: Session = Session()

    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["report_interpretation_answer"] = (
            self.__chat_report_interpretation_answer__
        )

    def __search_docs__(
            self,
            query: str = "用户query",
            knowledge_base_name: str = "新奥百科知识库",
            top_k: int = 3,
            score_threshold: float = 0.5,
    ) -> str:
        """从指定知识库搜索相关文档"""
        url = self.gsr.api_config["langchain"] + "/knowledge_base/search_docs"
        payload = dumpJS(
            {
                "query": query,
                "knowledge_base_name": knowledge_base_name,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
            ensure_ascii=True,
        )
        docs = self.session.post(url, data=payload).json()
        if docs:
            content = "## 相关知识\n"
            for doc in docs:
                content += f"{doc['page_content']}\n\n"
        else:
            content = "## 相关知识\n无"
        return content.strip()

    def __compose_message__(
            self,
            history: List[Dict[str, str]],
            intentCode: str = "report_interpretation_chat",
            **kwargs,
    ):
        """组装消息"""
        messages = []
        # 首次补充system信息
        if len(history) == 1 and history[0]["role"] == "user":
            # base_info = kwargs["promptParam"]["report_ocr_result"]
            base_info = (
                "患者姓名：张叔叔{life_entropy}\n"
                "张叔叔：年龄68岁，身高170cm，体重70kg，所患疾病：高血压；饮食喜好：喜甜\n"
                "用户：张叔叔的女儿\n"
            )
            life_entropy = kwargs["promptParam"].get("life_entropy", "60.85")
            base_info = base_info.format(life_entropy=f", 生命熵：{life_entropy}")
            external_knowledge = self.__search_docs__(query=history[-1]["content"])
            logger.debug(f"查询到知识库的内容:\n{external_knowledge}")
            system_prompt = CUSTOM_CHAT_REPOR_TINTERPRETATION_ANSWER_SYS_PROMPT.format(
                base_info=base_info, external_knowledge=external_knowledge
            )
            messages = [
                           {"role": "system", "content": system_prompt, "intentCode": intentCode}
                       ] + messages
        for idx in range(len(history)):
            msg = history[idx]
            if msg["role"] == "assistant" and msg.get("function_call"):
                content = f"Thought: {msg['content']}\nDoctor: {msg['function_call']['arguments']}"
                messages.append({"role": "assistant", "content": content})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages[-1]["intentCode"] = intentCode
        return messages

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        try:
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            if thought_index == -1 or doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8: doctor_index].strip()
            doctor = text[doctor_index + 8:].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __chat_report_interpretation_answer__(self, tool: str = "AskHuman", **kwargs):
        """报告解读"""
        model = self.gsr.model_config["report_interpretation_chat"]
        # model = "Qwen-72B-Chat"
        messages = self.__compose_message__(**kwargs)
        logger.info(f"Custom Chat 报告解读Answer LLM Input: {dumpJS(messages)}")
        chat_response = callLLM(
            model=model,
            history=messages,
            temperature=0.7,
            max_tokens=4096,
            top_p=0.5,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=False)
        logger.info(f"Custom Chat 报告解读Answer LLM Output: \n{content}")
        thought, content = self.__parse_response__(content)
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义报告解读对话Answer",
        )
        _contents = []
        sch = -1
        return mid_vars, messages, _contents, sch, (thought, content, tool)
