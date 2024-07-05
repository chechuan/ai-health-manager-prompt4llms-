# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-05 15:14:07
@desc    :   专家模型 & 独立功能
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import asyncio
from copy import deepcopy
import json
import random
import re
import sys
import time
from os.path import basename
from pathlib import Path

import json5
import openai
from fastapi.exceptions import ValidationException
from requests import Session

from src.utils.api_protocal import (
    USER_PROFILE_KEY_MAP,
    DoctorInfo,
    DrugPlanItem,
    KeyIndicators,
    UserProfile,
    bloodPressureLevelResponse,
)

sys.path.append(Path(__file__).parents[4].as_posix())
import datetime
from datetime import datetime, timedelta
from string import Template
from typing import AsyncGenerator, Dict, Generator, List, Literal, Optional, Union

from langchain.prompts.prompt import PromptTemplate
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

from chat.qwen_chat import Chat
from data.constrant import *
from data.constrant import DEFAULT_RESTAURANT_MESSAGE, HOSPITAL_MESSAGE
from data.jiahe_prompt import *
from data.jiahe_util import *
from data.test_param.test import testParam
from src.pkgs.models.utils import ParamTools
from src.prompt.model_init import ChatMessage, acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.module import (
    InitAllResource,
    accept_stream_response,
    clock,
    compute_blood_pressure_level,
    construct_naive_response_generator,
    download_from_oss,
    dumpJS,
    param_check,
    parse_examination_plan,
    calculate_bmr,
    parse_measurement,
    parse_historical_diets,
    async_clock,
    convert_meal_plan_to_text,
)


class expertModel:
    indicatorCodeMap = {
        "收缩压": "lk1589863365641",
        "舒张压": "lk1589863365791",
        "心率": "XYZBXY001005",
    }

    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr = gsr
        self.gsr.expert_model = self
        self.client = openai.OpenAI()

    def load_image_config(self):
        self.image_font_path = Path(__file__).parent.parent.parent.parent.joinpath(
            "data/font/simsun.ttc"
        )
        if not self.image_font_path.exists():
            logger.error(f"font file not found: {self.image_font_path}")
            exit(1)

    def check_number(x: Union[str, int, float], key: str):  # type: ignore
        """检查数字
        - Args

            x 输入值 (str, int, float)
        """
        try:
            x = float(x)
            return float(x)
        except Exception as err:
            raise f"{key} must can be trans to number."

    @staticmethod
    def tool_compute_bmi(weight: Union[int, float], height: Union[int, float]) -> float:  # type: ignore
        """计算bmi

        - Args

            weight 体重(kg)

            height 身高(m)
        """
        assert type(weight) is float or int, "type `weight` must be a number"
        assert type(height) is float or int, "type `height` must be a number"
        assert (
            weight > 0 and weight < 300
        ), f"value `weigth`={weight} is not valid ∈ (0, 300)"
        assert (
            height > 0 and height < 2.5
        ), f"value `height`={height} is not valid ∈ (0, 2.5)"
        bmi = round(weight / height**2, 1)
        return bmi

    @staticmethod
    def tool_compute_max_heart_rate(age: Union[int, float]) -> int:
        """计算最大心率

        max_heart_rate = 220-年龄（岁）
        """
        try:
            age = float(age)
        except Exception as e:
            logger.exception(e)
            raise f"type `age`={age} must be a number"
        return int(220 - age)

    @staticmethod
    def tool_compute_exercise_target_heart_rate_for_old(age: Union[int, float]) -> int:
        """计算最大心率

        max_heart_rate = 170-年龄（岁）
        """
        try:
            age = float(age)
        except Exception as e:
            logger.exception(e)
            raise f"type `age`={age} must be a number"
        return int(170 - age)

    @staticmethod
    def tool_assert_body_status(age: Union[int, float], bmi: Union[int, float]) -> str:
        """判断体重状态

        - Rules
            体重偏低：18≤年龄≤64：BMI＜18.5；年龄＞64：BMI＜20
            体重正常：18≤年龄≤64：18.5≤BMI＜24；年龄＞64：20≤BMI＜26.9
            超重：18≤年龄≤64：24≤BMI＜28，年龄＞64：26.9≤BMI＜28
            肥胖：年龄≥18：BMI≥28
        """
        expertModel.check_number(age, "age")
        expertModel.check_number(bmi, "bmi")
        assert age < 18, f"not support age < {age}"
        status = ""
        if 18 <= age <= 64:
            if bmi < 18.5:
                status = "体重偏低"
            elif 18.5 <= bmi < 24:
                status = "体重正常"
            elif 24 <= bmi < 28:
                status = "超重"
        elif 64 < age:
            if bmi < 20:
                status = "体重偏低"
            elif 20 <= bmi < 26.9:
                status = "体重正常"
            elif 26.9 <= bmi < 28:
                status = "超重"
        if not status and bmi >= 28:
            status = "肥胖"
        else:
            status = "体重正常"
        return status

    @staticmethod
    def emotions(**kwargs):
        cur_date = kwargs.get("cur_date", "")
        if not cur_date:
            cur_date = kwargs.get("promptParam", "").get("cur_date", "")
        level = kwargs.get("level", "")
        if not level:
            level = kwargs.get("promptParam", "").get("level", "")
        emos = ["中度", "中度", "中度", "中度", "中度", "中度", level]
        sleeps = [
            "77（良好）",
            "56（较差）",
            "78（良好）",
            "78（良好）",
            "85（最佳）",
            "65（一般）",
            "71（良好）",
        ]
        curDate = datetime.now().date()
        emotion = ""
        sleep = ""
        x = 0
        for i in range(6, -1, -1):
            delta = timedelta(days=i)
            previous_datetime = curDate - delta
            emotion += str(previous_datetime) + ":" + emos[x] + "\n"
            sleep += str(previous_datetime) + ":" + sleeps[x] + "\n"
            x += 1
        prompt = emotions_prompt.format(emotion, sleep)
        messages = [{"role": "user", "content": prompt}]
        logger.debug("压力模型输入:" + json.dumps(messages, ensure_ascii=False))
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.8,
            temperature=0.0,
            do_sample=False,
            model="Qwen1.5-72B-Chat",
        )
        logger.debug("压力模型输出:" + generate_text)
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        if generate_text.find("\nDoctor") == -1:
            content = generate_text
        else:
            outIdx = generate_text.find("\nDoctor") + 8
            content = generate_text[outIdx:].split("\n")[0].strip()
        return {
            "thought": thought,
            "scheme_gen": 0,
            "content": content
            + "已为您智能匹配了最适合您的减压方案，帮助您改善睡眠、缓解压力。",
            "scene_ending": True,
        }

    @staticmethod
    def weight_trend(cur_date, weight):
        prompt = weight_trend_prompt.format(cur_date, weight)
        messages = [{"role": "user", "content": prompt}]
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.8,
            temperature=0.0,
            do_sample=False,
            model="Qwen1.5-72B-Chat",
        )
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        outIdx = generate_text.find("\nOutput") + 8
        content = generate_text[outIdx:].split("\n")[0].strip()
        return {"thought": thought, "content": content, "scene_ending": True}

    @staticmethod
    def fat_reduction(**kwargs):
        def get_scheme_modi_type(text):
            if "运动" in text and "不满意" in text:
                return "scheme_changed"
            elif "饮食" in text and "不满意" in text:
                return "scheme_changed"
            elif "整体" in text and "不满意" in text:
                return "scheme_changed"
            else:
                return "scheme_no_change"

        cur_date = kwargs["promptParam"].get("cur_date", "").split(" ")[0]
        weight = kwargs["promptParam"].get("weight", "")
        query = ""
        if len(kwargs["history"]) > 0:
            query = kwargs["history"][-1]["content"]
        # if not query:
        #     return {'thought': '', 'content': f'您今日体重为{weight}', 'scene_ending': False, 'scheme_gen':False}
        if query:
            # 判断是否是对方案不满意及对方案某一部分不满意
            prompt = weight_scheme_modify_prompt.format(query)
            logger.debug("进入体重方案修改流程。。。")
        else:
            # query = query if query else "减脂效果不好，怎么改善？"
            current_date = datetime.now().date()
            weights = [
                "74.6kg",
                "75kg",
                "75.3kg",
                "75.5kg",
                "75.8kg",
                "75.9kg",
                "75.4kg",
                "75.7kg",
                "75.4kg",
                "75.6kg",
                "75.3kg",
                "75.6kg",
                "75.3kg",
            ]
            weight_msg = ""
            for i in range(len(weights)):
                d = current_date - timedelta(days=len(weights) - i)
                weight_msg += f"{d}: {weights[i]}\n"

            prompt = fat_reduction_prompt.format(
                weight_msg, cur_date, weight, "减脂效果不好，怎么改善？"
            )
            logger.debug("进入体重出方案流程。。。")
        messages = [{"role": "user", "content": prompt}]
        logger.debug(
            "体重方案/修改模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.8,
            temperature=0.0,
            do_sample=False,
            model="Qwen1.5-72B-Chat",
        )
        logger.debug("体重方案/修改模型输出： " + generate_text)
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        logger.debug("体重方案/修改模型thought： " + thought)
        if generate_text.find("\nOutput") == -1:
            content = generate_text
        else:
            outIdx = generate_text.find("\nOutput") + 8
            content = generate_text[outIdx:].split("\n")[0].strip()
        if not query:
            try:
                num = round(float(weight.replace("kg", "")) - 75.4, 1)
                if num < 0:
                    cnt = f"体重较上周减少{num}kg。"
                else:
                    cnt = f"体重较上周增加{num}kg。"

            except Exception as err:
                return {
                    "thought": thought,
                    "contents": [
                        f"您今日体重为{weight}。",
                        "健康报告显示您的健康处于平衡状态。"
                        + content
                        + "这里是您下周的方案，请查收。",
                    ],
                    "scene_ending": False,
                    "scheme_gen": 1,
                    "modi_scheme": "scheme_no_change",
                    "weight_trend_gen": True,
                }
            finally:
                return {
                    "thought": thought,
                    "contents": [
                        f"您今日体重为{weight}。",
                        cnt,
                        "健康报告显示您的健康处于平衡状态。"
                        + content
                        + "这里是您下周的方案，请查收。",
                    ],
                    "scene_ending": False,
                    "scheme_gen": 2,
                    "modi_scheme": "scheme_no_change",
                    "weight_trend_gen": True,
                }
        else:
            modi_type = get_scheme_modi_type(content)
            return {
                "thought": thought,
                "contents": ["好的，已重新帮您生成了健康方案，请查收。"],
                "scene_ending": False,
                "scheme_gen": 0,
                "modi_scheme": modi_type,
                "weight_trend_gen": False,
            }

    @staticmethod
    def tool_rules_blood_pressure_level_doctor_rec(**kwargs) -> dict:
        # 血压问诊，出方案

        bps = kwargs.get("promptParam", {}).get("blood_pressure", [])
        bp_msg = ""
        ihm_health_sbp_list = []
        ihm_health_dbp_list = []
        for b in bps:
            if not b:
                continue
            date = b.get("date", "")
            sbp = b.get("ihm_health_sbp", "")
            dbp = b.get("ihm_health_dbp", "")
            ihm_health_dbp_list.append(dbp)
            ihm_health_sbp_list.append(sbp)
            bp_msg += f"{date}|{str(sbp)}|{str(dbp)}|mmHg|\n"

        history = kwargs.get("his", [])
        b_history = kwargs.get("backend_history", [])
        query = history[-1]["content"] if history else ""
        ihm_health_sbp = ihm_health_sbp_list[-1]
        ihm_health_dbp = ihm_health_dbp_list[-1]

        def inquire_gen(hitory, bp_message, iq_n=7):

            history = [
                {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
                for i in hitory
            ]
            hist_s = "\n".join([f"{i['role']}: {i['content']}" for i in history])
            current_date = datetime.now().date()
            drug_msg = ""
            drug_situ = [
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
            ]
            days = []
            for i in range(len(drug_situ)):
                d = current_date - timedelta(days=len(drug_situ) - i - 1)
                drug_msg += f"|{d}| {drug_situ[i]}"
                days.append(d)
            if len(history) >= iq_n:
                messages = [
                    {
                        "role": "user",
                        "content": blood_pressure_scheme_prompt.format(
                            bp_message, drug_msg, current_date, hist_s
                        ),
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": blood_pressure_inquiry_prompt.format(
                            bp_message, drug_msg, current_date, hist_s
                        ),
                    }
                ]  # + history
            logger.debug(
                "血压问诊模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen1.5-72B-Chat",
            )
            logger.debug("血压问诊模型输出： " + generate_text)
            return generate_text

        def blood_pressure_inquiry(history, iq_n=7):
            generate_text = inquire_gen(history, bp_msg, iq_n=iq_n)

            if generate_text.find("Thought") == -1:
                lis = [
                    "结合用户个人血压信息，为用户提供帮助。",
                    "结合用户情况，帮助用户降低血压。",
                ]
                thought = random.choice(lis)
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Assistant") == -1:
                content = generate_text
            else:
                outIdx = generate_text.find("Assistant") + 10
                content = generate_text[outIdx:].strip()
                if content.find("Assistant") != -1:
                    content = content[: content.find("Assistant")]
                if content.find("Thought") != -1:
                    content = content[: content.find("Thought")]

            return thought, content

        thought, content = blood_pressure_inquiry(history, iq_n=4)

        if "？" in content or "?" in content:  # 问诊
            return {
                "level": 0,
                "contents": [content],
                "idx": 0,
                "thought": thought,
                "scheme_gen": -1,
                "scene_ending": False,
                "blood_trend_gen": False,
                "notifi_daughter_doctor": False,
                "call_120": False,
                "is_visit": False,
                "events": [],
            }
        else:  # 出结论
            # thought, cont = blood_pressure_pacify(history, query)  #安抚
            return {
                "level": 0,
                "contents": [content],
                "idx": 0,
                "thought": thought,
                "scheme_gen": 0,
                "scene_ending": True,
                "blood_trend_gen": False,
                "notifi_daughter_doctor": False,
                "call_120": False,
                "is_visit": False,
                "events": [],
            }

    @classmethod
    def _fetch_fake_ques_blood_pressure_v2(cls):
        ques_examples = [
            "Thought: 我想知道你的血压在什么时候变化最明显，是早上起床后还是晚上休息前？还有，你感觉头痛、眩晕或者心悸这些症状有出现过吗？\nAssistant: 你注意到你的血压是在一天中的哪个时段变化最明显吗？是早上刚醒时还是晚上准备睡觉前？另外，你有没有体验过头痛、头晕或者心跳不规律这些症状？",
            "Thought: 我想了解你的血压在何时达到峰值，是否有头痛、眩晕等不适症状。\nAssistant: 你有没有注意到血压升高的时候，是否有头痛、眩晕或者心慌的感觉？这些通常会在什么时候发生，比如早上起床后，还是晚上休息时？",
            "Thought: 想了解血压波动与日常生活可能的关系\nAssistant: 你最近有没有感觉到头痛、眩晕或者心悸的症状？这些可能与血压波动有关。还有，你的日常生活节奏有无显著变化？比如工作压力、饮食习惯或睡眠模式。",
            "Thought: 我想知道你的血压变化是否有特定模式，以及你的身体对药物的反应如何。\nAssistant: 你注意到血压升高通常发生在什么时间吗？服药后多久会感到最舒缓？",
            "Thought: 我想了解你的血压在何时达到峰值，有什么特定触发因素吗？另外，你有没有出现头痛、眩晕或胸闷等不适症状？\nAssistant: 你注意到血压升高的时候有什么特别的情况吗？比如情绪紧张或者剧烈运动后？还有，你有没有头痛、头晕或者胸闷的感觉？",
            "Thought: 想了解血压波动时的主观感受和症状\nAssistant: 你最近有没有头痛、眩晕、心悸或者胸闷的症状？这些可能与血压波动有关。",
            "Thought: 我想知道你的血压在什么时候变化最明显，是早上起床后还是晚上睡前？还有，你感觉头晕或心跳加速的情况有规律吗？\nAssistant: 你注意到你的血压是在一天中的哪个时段变化最明显吗？是早上刚起床的时候还是晚上准备睡觉的时候？另外，你有没有发现头晕或者心跳加速的情况，它们是否有特定的时间规律？",
            "Thought: 我需要了解你的血压控制情况和身体感受。\nAssistant: 你注意到血压在不同日期有明显变化吗？有没有出现头痛、眩晕或胸闷等不适感？",
        ]
        e = random.choice(ques_examples)
        thought = e.split("\n")[0].replace("Thought:", "").strip()
        content = e.split("\n")[1].replace("Assistant:", "").strip()
        return thought, content

    @staticmethod
    def tool_rules_blood_pressure_level_2(**kwargs) -> dict:
        """计算血压等级
        - Args
            ihm_health_sbp(int, float) 收缩压
            ihm_health_dbp(int, float) 舒张压
        - Rules
            有既往史: 用户既往史有高血压则进入此流程管理
            1.如血压达到高血压3级(血压>180/110) 则呼叫救护车。
            2.如血压达到高血压2级(收缩压160-179，舒张压100-109)
                1.预问诊(大模型) (步骤1结果不影响步骤2/3)
                2.是否通知家人 (步骤2)
                3.确认是否通知家庭医师 (步骤3)
            3.如血压达到高血压1级(收缩压140-159，90-99)
                1.预问诊
                2.看血压是否波动，超出日常值30%，则使用“智能呼叫工具”通知家庭医师。日常值30%以内，则嘱患者密切监测，按时服药，注意休息。(调预警模型接口，准备历史血压数据)
            4.如血压达到正常高值《收缩压120-139,舒张压80-89)，
                1.预问诊
                2.则嘱患者密切监测，按时服药，注意休息。
            预问诊事件: 询问其他症状，其他症状的性质，持续时间等。 (2-3轮会话)
        """

        def get_level(ihm_health_sbp, ihm_health_dbp):
            if ihm_health_sbp >= 180 or ihm_health_dbp >= 110:
                return "3级高血压"
            elif 179 >= ihm_health_sbp >= 160 or 109 >= ihm_health_dbp >= 100:
                return "2级高血压"
            elif 159 >= ihm_health_sbp >= 140 or 99 >= ihm_health_dbp >= 90:
                return "1级高血压"
            else:
                return "正常血压"

        def broadcast_gen(bps_msg):
            """
            1. 总结用户信息
            2. 出具血压风险播报以及出具生活改善建议
            """
            t = Template(blood_pressure_risk_advise_prompt)
            prompt = t.substitute(
                bp_msg=bp_msg,
                age=kwargs["promptParam"]["askAge"],
                sex=kwargs["promptParam"]["askSix"],
                height=kwargs["promptParam"]["askHeight"],
                weight=kwargs["promptParam"]["askWeight"],
                disease=kwargs["promptParam"]["askDisease"],
                family_med_history=kwargs["promptParam"]["askFamilyHistory"],
            )
            msg2 = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            logger.debug(
                "血压风险建议模型输入： " + json.dumps(msg2, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=msg2,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                model="Qwen1.5-32B-Chat",
            )
            logger.debug("血压风险建议模型输出： " + generate_text)

            if generate_text.find("Thought") == -1:
                thought = "出具血压风险播报以及生活改善建议"
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Assistant") == -1:
                content = generate_text.replace("Thought:", "").strip()
            else:
                outIdx = generate_text.find("Assistant") + 10
                content = generate_text[outIdx:].strip()
                if content.find("Assistant") != -1:
                    content = content[: content.find("Assistant")]
                if content.find("Thought") != -1:
                    content = content[: content.find("Thought")]

            return thought, content

        def inquire_gen(hit, bp_message, iq_n=7):
            history = [
                {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
                for i in hit
            ]
            _role_map = {"user": "用户", "assistant": "医生助手"}
            hist_s = "\n".join(
                [f"{_role_map.get(i['role'])}: {i['content']}" for i in history]
            )
            current_date = datetime.now().date()
            # current_date = datetime.datetime.now()
            drug_situ, drug_msg = "", [
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
            ]
            days = []
            for i in range(len(drug_situ)):
                d = current_date - timedelta(days=len(drug_situ) - i - 1)
                drug_msg += f"|{d}| {drug_situ[i]}\n"
                days.append(d)
            if len(history) >= iq_n:  # 通过总轮数控制结束
                t = Template(blood_pressure_scheme_prompt)
                prompt = t.substitute(
                    bp_msg=bp_message,
                    history=hist_s,
                    age=kwargs["promptParam"]["askAge"],
                    sex=kwargs["promptParam"]["askSix"],
                    height=kwargs["promptParam"]["askHeight"],
                    weight=kwargs["promptParam"]["askWeight"],
                    disease=kwargs["promptParam"]["askDisease"],
                    family_med_history=kwargs["promptParam"]["askFamilyHistory"],
                )
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            else:  # 正常流程 血压波动问诊询问两次, 询问的内容给了固定话术
                messages = [
                    {
                        "role": "user",
                        "content": blood_pressure_inquiry_prompt.format(
                            bp_message, hist_s
                        ),
                    }
                ]
            logger.debug(
                "血压问诊模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen1.5-32B-Chat",
            )
            logger.debug("血压问诊模型输出： " + generate_text)
            return generate_text

        def blood_pressure_inquiry(hist, query, iq_n=7):
            generate_text = inquire_gen(hist, bp_msg, iq_n=iq_n)
            if len(hist) < iq_n:
                tht, cont = expertModel._fetch_fake_ques_blood_pressure_v2()

                if generate_text.find("Thought") == -1:
                    thought = tht
                    content = cont
                else:
                    thoughtIdx = generate_text.find("Thought") + 8
                    thought = generate_text[thoughtIdx:].split("\n")[0].strip()
                if generate_text.find("Assistant") == -1:
                    thought = tht
                    content = cont
                else:
                    outIdx = generate_text.find("Assistant") + 10
                    content = generate_text[outIdx:].strip()
                    if content.find("Assistant") != -1:
                        content = content[: content.find("Assistant")]
                    if content.find("Thought") != -1:
                        content = content[: content.find("Thought")]
            else:
                thought = ""
                content = generate_text

            return thought, content

        def blood_pressure_pacify(history, query):
            history = [
                {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("Doctor" if not i["role"] == "user" else "user")
                    + f": {i['content']}"
                    + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = blood_pressure_pacify_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            logger.debug(
                "血压安抚模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            )
            logger.debug("血压安抚模型输出： " + generate_text)
            if generate_text.find("\nThought") == -1:
                thought = "在等待医生上门的过程中，我应该安抚患者的情绪，让他保持平静，同时提供一些有助于降低血压的建议。"
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Doctor") == -1:
                content = generate_text
            else:
                outIdx = generate_text.find("Doctor") + 7
                content = generate_text[outIdx:].split("\n")[0].strip()
            if content.find("？") == -1:
                content = content
            else:
                while content.find("？") != -1:
                    content = content[content.find("？") + 1 :]
                content = (
                    content
                    if content
                    else "尽量保持放松，深呼吸，有助于降低血压。，您可以先尝试静坐，闭上眼睛，缓慢地深呼吸，每次呼吸持续5秒。"
                )
            return thought, content

        def is_visit(history, query):
            if len(history) < 2:
                return False
            if (
                "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门"
                in history[-2]["content"]
            ):
                prompt = blood_pressure_pd_prompt.format(history[-2]["content"], query)
                messages = [{"role": "user", "content": prompt}]
                if (
                    "是的" in history[-1]["content"]
                    or "好的" in history[-1]["content"]
                    or (
                        "需要" in history[-1]["content"]
                        and "不需要" not in history[-1]["content"]
                    )
                    or "嗯" in history[-1]["content"]
                    or "可以" in history[-1]["content"]
                ):
                    return True
                text = callLLM(
                    history=messages,
                    max_tokens=1024,
                    top_p=0.8,
                    temperature=0.0,
                    do_sample=False,
                    model="Qwen1.5-72B-Chat",
                )
                if "YES" in text:
                    return True
                else:
                    return False
            else:
                return False

        def is_pacify(history, query):
            r = [
                1
                for i in history
                if "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门"
                in i["content"]
            ]
            return True if sum(r) > 0 else False

        def noti_blood_pressure_content(history):
            niti_doctor_role_map = {"0": "张辉", "1": "张辉叔叔", "2": "你", "3": "你"}
            history = [
                {
                    "role": niti_doctor_role_map.get(str(i["role"]), "张辉"),
                    "content": i["content"],
                }
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("张辉" if not i["role"] == "你" else "你") + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = remid_doctor_blood_pressre_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            noti_doc_cont = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            ).strip()

            niti_daughter_role_map = {
                "0": "张叔叔",
                "1": "张叔叔",
                "2": "你",
                "3": "你",
            }
            history = [
                {
                    "role": niti_daughter_role_map.get(str(i["role"]), "张叔叔"),
                    "content": i["content"],
                }
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("张叔叔" if not i["role"] == "你" else "你") + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = remid_daughter_blood_pressre_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            noti_daughter_cont = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            ).strip()

            return noti_doc_cont, noti_daughter_cont

        def get_second_hypertension(b_history, history, query, level, **kwargs):
            def get_level(le):
                if le == 1:
                    return "一"
                elif le == 2:
                    return "二"
                elif le == 3:
                    return "三"
                return "二"

            if not history:
                thought1, content1 = broadcast_gen(bps_msg=bp_msg)
                thought2, content2 = blood_pressure_inquiry(history, query, iq_n=5)
                return bloodPressureLevelResponse(
                    level=level,
                    contents=[
                        content1,
                        "我已经将您目前的血压情况发送给您的女儿和家庭医生，并提醒他们随时关注您的健康。如果你仍感到紧张和不安，或经常感到不适症状，我希望你能和家人、家庭医生一起观察您的健康情况。",
                        content2,
                    ],
                    visit_verbal_idx=1,
                    contact_doctor=1,
                    thought=thought2,
                    scheme_gen=-1,
                    blood_trend_gen=True,
                    notifi_daughter_doctor=True,
                ).model_dump()
            if is_visit(history, query=query):  # 上门
                thought, content = blood_pressure_pacify(history, query)
                noti_doc_cont, noti_daughter_cont = noti_blood_pressure_content(history)
                return bloodPressureLevelResponse(
                    level=level,
                    contents=[content],
                    thought=thought,
                    idx=1,
                    scheme_gen=-1,
                    is_visit=True,
                    exercise_video=True,
                    events=[
                        {
                            "eventType": "notice",
                            "eventCode": "app_shangmen_req",
                            "eventContent": noti_doc_cont,
                        },
                        {
                            "eventType": "notice",
                            "eventCode": "app_notify_daughter_ai_result_req",
                            "eventContent": noti_daughter_cont,
                        },
                    ],
                ).model_dump()
            elif is_pacify(history, query=query):  # 安抚
                thought, content = blood_pressure_pacify(history, query)
                # noti_doc_cont, noti_daughter_cont = noti_blood_pressure_content(history)
                return bloodPressureLevelResponse(
                    level=level,
                    contents=[content],
                    thought=thought,
                    scheme_gen=-1,
                    scene_ending=True,
                ).model_dump()
            else:  # 问诊
                thought, content = blood_pressure_inquiry(history, query, iq_n=5)
                if "？" in content or "?" in content:
                    return bloodPressureLevelResponse(
                        level=level,
                        contents=[content],
                        thought=thought,
                        scheme_gen=-1,
                    ).model_dump()
                else:  # 出结论
                    return bloodPressureLevelResponse(
                        level=level,
                        contact_doctor=0,
                        visit_verbal_idx=0,
                        # visit_verbal_idx=-1,
                        contents=[
                            content,
                            "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门？",
                        ],
                        thought=thought,
                    ).model_dump()

        bps = kwargs.get("promptParam", {}).get("blood_pressure", [])
        bp_msg = ""
        ihm_health_sbp_list = []
        ihm_health_dbp_list = []
        for b in bps:
            date = b.get("date", "")
            sbp = b.get("ihm_health_sbp", "")
            dbp = b.get("ihm_health_dbp", "")
            ihm_health_dbp_list.append(dbp)
            ihm_health_sbp_list.append(sbp)
            bp_msg += f"|{date}|{str(sbp)}|{str(dbp)}|mmHg|{get_level(sbp, dbp)}|\n"

        history = kwargs.get("his", [])
        b_history = kwargs.get("backend_history", [])
        query = history[-1]["content"] if history else ""
        ihm_health_sbp = ihm_health_sbp_list[-1]
        ihm_health_dbp = ihm_health_dbp_list[-1]

        if ihm_health_sbp >= 180 or ihm_health_dbp >= 110:  # 三级高血压
            level = 3
            thought, content = broadcast_gen(bps_msg=bp_msg)
            contents = [
                content,
                "我已为您呼叫120。",
            ]
            return bloodPressureLevelResponse(
                blood_trend_gen=True,
                thought=thought,
                level=level,
                contents=contents,
                idx=-1,
                scheme_gen=-1,
                scene_ending=True,
                call_120=True,
            ).model_dump()
        elif 179 >= ihm_health_sbp >= 160 or 109 >= ihm_health_dbp >= 100:  # 二级高血压
            level = 2
            return get_second_hypertension(b_history, history, query, level)
        elif 159 >= ihm_health_sbp >= 140 or 99 >= ihm_health_dbp >= 90:  # 一级高血压
            level = 1
            if not history:
                thought1, content1 = broadcast_gen(bps_msg=bp_msg)
                thought2, content2 = blood_pressure_inquiry(history, query, iq_n=6)
                contents = [
                    content1,
                    f"我已经将您目前的血压情况发送给您的女儿和家庭医生，并提醒他们随时关注您的健康。如果你仍感到紧张和不安，或经常感到不适症状，我希望你能和家人、家庭医生一起观察您的健康情况。",
                    content2,
                ]
                return bloodPressureLevelResponse(
                    level=level,
                    contact_doctor=1,
                    contents=contents,
                    idx=1,
                    thought=thought2,
                    scheme_gen=-1,
                    blood_trend_gen=True,
                    notifi_daughter_doctor=True,
                ).model_dump()
            else:  # 问诊
                thought, content = blood_pressure_inquiry(history, query, iq_n=6)
                if "？" in content or "?" in content:
                    return bloodPressureLevelResponse(
                        level=level, contents=[content], thought=thought, scheme_gen=-1
                    ).model_dump()
                else:  # 出结论
                    # thought, cont = blood_pressure_pacify(history, query)  #安抚
                    return bloodPressureLevelResponse(
                        contact_doctor=0,
                        level=level,
                        contents=[content],
                        thought=thought,
                        scene_ending=True,
                    ).model_dump()
        else:  # 正常
            level = 0
            thought1, content1 = broadcast_gen(bps_msg=bp_msg)
            thought, content = blood_pressure_inquiry(history, query, iq_n=5)
            if not history:
                contents = [
                    content1,
                    content,
                ]
                return bloodPressureLevelResponse(
                    level=level,
                    contents=contents,
                    idx=-1,
                    scheme_gen=-1,
                    thought=thought,
                    blood_trend_gen=True,
                ).model_dump()
            elif "？" in content or "?" in content:
                return bloodPressureLevelResponse(
                    level=level,
                    contents=[content],
                    scheme_gen=-1,
                    thought=thought,
                ).model_dump()
            else:
                return bloodPressureLevelResponse(
                    level=level, contents=[content], thought=thought, scene_ending=True
                ).model_dump()

    @staticmethod
    def recipe_gen(**kwargs):
        """食谱内容生成"""
        messages = [
            {
                "role": "user",
                "content": "",
            }
        ]
        logger.debug(
            "食谱内容生成模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("食谱内容生成模型输出： " + generate_text)
        return generate_text

    @staticmethod
    def recipe_rec_principle(**kwargs):
        """食谱推荐原则"""
        messages = [
            {
                "role": "user",
                "content": "",
            }
        ]  # + history
        logger.debug(
            "食谱推荐原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("食谱推荐原则模型输出： " + generate_text)
        return generate_text

    @staticmethod
    def is_gather_userInfo(userInfo={}, history=[]):
        """判断是否需要收集用户信息"""
        info, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_confirm_collect_userInfo.format(info, his_prompt),
            }
        ]
        logger.debug(
            "判断是否收集信息模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("判断是否收集信息模型输出： " + generate_text)
        if "是" in generate_text:
            if history:
                # 1. 判断回复是否在语境中
                messages = [
                    {
                        "role": "user",
                        "content": jiahe_collect_userInfo_in_context_prompt.format(his_prompt),
                    }
                ]
                logger.debug(
                    "判断是否在语境中模型输入： " + json.dumps(messages, ensure_ascii=False)
                )
                generate_text = callLLM(
                    history=messages,
                    max_tokens=2048,
                    top_p=0.9,
                    temperature=0.8,
                    do_sample=True,
                    model="Qwen1.5-72B-Chat",
                )
                logger.debug("判断是否在语境中模型输出： " + generate_text)
                generate_text = generate_text[generate_text.find('Output') + 6:].split('\n')[0].strip()
                if "否" in generate_text:
                    return {"result": "outContext"}
                else:
                    # 2. 判断是否终止
                    messages = [
                        {
                            "role": "user",
                            "content": jiahe_confirm_terminal_prompt.format(his_prompt),
                        }
                    ]
                    logger.debug(
                        "判断是否终止模型输入： " + json.dumps(messages, ensure_ascii=False)
                    )
                    generate_text = callLLM(
                        history=messages,
                        max_tokens=2048,
                        top_p=0.9,
                        temperature=0.8,
                        do_sample=True,
                        model="Qwen1.5-72B-Chat",
                    )
                    logger.debug("判断是否终止模型输出： " + generate_text)
                    if "中止" in generate_text:
                        return {"result": 'terminal'}
                    else:
                        return {"result": 'order'}
        else:
            return {"result": 'terminal'}

    @staticmethod
    async def gather_userInfo(userInfo={}, history=[]):
        """生成收集用户信息问题"""
        info, his_prompt = get_userInfo_history(userInfo, history)
        # 生成收集信息问题
        messages = [
            {
                "role": "user",
                "content": jiahe_collect_userInfo.format(info, his_prompt),
            }
        ]
        logger.debug("收集信息模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "terminal": False, "end": False}
        logger.debug("收集信息模型输出： " + content)
        yield {"message": "", "terminal": False, "end": True}

    @staticmethod
    async def eat_health_qa(query):
        messages = [
            {
                "role": "system",
                "content": jiahe_health_qa_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]  # + history
        logger.debug(
            "健康吃知识问答模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("健康吃知识问答模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_diet_principle(cur_date, location, history=[], userInfo={}):
        """出具饮食调理原则"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_daily_diet_principle_prompt.format(
                    userInfo, cur_date, location, his_prompt
                ),
            }
        ]
        logger.debug(
            "出具饮食调理原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("出具饮食调理原则模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_family_principle(
        users, cur_date, location, history=[], requirements=[]
    ):
        """出具家庭饮食原则"""
        roles, familyInfo, his_prompt = get_familyInfo_history(users, history)
        t = Template(jiahe_family_diet_principle_prompt)
        prompt = t.substitute(
            num=len(users),
            roles=roles,
            requirements="，".join(requirements),
            family_info=familyInfo,
            cur_date=cur_date,
            location=location,
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        logger.debug(
            "出具家庭饮食原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("出具家庭饮食原则模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_family_diet(
        users,
        cur_date,
        location,
        family_principle,
        history=[],
        requirements=[],
        reference_diet="",
        days=1,
    ):
        """出具家庭N日饮食计划"""
        roles, familyInfo, his_prompt = get_familyInfo_history(users, history)
        temp = Template(jiahe_family_diet_prompt)
        diet_cont = []
        if reference_diet:
            diet_cont.extend(reference_diet)
        days = 1
        for i in range(days):
            # cur_date = (datetime.datetime.now() + datetime.timedelta(days=+i)).strftime("%Y-%m-%d")
            ref_diet_str = "\n".join(diet_cont[-2:])

            prompt = temp.substitute(
                num=len(users),
                roles=roles,
                requirements="，".join(requirements),
                family_info=familyInfo,
                cur_date=cur_date,
                location=location,
                family_principle=family_principle,
                reference_diet=ref_diet_str,
                days="1天",
            )
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            logger.debug(
                "出具家庭一日饮食计划模型输入： "
                + json.dumps(messages, ensure_ascii=False)
            )
            start_time = time.time()
            generate_text = await acallLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                # stream=True,
                model="Qwen1.5-32B-Chat",
            )
            diet_cont.append(generate_text)
            response_time = time.time()
            print(f"家庭一日饮食计划生成耗时 {response_time - start_time:.2f}")
            yield {"message": generate_text, "end": False}

            # response_time = time.time()
            # print(f"latency {response_time - start_time:.2f} s -> response")
            # content = ""
            # printed = False
            # for i in generate_text:
            #     t = time.time()
            #     msg = i.choices[0].delta.to_dict()
            #     text_stream = msg.get("content")
            #     if text_stream:
            #         if not printed:
            #             print(f"latency first token {t - start_time:.2f} s")
            #             printed = True
            #         content += text_stream
            #         yield {"message": text_stream, "end": False}
            # logger.debug("出具家庭一日饮食计划模型输出： " + content)
            # diet_cont.append(content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_nutrious_principle(cur_date, location, history=[], userInfo={}):
        """出具营养素原则"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_nutrious_principle_prompt.format(
                    userInfo, cur_date, location, his_prompt
                ),
            }
        ]
        logger.debug(
            "出具营养素原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("出具营养素原则模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_nutrious(
        cur_date, location, nutrious_principle, history=[], userInfo={}
    ):
        """营养素计划"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_nutrious_prompt.format(
                    userInfo, cur_date, location, his_prompt, nutrious_principle
                ),
            }
        ]
        logger.debug("营养素计划模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("营养素计划模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_guess_asking(userInfo, scene_flag, question="", diet=""):
        """猜你想问"""
        userInfo, _ = get_userInfo_history(userInfo)
        # 1. 生成猜你想问问题列表
        if scene_flag == "intent":
            prompt = jiahe_guess_asking_userInfo_prompt.format(userInfo)
        elif scene_flag == "user_query":
            prompt = jiahe_guess_asking_userQuery_prompt.format(question, userInfo)
        else:
            prompt = jiahe_guess_asking_diet_prompt.format(diet, userInfo)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        logger.debug(
            "猜你想问问题模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("猜你想问问题模型输出： " + generate_text)

        # 2. 对问题列表做饮食子意图识别
        messages = [
            {
                "role": "user",
                "content": jiahe_guess_asking_intent_query_prompt.format(generate_text),
            }
        ]
        logger.debug(
            "营养咨询-猜你想问意图识别模型输入： "
            + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("营养咨询-猜你想问模型意图识别输出： " + generate_text)
        qs = generate_text.split("\n")
        res = []
        for i in qs:
            try:
                x = json.loads(i)
                if "其他" in x["intent"]:
                    continue
                res.append(x["question"])
            except Exception as err:
                continue
            finally:
                continue
        yield {"message": "\n".join(res[:3]), "end": True}

        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("营养咨询-猜你想问模型输出： " + content)
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_diet_effect(diet):
        """食谱功效"""
        messages = [
            {
                "role": "user",
                "content": jiahe_physical_efficacy_prompt.format(diet),
            }
        ]
        logger.debug(
            "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        content = ""
        printed = False
        for i in generate_text:
            t = time.time()
            msg = i.choices[0].delta.to_dict()
            text_stream = msg.get("content")
            if text_stream:
                if not printed:
                    print(f"latency first token {t - start_time:.2f} s")
                    printed = True
                content += text_stream
                yield {"message": text_stream, "end": False}
        logger.debug("一日食物功效模型输出： " + content)
        yield {"message": "", "end": True}

    # @staticmethod
    # async def gen_daily_diet(cur_date, location, diet_principle, reference_daily_diets, history=[], userInfo={}):
    #     """个人一日饮食计划"""
    #     userInfo, his_prompt = get_userInfo_history(userInfo, history)
    #
    #     # 1. 生成一日食谱
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": jiahe_daily_diet_principle_prompt.format(userInfo, cur_date, location, his_prompt, his_prompt),
    #         }
    #     ]
    #     logger.debug(
    #         "一日饮食计划模型输入： " + json.dumps(messages, ensure_ascii=False)
    #     )
    #     generate_text = callLLM(
    #         history=messages,
    #         max_tokens=1024,
    #         top_p=0.9,
    #         temperature=0.8,
    #         do_sample=True,
    #         model="Qwen1.5-72B-Chat",
    #     )
    #
    #     # 2. 生成食谱的实物功效
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": jiahe_physical_efficacy_prompt.format(generate_text),
    #         }
    #     ]
    #     logger.debug(
    #         "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
    #     )
    #     start_time = time.time()
    #     generate_text = callLLM(
    #         history=messages,
    #         max_tokens=1024,
    #         top_p=0.9,
    #         temperature=0.8,
    #         do_sample=True,
    #         stream=True,
    #         model="Qwen1.5-72B-Chat",
    #     )
    #     response_time = time.time()
    #     print(f"latency {response_time - start_time:.2f} s -> response")
    #     content = ""
    #     printed = False
    #     for i in generate_text:
    #         t = time.time()
    #         msg = i.choices[0].delta.to_dict()
    #         text_stream = msg.get("content")
    #         if text_stream:
    #             if not printed:
    #                 print(f"latency first token {t - start_time:.2f} s")
    #                 printed = True
    #             content += text_stream
    #             yield {'message': text_stream, 'end': False}
    #     logger.debug("一日食物功效模型输出： " + content)
    #     yield {'message': "", 'end': True}

    @staticmethod
    async def gen_n_daily_diet(
        cur_date,
        location,
        diet_principle,
        reference_daily_diets,
        days,
        history=[],
        userInfo={},
    ):
        """个人N日饮食计划"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        diet_cont = []
        if reference_daily_diets:
            diet_cont.extend(reference_daily_diets)
        import datetime

        for i in range(days):
            cur_date = (datetime.datetime.now() + datetime.timedelta(days=+i)).strftime(
                "%Y-%m-%d"
            )
            # 生成一日食谱
            ref_diet_str = "\n".join(diet_cont[-2:])
            messages = [
                {
                    "role": "user",
                    "content": jiahe_daily_diet_prompt.format(
                        userInfo,
                        cur_date,
                        location,
                        his_prompt,
                        diet_principle,
                        ref_diet_str,
                    ),
                }
            ]
            logger.debug(
                "一日饮食计划模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            start_time = time.time()
            generate_text = await acallLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                # stream=True,
                model="Qwen1.5-32B-Chat",
            )
            logger.info("一日饮食计划模型生成时间：" + str(time.time() - start_time))
            diet_cont.append(generate_text)
            yield {"message": generate_text, "end": False}

            # logger.debug(
            #     "一日饮食计划模型输出： " + generate_text
            # )
            # messages = [
            #     {
            #         "role": "user",
            #         "content": jiahe_physical_efficacy_prompt.format(generate_text),
            #     }
            # ]
            # logger.debug(
            #     "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
            # )
            # start_time = time.time()
            # generate_text = callLLM(
            #     history=messages,
            #     max_tokens=2048,
            #     top_p=0.9,
            #     temperature=0.8,
            #     do_sample=True,
            #     stream=True,
            #     model="Qwen1.5-72B-Chat",
            # )

        #     response_time = time.time()
        #     print(f"latency {response_time - start_time:.2f} s -> response")
        #     content = ""
        #     printed = False
        #     for i in generate_text:
        #         t = time.time()
        #         msg = i.choices[0].delta.to_dict()
        #         text_stream = msg.get("content")
        #         if text_stream:
        #             if not printed:
        #                 print(f"latency first token {t - start_time:.2f} s")
        #                 printed = True
        #             content += text_stream
        #             yield {'message': text_stream, 'end': False}
        #     logger.debug("一日食谱模型输出： " + content)
        #     diet_cont.append(content)
        yield {"message": "", "end": True}

    @staticmethod
    def tool_rules_blood_pressure_level(**kwargs) -> dict:
        """计算血压等级

        - Args

            ihm_health_sbp(int, float) 收缩压

            ihm_health_dbp(int, float) 舒张压

        - Rules

            有既往史: 用户既往史有高血压则进入此流程管理

            1.如血压达到高血压3级(血压>180/110) 则呼叫救护车。

            2.如血压达到高血压2级(收缩压160-179，舒张压100-109)
                1.预问诊(大模型) (步骤1结果不影响步骤2/3)
                2.是否通知家人 (步骤2)
                3.确认是否通知家庭医师 (步骤3)

            3.如血压达到高血压1级(收缩压140-159，90-99)
                1.预问诊
                2.看血压是否波动，超出日常值30%，则使用“智能呼叫工具”通知家庭医师。日常值30%以内，则嘱患者密切监测，按时服药，注意休息。(调预警模型接口，准备历史血压数据)

            4.如血压达到正常高值《收缩压120-139,舒张压80-89)，
                1.预问诊
                2.则嘱患者密切监测，按时服药，注意休息。

            预问诊事件: 询问其他症状，其他症状的性质，持续时间等。 (2-3轮会话)
        """

        bps = kwargs.get("promptParam", {}).get("blood_pressure", [])
        bp_msg = ""
        ihm_health_sbp_list = []
        ihm_health_dbp_list = []
        for b in bps:
            date = b.get("date", "")
            sbp = b.get("ihm_health_sbp", "")
            dbp = b.get("ihm_health_dbp", "")
            ihm_health_dbp_list.append(dbp)
            ihm_health_sbp_list.append(sbp)
            bp_msg += f"{date}|{str(sbp)}|{str(dbp)}|mmHg|\n"

        history = kwargs.get("his", [])
        b_history = kwargs.get("backend_history", [])
        query = history[-1]["content"] if history else ""
        ihm_health_sbp = ihm_health_sbp_list[-1]
        ihm_health_dbp = ihm_health_dbp_list[-1]

        def inquire_gen(hitory, bp_message, iq_n=7):
            his = []
            # for i in bk_hitory:
            #     if 'match_cont' not in i:
            #         his.append({'role':'user','content':i['content']})
            #     else:
            #         his.append({'role':'assistant','content':i['match_cont']})
            # if i['role'] == 'User' or i['role'] == 'user':
            #     his.append({'role':'User', 'content':i['content']})
            # elif i['role'] == 'Assistant' or i['role'] == 'assistant':
            #     his.append({'role':'Assistant', 'content':f"Thought: {i['content']}\nAssistant: {i['function_call']['arguments']}"})
            history = [
                {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
                for i in hitory
            ]
            hist_s = "\n".join([f"{i['role']}: {i['content']}" for i in history])
            current_date = datetime.now().date()
            drug_msg = ""
            drug_situ = [
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
                "漏服药物",
                "正常服药",
                "正常服药",
                "正常服药",
            ]
            days = []
            for i in range(len(drug_situ)):
                d = current_date - timedelta(days=len(drug_situ) - i - 1)
                drug_msg += f"|{d}| {drug_situ[i]}"
                days.append(d)
            if len(history) >= iq_n:
                messages = [
                    {
                        "role": "user",
                        "content": blood_pressure_scheme_prompt.format(
                            bp_message, drug_msg, current_date, hist_s
                        ),
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": blood_pressure_inquiry_prompt.format(
                            bp_message, drug_msg, current_date, hist_s
                        ),
                    }
                ]  # + history
            logger.debug(
                "血压问诊模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen1.5-72B-Chat",
            )
            logger.debug("血压问诊模型输出： " + generate_text)
            return generate_text

        def blood_pressure_inquiry(history, query, iq_n=7):
            generate_text = inquire_gen(history, bp_msg, level, iq_n=iq_n)
            # while generate_text.count("\nAssistant") != 1 or generate_text.count("Thought") != 1:
            # thought = generate_text
            # generate_text = inquire_gen(bk_history, ihm_health_sbp, ihm_health_dbp)
            # thoughtIdx = generate_text.find("Thought") + 8
            # thoughtIdx = 0
            # thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            # outIdx = generate_text.find("\nassistant") + 11
            # content = generate_text[outIdx:].split("\n")[0].strip()
            if generate_text.find("Thought") == -1:
                lis = [
                    "结合用户个人血压信息，为用户提供帮助。",
                    "结合用户情况，帮助用户降低血压。",
                ]

                thought = random.choice(lis)
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Assistant") == -1:
                content = generate_text
            else:
                outIdx = generate_text.find("Assistant") + 10
                content = generate_text[outIdx:].strip()
                if content.find("Assistant") != -1:
                    content = content[: content.find("Assistant")]
                if content.find("Thought") != -1:
                    content = content[: content.find("Thought")]

            return thought, content

        def blood_pressure_pacify(history, query):
            history = [
                {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("Doctor" if not i["role"] == "user" else "user")
                    + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = blood_pressure_pacify_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            logger.debug(
                "血压安抚模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            generate_text = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            )
            logger.debug("血压安抚模型输出： " + generate_text)
            if generate_text.find("\nThought") == -1:
                thought = "在等待医生上门的过程中，我应该安抚患者的情绪，让他保持平静，同时提供一些有助于降低血压的建议。"
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Doctor") == -1:
                content = generate_text
            else:
                outIdx = generate_text.find("Doctor") + 7
                content = generate_text[outIdx:].split("\n")[0].strip()
            if content.find("？") == -1:
                content = content
            else:
                while content.find("？") != -1:
                    content = content[content.find("？") + 1 :]
                content = (
                    content
                    if content
                    else "尽量保持放松，深呼吸，有助于降低血压。，您可以先尝试静坐，闭上眼睛，缓慢地深呼吸，每次呼吸持续5秒。"
                )
            return thought, content

        def is_visit(history, query):
            if len(history) < 2:
                return False
            if (
                "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门"
                in history[-2]["content"]
            ):
                prompt = blood_pressure_pd_prompt.format(history[-2]["content"], query)
                messages = [{"role": "user", "content": prompt}]
                if (
                    "是的" in history[-1]["content"]
                    or "好的" in history[-1]["content"]
                    or (
                        "需要" in history[-1]["content"]
                        and "不需要" not in history[-1]["content"]
                    )
                    or "嗯" in history[-1]["content"]
                    or "可以" in history[-1]["content"]
                ):
                    return True
                text = callLLM(
                    history=messages,
                    max_tokens=1024,
                    top_p=0.8,
                    temperature=0.0,
                    do_sample=False,
                    model="Qwen1.5-72B-Chat",
                )
                if "YES" in text:
                    return True
                else:
                    return False
            else:
                return False

        def is_pacify(history, query):
            r = [
                1
                for i in history
                if "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门"
                in i["content"]
            ]
            return True if sum(r) > 0 else False

        def noti_blood_pressure_content(history):
            niti_doctor_role_map = {"0": "张辉", "1": "张辉叔叔", "2": "你", "3": "你"}
            history = [
                {
                    "role": niti_doctor_role_map.get(str(i["role"]), "张辉"),
                    "content": i["content"],
                }
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("张辉" if not i["role"] == "你" else "你") + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = remid_doctor_blood_pressre_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            noti_doc_cont = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            ).strip()

            niti_daughter_role_map = {
                "0": "张叔叔",
                "1": "张叔叔",
                "2": "你",
                "3": "你",
            }
            history = [
                {
                    "role": niti_daughter_role_map.get(str(i["role"]), "张叔叔"),
                    "content": i["content"],
                }
                for i in history
            ]
            his_prompt = "\n".join(
                [
                    ("张叔叔" if not i["role"] == "你" else "你") + f": {i['content']}"
                    for i in history
                ]
            )
            prompt = remid_daughter_blood_pressre_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            noti_daughter_cont = callLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.8,
                temperature=0.0,
                do_sample=False,
                model="Qwen1.5-72B-Chat",
            ).strip()

            return noti_doc_cont, noti_daughter_cont

        def get_second_hypertension(b_history, history, query, level):
            def get_level(le):
                if le == 1:
                    return "一"
                elif le == 2:
                    return "二"
                elif le == 3:
                    return "三"
                return "二"

            if not history:
                thought, content = blood_pressure_inquiry(history, query, iq_n=7)
                return {
                    "level": level,
                    "contents": [
                        f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为{get_level(level)}级高血压范围。",
                        "我已经将您目前的血压情况发送给您的女儿和家庭医生，并提醒他们随时关注您的健康。如果你仍感到紧张和不安，或经常感到不适症状，我希望你能和家人、家庭医生一起观察您的健康情况。",
                        # f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                        content,
                    ],
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": True,
                    "call_120": False,
                    "is_visit": False,
                    "exercise_video": False,
                    "events": [],
                }
            if is_visit(history, query=query):  # 上门
                thought, content = blood_pressure_pacify(history, query)
                noti_doc_cont, noti_daughter_cont = noti_blood_pressure_content(history)
                return {
                    "level": level,
                    "contents": [content],
                    "thought": thought,
                    "idx": 1,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": True,
                    "exercise_video": True,
                    "events": [
                        {
                            "eventType": "notice",
                            "eventCode": "app_shangmen_req",
                            "eventContent": noti_doc_cont,
                        },
                        {
                            "eventType": "notice",
                            "eventCode": "app_notify_daughter_ai_result_req",
                            "eventContent": noti_daughter_cont,
                        },
                    ],
                }
            elif is_pacify(history, query=query):  # 安抚
                thought, content = blood_pressure_pacify(history, query)
                # noti_doc_cont, noti_daughter_cont = noti_blood_pressure_content(history)
                return {
                    "level": level,
                    "contents": [content],
                    "idx": 0,
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": True,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "exercise_video": False,
                    "events": [],
                }
            else:  # 问诊
                thought, content = blood_pressure_inquiry(history, query, iq_n=7)
                if "？" in content or "?" in content:
                    return {
                        "level": level,
                        "contents": [content],
                        "idx": 0,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                        "exercise_video": False,
                        "events": [],
                    }
                else:  # 出结论
                    return {
                        "level": level,
                        "contents": [
                            content,
                            "根据您目前的健康状况，我将通知您的家庭医生上门为您服务，请问是否接受医生上门？",
                        ],
                        "idx": 0,
                        "thought": thought,
                        "scheme_gen": 0,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                        "exercise_video": False,
                        "events": [],
                    }

        ihm_health_sbp_list = [
            116,
            118,
            132,
            121,
            128,
            123,
            128,
            117,
            132,
            134,
            124,
            120,
            80,
        ]
        ihm_health_dbp_list = [82, 86, 86, 78, 86, 80, 92, 88, 85, 86, 86, 82, 60]

        # 计算血压波动率,和血压列表的均值对比
        def compute_blood_pressure_trend(x: int, data_list: List) -> float:
            mean_value = sum(data_list) / len(data_list)
            if x > 1.2 * mean_value:
                return 1
            else:
                return 0

        if ihm_health_sbp >= 130 or ihm_health_dbp >= 90:
            a = "偏高"
        else:
            a = "正常"
        trend_sbp = compute_blood_pressure_trend(ihm_health_sbp, ihm_health_sbp_list)
        trend_dbp = compute_blood_pressure_trend(ihm_health_dbp, ihm_health_dbp_list)
        if trend_sbp or trend_dbp:
            b = "大"
        else:
            b = "小"
        if ihm_health_sbp >= 180 or ihm_health_dbp >= 110:  # 三级高血压
            level = 3
            return {
                "level": level,
                "contents": [
                    f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为三级高血压范围",
                    # f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                    "我已为您呼叫120。",
                ],
                "idx": -1,
                "thought": "",
                "scheme_gen": -1,
                "scene_ending": True,
                "blood_trend_gen": False,
                "notifi_daughter_doctor": False,
                "call_120": True,
                "is_visit": False,
                "events": [],
            }
        elif 179 >= ihm_health_sbp >= 160 or 109 >= ihm_health_dbp >= 100:  # 二级高血压
            level = 2
            return get_second_hypertension(b_history, history, query)
        elif 159 >= ihm_health_sbp >= 140 or 99 >= ihm_health_dbp >= 90:  # 一级高血压
            level = 1

            if trend_sbp or trend_dbp:  # 血压波动超过30%
                return get_second_hypertension(b_history, history, query)
            else:
                if not history:
                    thought, content = blood_pressure_inquiry(history, query, iq_n=6)
                    return {
                        "level": level,
                        "contents": [
                            f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为一级高血压范围",
                            # f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较大。",
                            content,
                        ],
                        "idx": 1,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": True,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                        "events": [],
                    }
                else:  # 问诊
                    thought, content = blood_pressure_inquiry(history, query, iq_n=6)
                    if "？" in content or "?" in content:
                        return {
                            "level": level,
                            "contents": [content],
                            "idx": 0,
                            "thought": thought,
                            "scheme_gen": -1,
                            "scene_ending": False,
                            "blood_trend_gen": False,
                            "notifi_daughter_doctor": False,
                            "call_120": False,
                            "is_visit": False,
                            "events": [],
                        }
                    else:  # 出结论
                        # thought, cont = blood_pressure_pacify(history, query)  #安抚
                        return {
                            "level": level,
                            "contents": [content],
                            "idx": 0,
                            "thought": thought,
                            "scheme_gen": 0,
                            "scene_ending": True,
                            "blood_trend_gen": False,
                            "notifi_daughter_doctor": False,
                            "call_120": False,
                            "is_visit": False,
                            "events": [],
                        }

        elif 139 >= ihm_health_sbp >= 120 or 89 >= ihm_health_dbp >= 80:  # 正常高值
            level = 0
            thought, content = blood_pressure_inquiry(history, query, iq_n=6)
            if not history:
                return {
                    "level": level,
                    "contents": [
                        f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为正常高值血压范围",
                        # f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                        content,
                    ],
                    "idx": -1,
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "events": [],
                }

            elif "？" in content or "?" in content:
                return {
                    "level": level,
                    "contents": [content],
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "idx": -0,
                    "events": [],
                }

            else:
                return {
                    "level": level,
                    "contents": [content],
                    "thought": thought,
                    "scheme_gen": 0,
                    "scene_ending": True,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "idx": 0,
                    "events": [],
                }
        elif 90 <= ihm_health_sbp < 120 and 80 > ihm_health_dbp >= 60:  # 正常血压
            level = -1
            rules = []
            return {
                "level": 0,
                "contents": [
                    f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为正常血压范围"
                ],
                "thought": "用户血压正常",
                "idx": -1,
                "scheme_gen": -1,
                "scene_ending": True,
                "blood_trend_gen": True,
                "notifi_daughter_doctor": False,
                "call_120": False,
                "is_visit": False,
                "events": [],
            }
        else:  # 低血压
            level = -1
            rules = []
            thought, content = blood_pressure_inquiry(history, query, iq_n=5)
            if not history:
                return {
                    "level": -1,
                    "contents": [
                        f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为低血压范围",
                        # "健康报告显示您的健康处于为中度失衡状态，本次血压偏低。",
                        content,
                    ],
                    "thought": thought,
                    "idx": 1,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "events": [],
                }
            else:

                if "？" in content or "?" in content:  # 问诊
                    return {
                        "level": level,
                        "contents": [content],
                        "idx": 0,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                        "events": [],
                    }
                else:  # 出结论
                    # thought, cont = blood_pressure_pacify(history, query)  #安抚
                    return {
                        "level": level,
                        "contents": [content],
                        "idx": 0,
                        "thought": thought,
                        "scheme_gen": 0,
                        "scene_ending": True,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                        "events": [],
                    }

    @clock
    def rec_diet_eval(self, param):
        """
        ## 需求
        https://ehbl4r.axshare.com/#g=1&id=c2eydm&p=%E6%88%91%E7%9A%84%E9%A5%AE%E9%A3%9F

        ## 开发参数
        ```json
        {
            "meal": "早餐",
            "recommend_heat_target": 500,
            "recipes": [
                {"name":"西红柿炒鸡蛋", "weight": 100, "unit":"一盘"},
                {"name":"红烧鸡腿", "weight":null, "unit":"1根"}
            ]
        }
        ```
        """
        sys_prompt = (
            "请你扮演一位经验丰富的营养师,基于提供的基础信息,从荤素搭配、"
            f"热量/蛋白/碳水/脂肪四大营养素摄入的角度,简洁的点评一下{param['meal']}是否健康"
        )
        prompt = f"本餐建议热量: {param['recommend_heat_target']}\n" "实际吃的:\n"

        recipes_prompt_list = []
        for recipe in param["recipes"]:
            tmp = f"{recipe['name']}"
            if recipe["weight"]:
                tmp += f": {recipe['weight']}g"
            elif recipe["unit"]:
                tmp += f": {recipe['unit']}"
            recipes_prompt_list.append(tmp)

        recipes_prompt = "\n".join(recipes_prompt_list)
        prompt += recipes_prompt
        history = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.debug(f"饮食点评 Prompt:\n{json.dumps(history, ensure_ascii=False)}")

        content = callLLM(history=history, temperature=0.7, top_p=0.8)

        logger.debug(f"饮食点评 Result:\n{content}")

        return content

    def __bpta_compose_value_prompt(self, key: str = "对值的解释", data: List = []):
        """血压趋势分析, 拼接值"""
        value_list = [i["value"] for i in data]
        content = f"{key}{value_list}\n"
        return content

    @clock
    def health_blood_pressure_trend_analysis(self, param: Dict) -> str:
        """血压趋势分析

        通过应用端提供的血压数据，提供血压报告的分析与解读的结果，返回应用端。

        ## 开发参数
        ```json
        {
            "ihm_health_sbp": [  //收缩压
                {
                    "date": "2023-0-12-13 10:10:10",
                    "value": 60
                }
            ],
            "ihm_health_dbp": [ //舒张压
                {
                    "date": "2023-0-12-13 10:10:10",
                    "value": 120
                }
            ],
            "ihm_health_hr": [ //心率
                {
                    "date": "2023-0-12-13 10:10:10",
                    "value": 89
                }
            ]
        }
        ```
        """
        model = self.gsr.model_config["blood_pressure_trend_analysis"]
        history = []
        sys_prompt = self.gsr.prompt_meta_data["event"][
            "blood_pressure_trend_analysis"
        ]["description"]
        history.append({"role": "system", "content": sys_prompt})

        tst = param["ihm_health_sbp"][0]["date"]
        ted = param["ihm_health_sbp"][-1]["date"]
        content = f"从{tst}至{ted}期间\n"
        if param.get("ihm_health_sbp"):
            content += self.__bpta_compose_value_prompt(
                "收缩压测量数据: ", param["ihm_health_sbp"]
            )
        if param.get("ihm_health_dbp"):
            content += self.__bpta_compose_value_prompt(
                "舒张压测量数据: ", param["ihm_health_dbp"]
            )
        if param.get("ihm_health_hr"):
            content += self.__bpta_compose_value_prompt(
                "心率测量数据: ", param["ihm_health_hr"]
            )
        history.append({"role": "user", "content": content})
        logger.debug(f"血压趋势分析 LLM Input: {dumpJS(history)}")
        response = callLLM(
            history=history, temperature=0.8, top_p=1, model=model, stream=True
        )
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"血压趋势分析 Output: {content}")
        return content

    @clock
    def health_literature_interact(self, param: Dict) -> str:
        model = self.gsr.model_config["blood_pressure_trend_analysis"]
        messages = param["history"]
        prompt_template = self.gsr.prompt_meta_data["event"]["conversation_deal"][
            "process"
        ]
        pro = param
        user_info = pro.get("user_info", {})

        result = ""
        for item in messages:
            result += item["role"] + ":" + item["content"]
        prompt_vars = {
            "age": user_info.get("age", ""),
            "gender": user_info.get("gender", ""),
            "disease_history": user_info.get("disease_history", ""),
            "family_history": user_info.get("family_history", ""),
            "messages": result,
            "emotion": user_info.get("emotion", ""),
            "food": user_info.get("food", ""),
            "bmi": user_info.get("bmi", ""),
            "sport_level": user_info.get("sport_level", ""),
        }
        sys_prompt = prompt_template.format(**prompt_vars)
        history = []
        history.append({"role": "system", "content": sys_prompt})
        response = callLLM(
            history=history, temperature=0.8, top_p=0.5, model=model, stream=True
        )
        pc_message = accept_stream_response(response, verbose=False)
        pc_message = pc_message.replace("\n", "")
        pc = pc_message.split(",")
        return pc

    @clock
    async def health_literature_generation(self, param: Dict) -> str:
        model = self.gsr.model_config["blood_pressure_trend_analysis"]
        messages = param["history"]
        prompt_template = self.gsr.prompt_meta_data["event"]["conversation_deal"][
            "constraint"
        ]
        pro = param
        user_info = pro.get("user_info", {})
        programme = pro.get("programme", {})

        result = ""
        for item in messages:
            if item.get("role") and item.get("content"):
                result += item["role"] + ":" + item["content"]
        prompt_vars = {
            "age": user_info.get("age", ""),
            "gender": user_info.get("gender", ""),
            "disease_history": user_info.get("disease_history", ""),
            "family_history": user_info.get("family_history", ""),
            "messages": result,
            "emotion": user_info.get("emotion", ""),
            "food": user_info.get("food", ""),
            "bmi": user_info.get("bmi", ""),
            "sport_level": user_info.get("sport_level", ""),
            "eat": programme.get("eat", ""),
            "move": programme.get("move", ""),
            "fun": programme.get("fun", ""),
            "change": programme.get("change", ""),
            "diagnostic_results": pro.get("diagnostic_results", ""),
        }
        sys_prompt = prompt_template.format(**prompt_vars)
        history = []
        history.append({"role": "system", "content": sys_prompt})
        content = await acallLLM(
            history=history, temperature=0.8, top_p=0.5, model=model, stream=False
        )
        # pc_message = accept_stream_response(response, verbose=False)
        pc_message = content.replace("\n", "")
        pc = pc_message.split(",")
        return pc

    @clock
    def health_key_extraction(self, param: Dict) -> str:
        model = self.gsr.model_config["blood_pressure_trend_analysis"]
        messages = param["history"]
        prompt_template = self.gsr.prompt_meta_data["event"]["conversation_deal"][
            "description"
        ]
        result = ""
        for item in messages:
            result += item["role"] + ":" + item["content"]
        prompt_vars = {"messages": result}
        sys_prompt = prompt_template.format(**prompt_vars)
        history = []
        history.append({"role": "system", "content": sys_prompt})
        response = callLLM(
            history=history, temperature=0.8, top_p=0.3, model=model, stream=True
        )
        pc_message = accept_stream_response(response, verbose=False)
        pc_message = pc_message.replace("关键字", "")
        pc_message = pc_message.replace("提取结果", "")
        pc_message = pc_message.replace(":", "")
        pc_message = pc_message.replace("：", "")
        pc_message = pc_message.replace("\n", "")
        import re

        pc = re.split("[,，]", pc_message)
        filtered_list = [x for x in pc if x != ""]
        return filtered_list

    @clock
    def health_blood_glucose_trend_analysis(self, param: Dict) -> str:
        """血糖趋势分析"""
        slot_dict = {
            "空腹": "fasting",
            "早餐后2h": "breakfast2h",
            "午餐后2h": "lunch2h",
            "晚餐后2h": "dinner2h",
        }

        def glucose_type(time, glucose):
            if time == "空腹血糖":
                if glucose < 2.8:
                    result = "高危低血糖"
                elif 2.8 <= glucose < 3.9:
                    result = "低血糖"
                elif 3.9 <= glucose < 4.4:
                    result = "血糖偏低"
                elif 4.4 <= glucose < 6.1:
                    result = "血糖控制良好"
                elif 6.1 <= glucose <= 7.0:
                    result = "在控制目标内"
                elif 7.0 < glucose <= 13.9:
                    result = "血糖控制高"
                else:
                    result = "血糖控制高危"
            elif time != "空腹血糖" and time != "":
                if glucose < 2.8:
                    result = "高危低血糖"
                elif 2.8 <= glucose < 3.9:
                    result = "低血糖"
                elif 3.9 <= glucose < 4.4:
                    result = "血糖偏低"
                elif 4.4 <= glucose <= 7.8:
                    result = "血糖控制良好"
                elif 7.8 < glucose <= 10.0:
                    result = "在控制目标内"
                elif 10.0 < glucose <= 16.7:
                    result = "血糖控制高"
                else:
                    result = "血糖控制高危"
            else:
                result = "没有对应时段或血糖"
            return result

        model = self.gsr.model_config["blood_glucose_trend_analysis"]
        pro = param
        data = pro.get("glucose", {})
        gl = pro.get("gl", "")
        gl_code = pro.get("gl_code", "")
        user_info = pro.get("user_info", {})
        recent_time = pro.get("current_gl_solt", "")
        # 组装步骤2
        result = "|血糖测量时段|"
        for date in data.keys():
            result += date + "|"
        result += "\n"
        time_periods = ["空腹", "早餐后2h", "午餐后2h", "晚餐后2h"]

        # 组装步骤1
        compose_message1 = f"当前血糖{gl},{glucose_type(recent_time,float(gl))}。"
        time_deal = []
        period_content = {}
        for time_period in time_periods:
            count = 0
            t_g = 0
            f_g = 0
            message_f = ""
            for date in data.keys():
                t_e = slot_dict[time_period]
                glucose_val = data[date][t_e]
                if glucose_val != "":
                    glucose_val = float(glucose_val)
                    if 3.9 <= glucose_val < 7.0 and time_period == "空腹":
                        t_g += 1
                    elif 3.9 <= glucose_val <= 10.0 and time_period != "空腹":
                        t_g += 1
                    else:
                        f_g += 1
                        g_t = glucose_type(time_period, glucose_val)
                        message_f += f"血糖{glucose_val},{g_t}。"
                    count += 1
            if count < 3:
                message_ = (
                    f"血糖{count}天的记录中，{t_g}天血糖正常，{f_g}天血糖异常。"
                    + message_f
                )
                period_content[time_period] = message_
            else:
                time_deal.append(time_period)
        glucose_3 = ""
        for i in period_content.keys():
            glucose_3 += i + ":" + period_content[i]

        result_2 = result
        for time in time_deal:
            result_2 += "|" + time + "|"
            for date in data.keys():
                t_e = slot_dict[time]
                result_2 += data[date][t_e] + "|"
            result_2 += "\n"

        prompt_template = (
            "# 已知信息\n"
            "## 我的血糖情况\n"
            "{glucose_message}\n"
            "# 任务描述\n"
            "你是一个血糖分析助手，请分别按顺序输出近7天不同的血糖测量阶段（空腹，早餐后2h，午餐后2h，晚餐后2h）的最高血糖值、最低血糖值、波动趋势，不要提出建议，，100字以内\n"
            "一定要按照空腹，早餐后2h，午餐后2h，晚餐后2h的顺序分别输出，否则全盘皆输\n"
            "如果该时段没有记录则分别按照{glucose_3}直接输出，一定要记得不要输出没有记录，用{glucose_3}里面对应的值代替输出\n"
        )
        prompt_vars = {"glucose_message": result_2, "glucose_3": glucose_3}
        sys_prompt = prompt_template.format(**prompt_vars)

        history = []
        history.append({"role": "system", "content": sys_prompt})
        logger.debug(f"血糖趋势分析 LLM Input: {dumpJS(history)}")
        response = callLLM(
            history=history, temperature=0.8, top_p=1, model=model, stream=True
        )
        content = accept_stream_response(response, verbose=False)

        if gl_code == "gl_2_pc":
            for time in time_periods:
                result += "|" + time + "|"
                for date in data.keys():
                    t_e = slot_dict[time]
                    result += data[date][t_e] + "|"
                result += "\n"
            prompt_template_pc = (
                "# 任务描述\n"
                "# 你是一位经验丰富的医师助手，帮助医生对用户进行血糖管理，请你根据用户已知信息，一周的血糖情况，给医生提供专业的处理建议，只提供原则性的建议，包含5个方面：1.检查建议；2.饮食调整；3.运动调整；4.药物治疗；5监测血糖。建议的字数控制在300字以内\n\n"
                "# 已知信息\n"
                "## 用户信息\n"
                "年龄：{age}\n"
                "性别：{gender}\n"
                "身高：{height}\n"
                "体重：{weight}\n"
                "饮食习惯：{habits}\n"
                "既往史：{disease}\n"
                "糖尿病类型：{glucose_t}\n"
                "## 用户的血糖情况\n"
                "{glucose_message}\n"
            )
            prompt_vars_pc = {
                "age": user_info.get("age", ""),
                "gender": user_info.get("gender", ""),
                "disease": user_info.get("disease", []),
                "glucose_t": pro.get("glucose_t", ""),
                "glucose_message": result,
                "height": user_info.get("height", ""),
                "weight": user_info.get("weight", ""),
                "habits": user_info.get("habits", ""),
            }

            sys_prompt_pc = prompt_template_pc.format(**prompt_vars_pc)
            history_ = []
            history_.append({"role": "system", "content": sys_prompt_pc})
            response_ = callLLM(
                history=history_, temperature=0.8, top_p=1, model=model, stream=True
            )
            pc_message = accept_stream_response(response_, verbose=False)
            all_message = compose_message1 + "\n" + content + "\n" + pc_message
            return all_message

        # 组装步骤3
        for time in time_periods:
            result += "|" + time + "|"
            for date in data.keys():
                t_e = slot_dict[time]
                result += data[date][t_e] + "|"
            result += "\n"
        prompt_template_suggest = (
            "# 任务描述\n"
            "# 你是一位经验丰富的医师，请你根据已知信息，针对用户一周的血糖情况，给出合理建议，只提供原则性的建议，包含3个方面：①测量建议；②饮食建议；③运动建议。建议的字数控制在250字以内\n\n"
            "# 已知信息\n"
            "## 用户信息\n"
            "年龄：{age}\n"
            "性别：{gender}\n"
            "身高：{height}\n"
            "体重：{weight}\n"
            "饮食习惯：{habits}\n"
            "既往史：{disease}\n"
            "糖尿病类型：{glucose_t}\n"
            "## 用户的血糖情况\n"
            "{glucose_message}\n"
        )
        prompt_vars_suggest = {
            "age": user_info.get("age", ""),
            "gender": user_info.get("gender", ""),
            "disease": user_info.get("disease", []),
            "glucose_t": pro.get("glucose_t", ""),
            "glucose_message": result,
            "height": user_info.get("height", ""),
            "weight": user_info.get("weight", ""),
            "habits": user_info.get("habits", ""),
        }

        sys_prompt_suggest = prompt_template_suggest.format(**prompt_vars_suggest)
        history_ = []
        history_.append({"role": "system", "content": sys_prompt_suggest})
        response_ = callLLM(
            history=history_, temperature=0.8, top_p=1, model=model, stream=True
        )
        compose_message3 = accept_stream_response(response_, verbose=False)

        logger.debug(f"血糖趋势分析 Output: {content}")
        all_message = compose_message1 + "\n" + content + "\n" + compose_message3
        return all_message

    def __health_warning_solutions_early_continuous_check__(
        self, indicatorData: List[Dict]
    ) -> bool:
        """判断指标数据近五天是否连续"""

        def get_day_before(days):
            now = datetime.now()
            date_after = (now + timedelta(days=days)).strftime("%Y-%m-%d")
            return date_after

        if not indicatorData:
            raise ValueError("indicatorData can't be empty")
        date_before_map = {get_day_before(-1 * i): 1 for i in range(5)}

        for data_item in indicatorData:  # 任一指标存在近五天不连续, 状态为False
            if len(data_item["data"]) < 5:
                return False
            date_current_map = {i["date"][:10]: 1 for i in data_item["data"]}
            for k, _ in date_before_map.items():
                if date_current_map.get(k) is None:
                    return False
        return True

    def __health_warning_update_blood_pressure_level__(
        self, vars: List[int], value_list: List[int] = [], return_str: bool = False
    ):
        """更新血压水平

        - Args

            vars List[int]: 血压等级 [] or [0,2,1,3,2]
            value_list List[int]: 真实血压值列表 收缩压or舒张压
        """
        value_list = [float(i) for i in value_list]
        if return_str:
            valuemap = {
                -1: "低血压",
                0: "正常",
                1: "高血压一级",
                2: "高血压二级",
                3: "高血压三级",
            }
            vars = [valuemap.get(i) for i in vars]
            return vars
        if not vars:
            vars = [compute_blood_pressure_level(value) for value in value_list]
        else:
            new_vars = [compute_blood_pressure_level(value) for value in value_list]
            if len(vars) == len(new_vars):
                vars = [i if abs(i) > abs(j) else j for i, j in zip(vars, new_vars)]
        return vars

    def health_warning_solutions_early(self, param: Dict) -> str:
        """
        - 输入：客户的指标数据（C端客户通过手工录入、语音录入、医疗设备测量完的结果），用药情况（如果C端有用药情况）
        - 要求:
            1. 如果近5天都有数据，则推出预警解决方案的内容包括，指标最近的波动情况，在饮食、运动、日常护理方面的建议。见格式一
            2. 如果不满足连续条件（包括只有一条数据的情况）则预警方案只给出指标解读。见格式二

        - 输出：分析客户的指标数据，给出解决方案, 示例如下:

            格式一：从提供的数据来看，患者的血压在24小时内呈现下降趋势，收缩压下降了25%，舒张压下降了15%。建议其保持健康的生活习惯，如控制饮食，适量运动，同时定期测量血压和心率，及时了解自己的健康状况。如果血压持续下降，提醒患者及时就医。

            格式二：患者收缩压150、舒张压100，均高于正常范围，属于2级高血压。由于监测指标未达到报告分析要求，请您与患者进一步沟通。

        prd: https://alidocs.dingtalk.com/i/nodes/KGZLxjv9VGBk7RlwHeRpRpXrW6EDybno?utm_scene=team_space

        api: https://confluence.enncloud.cn/pages/viewpage.action?pageId=850011452#:~:text=%7D-,3.4.2%20%E9%A2%84%E8%AD%A6%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88,-%E5%8A%9F%E8%83%BD%E6%8F%8F%E8%BF%B0
        云效需求: https://devops.aliyun.com/projex/req/VOSE-3607# 《通过预警患者指标数据给出预警解决方案》

        ## 沟通记录
        1. 今天一定会有数据
        2. 三个指标数据同步存在
        3. 传近5天的数据 仅以此判断连续
        4. 收缩压和舒张压近五天连续 格式一
        """
        model = self.gsr.model_config["health_warning_solutions_early"]
        is_continuous = self.__health_warning_solutions_early_continuous_check__(
            param["indicatorData"]
        )  # 通过数据校验判断处理逻辑

        time_range = {
            i["date"][:10] for i in param["indicatorData"][0]["data"]
        }  # 当前的时间范围
        bpl = []
        ihm_health_sbp, ihm_health_dbp, ihm_health_hr = [], [], []
        if is_continuous:
            prompt_str = self.gsr.prompt_meta_data["event"][
                "warning_solutions_early_continuous"
            ]["description"]
            prompt_template = PromptTemplate.from_template(prompt_str)
            for i in param["indicatorData"]:
                if i["code"] == self.indicatorCodeMap["收缩压"]:  # 收缩压
                    ihm_health_sbp = [j["value"] for j in i["data"]]
                    bpl = self.__health_warning_update_blood_pressure_level__(
                        bpl, ihm_health_sbp
                    )
                elif i["code"] == self.indicatorCodeMap["舒张压"]:  # 舒张压
                    ihm_health_dbp = [j["value"] for j in i["data"]]
                    bpl = self.__health_warning_update_blood_pressure_level__(
                        bpl, ihm_health_dbp
                    )
                elif i["code"] == self.indicatorCodeMap["心率"]:  # 心率
                    ihm_health_hr = [j["value"] for j in i["data"]]
            ihm_health_blood_pressure_level = (
                self.__health_warning_update_blood_pressure_level__(
                    bpl, return_str=True
                )
            )
            prompt = prompt_template.format(
                time_start=min(time_range),
                time_end=max(time_range),
                ihm_health_sbp=ihm_health_sbp,
                ihm_health_dbp=ihm_health_dbp,
                ihm_health_blood_pressure_level=ihm_health_blood_pressure_level,
                ihm_health_hr=ihm_health_hr,
            )
        else:  # 非连续，只取当日指标
            prompt_str = self.gsr.prompt_meta_data["event"][
                "warning_solutions_early_not_continuous"
            ]["description"]
            prompt_template = PromptTemplate.from_template(prompt_str)
            for i in param["indicatorData"]:
                if i["code"] == self.indicatorCodeMap["收缩压"]:
                    ihm_health_sbp = [i["data"][-1]["value"]]
                elif i["code"] == self.indicatorCodeMap["舒张压"]:
                    ihm_health_dbp = [i["data"][-1]["value"]]
                elif i["code"] == self.indicatorCodeMap["心率"]:
                    ihm_health_hr = [i["data"][-1]["value"]]
            prompt = prompt_template.format(
                time_start=min(time_range),
                time_end=max(time_range),
                ihm_health_sbp=ihm_health_sbp,
                ihm_health_dbp=ihm_health_dbp,
                ihm_health_hr=ihm_health_hr,
            )
        history = [{"role": "user", "content": prompt}]
        response = callLLM(
            history=history, temperature=0.7, top_p=0.8, model=model, stream=True
        )
        content = accept_stream_response(response, verbose=False)
        return content

    def food_purchasing_list_manage(self, reply="好的-unknow reply", **kwds):
        """食材采购清单管理

        [
            "|名称|数量|单位|",
            "|---|---|---|",
            "|鸡蛋|500|g|",
            "|鸡胸肉|500g|g|",
            "|酱油|1|瓶|",
            "|牛腩|200|g|",
            "|菠菜|500|g|"
        ]

        """

        def parse_model_response(content):
            first_match_content = re.findall("```json(.*?)```", content, re.S)[
                0
            ].strip()
            ret = json.loads(first_match_content)
            reply, purchasing_list = ret["content"], ret["purchasing_list"]
            return reply, purchasing_list

        purchasing_list = kwds.get("purchasing_list")
        prompt = kwds.get("prompt")
        intentCode = "food_purchasing_list_management"
        event_msg = self.gsr.prompt_meta_data["event"][intentCode]
        sys_prompt = event_msg["description"] + event_msg["process"]
        model = self.gsr.model_config["food_purchasing_list_management"]
        sys_prompt = sys_prompt.replace(
            "{purchasing_list}", json.dumps(purchasing_list, ensure_ascii=False)
        )
        query = sys_prompt + f"\n用户说: {prompt}\n" + f"现采购清单:\n```json\n"
        history = [{"role": "user", "content": query}]
        logger.debug(f"食材采购清单管理 LLM Input: \n{history}")
        content = callLLM(history=history, temperature=0.7, model=model, top_p=0.8)
        logger.debug(f"食材采购清单管理 LLM Output: \n{content}")
        try:
            reply, purchasing_list = parse_model_response(content)
        except Exception as err:
            logger.exception(content)
            content = callLLM(history=history, temperature=0.7, model=model, top_p=0.8)
            try:
                reply, purchasing_list = parse_model_response(content)
            except Exception as err:
                logger.exception(err)
                logger.critical(content)
                purchasing_list = [
                    {"name": "鸡蛋", "quantity": 500, "unit": "g"},
                    {"name": "鸡胸肉", "quantity": 500, "unit": "g"},
                    {"name": "酱油", "quantity": 1, "unit": "瓶"},
                    {"name": "牛腩", "quantity": 200, "unit": "g"},
                    {"name": "菠菜", "quantity": 500, "unit": "g"},
                ]
        purchasing_list = self.__sort_purchasing_list_by_category__(purchasing_list)
        ret = {
            "purchasing_list": purchasing_list,
            "content": reply,
            "intentCode": intentCode,
            "dataSource": "语言模型",
            "intentDesc": self.gsr.intent_desc_map.get(
                intentCode, "食材采购清单管理-unknown intentCode desc error"
            ),
        }
        return ret

    def __sort_purchasing_list_by_category__(self, items: List[Dict]) -> List[Dict]:
        """根据分类对采购清单排序

        Args:
            items List[Dict]: 采购清单列表, 包含`name`, `classify`, `quantity`, `unit`四个字段

        Returns:
            List[Dict]: 排序后的采购清单列表
        """
        if items[0].get("classify") is None:  # 没有分类字段, 直接返回
            return items
        cat_map = {
            "水果": "001",
            "蔬菜": "002",
            "肉蛋类": "003",
            "水产类": "004",
            "米面粮油": "005",
            "营养保健": "006",
            "茶饮": "007",
            "奶类": "008",
        }
        items = [i for i in items if i.get("classify") and cat_map.get(i["classify"])]
        ret = list(sorted(items, key=lambda x: cat_map.get(x["classify"])))
        return ret

    def food_purchasing_list_generate_by_content(
        self, query: str, *args, **kwargs
    ) -> Dict:
        """根据用户输入内容生成食材采购清单"""
        if not query:
            raise ValueError("query can't be empty")
        model = self.gsr.model_config["food_purchasing_list_generate_by_content"]
        # system_prompt = self.gsr.prompt_meta_data['event']['food_purchasing_list_generate_by_content']['description']
        example_item = {
            "name": {"desc": "物品名称", "type": "str"},
            "classify": {"desc": "物品分类", "type": "str"},
            "quantity": {"desc": "数量", "tpye": "int"},
            "unit": {"desc": "单位", "type": "str"},
        }
        example_item_js = json.dumps(example_item, ensure_ascii=False)
        prompt_template_str = (
            "请你根据医生的建议帮我生成一份采购清单,要求如下:\n"
            "1. 每个推荐物品包含`name`, `classify`, `quantity`, `unit`四个字段\n"
            "2. 最终的格式应该是List[Dict],各字段描述及类型定义:\n[{{ example_item_js }}]\n"
            "3. classify字段可选范围包含：肉蛋类、水产类、米面粮油、蔬菜、水果、营养保健、茶饮、奶类，如医生建议中包含药品不能加入采购清单\n"
            "4. 蔬菜、肉蛋类、水产类、米面粮油单位为: g, 饮品、营养保健、奶类的单位可以是: 瓶、箱、包、盒、罐、桶, 水果的单位可以是: 个、斤、盒\n"
            "5. 只输出生成的采购清单，不包含任何其他内容\n"
            "6. 输出示例:\n```json\n```\n\n"
            "医生建议: \n{{ prompt }}"
        )
        prompt = prompt_template_str.replace(
            "{{ example_item_js }}", example_item_js
        ).replace("{{ prompt }}", query)
        history = [{"role": "user", "content": prompt}]
        logger.debug(
            f"根据用户输入生成采购清单 LLM Input: {json.dumps(history, ensure_ascii=False)}"
        )
        response = callLLM(
            history=history,
            temperature=0.3,
            model=model,
            top_p=0.8,
            stream=True,
        )
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"根据用户输入生成采购清单 LLM Output: \n{content}")
        purchasing_list_str = re.findall("```json(.*?)```", content, re.S)[0].strip()
        purchasing_list = json.loads(purchasing_list_str)

        purchasing_list = self.__sort_purchasing_list_by_category__(purchasing_list)
        return purchasing_list

    def rec_diet_reunion_meals_restaurant_selection(
        self, history=[], backend_history: List = [], **kwds
    ) -> Generator:
        """聚餐场景
        提供各餐厅信息
        群组中各角色聊天内容
        推荐满足角色提出条件的餐厅, 并给出推荐理由

        Args
            history List[Dict]: 群组对话信息

        Example
            ```json
            {
                "restaurant_message":"",
                "history": [
                    {"name":"龙傲天", "role":"爸爸", "content": "咱们每年年夜饭都在家吃，咱们今年下馆子吧！大家有什么意见？"},
                    {"name":"李蒹葭", "role":"妈妈", "content":"年夜饭我姐他们一家过来一起，咱们一共10个人，得找一个能坐10个人的包间，预算一两千吧"},
                    {"name":"龙霸天", "role":"爷爷", "content":"年夜饭得有鱼，找一家中餐厅，做鱼比较好吃的，孩子奶奶腿脚不太好，离家近点吧"},
                    {"name":"李秀莲", "role":"奶奶", "content":"我没什么意见，环境好一点有孩子活动空间就可以。"},
                    {"name":"龙翔", "role":"大儿子", "content":"我想吃海鲜！吃海鲜！"}
                ],
                "backend_history": []
            }
            ```
        """

        def make_system_prompt(kwds):
            message = (
                # kwds.get("restaurant_message") if kwds.get("restaurant_message") else DEFAULT_RESTAURANT_MESSAGE
                kwds.get("hospital_message")
                if kwds.get("hospital_message")
                else HOSPITAL_MESSAGE
            )
            event_msg = (
                self.gsr.prompt_meta_data["event"][
                    "reunion_meals_restaurant_selection"
                ]["description"]
                if not kwds.get("event_msg")
                else kwds.get("event_msg")
            )
            sys_prompt = event_msg.replace("{MESSAGE}", message)
            return sys_prompt

        def make_ret_item(message: str, end: bool, backend_history: List[Dict]) -> Dict:
            return {
                "message": message,
                "end": end,
                "backend_history": backend_history,
                "dataSource": "语言模型",
                "intentCode": "shared_decision",
                "intentDesc": "年夜饭共策",
                "type": "Result",
            }

        model = self.gsr.model_config["reunion_meals_restaurant_selection"]
        sys_prompt = make_system_prompt(kwds)
        # messages = [{"role":"system", "content": sys_prompt}] + backend_history
        messages = [{"role": "system", "content": sys_prompt}]
        try:
            query = ""
            for item in history:
                name, role, content = (
                    item.get("name"),
                    item.get("role"),
                    item.get("content"),
                )
                # {"name": "管家","role": "hb_qa@39238","content": "欢迎韩伟娜进入本群，您可以在这里畅所欲言了。"}  管家的信息无用, 过滤    2024年1月11日10:41:02
                if name == "管家":
                    continue
                user_input = ""
                user_input += name
                # TODO 当前role传的是hb_qa@39238这种信息，会导致生成内容中出现这样的信息, 取消把role信息传入提示中  2024年1月11日10:41:07
                # if role:
                #     user_input += f"({role})"

                # {"name": "郭燕","role": "hb_qa@39541","content": "@管家 今天晚上朋友聚餐，大概10人，想找一个热热闹闹，有表演的餐厅"}
                # 用户的输入信息中包含@管家, 进行过滤        2024年1月11日10:41:11
                content = content.replace("@管家 ", "")
                user_input += f": {content}\n"
                query += user_input
            messages.append({"role": "user", "content": query})
            logger.debug(f"共策 LLM Input: {json.dumps(messages, ensure_ascii=False)}")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.9,
                top_p=0.8,
                stream=True,
            )
            ret_content = ""
            for i in response:
                msg = i.choices[0].delta.to_dict()
                text_stream = msg.get("content")
                if text_stream:
                    ret_content += text_stream
                    # print(text_stream, end="", flush=True)
                    yield make_ret_item(text_stream, False, [])
            messages.append({"role": "assistant", "content": ret_content})

            logger.debug(f"共策 LLM Output: {ret_content}")
            yield make_ret_item("", True, messages[1:])
        except openai.APIError as e:
            logger.error(f"Model {model} generate error: {e}")
            yield make_ret_item("内容过长,超出模型处理氛围", True, [])
        except Exception as err:
            logger.error(f"Model {model} generate error: {err}")
            yield make_ret_item(repr(err), True, [])


class Agents:
    session = Session()
    ocr = RapidOCR()

    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr: InitAllResource = gsr
        setattr(gsr, "agents", self)
        self.regist_aigc_functions()
        self.__load_image_config__()
        self.client = openai.OpenAI()
        self.aclient = openai.AsyncOpenAI()

    def __load_image_config__(self):
        self.image_font_path = Path(__file__).parent.parent.parent.parent.joinpath(
            "data/font/simsun.ttc"
        )
        if not self.image_font_path.exists():
            logger.error(f"font file not found: {self.image_font_path}")
            exit(1)

    async def get_ocr(self, payload):
        import requests
        url = "http://10.228.67.99:26927/ocr"
        # payload = {'image_url': 'http://ai-health-manager-algorithm.oss-cn-beijing.aliyuncs.com/reportUpload/e7339bfc-3033-4200-a03f-9bc828004da3.jpg'}
        files = [
        ]
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        return response.json()

    async def __ocr_report__(self, **kwargs):
        """报告OCR功能"""
        payload = {'image_url': kwargs.get('url', '')}
        raw_result = await self.get_ocr(payload)
        docs = ""
        if raw_result:
            process_ocr_result = [line[1] for line in raw_result]
            logger.debug(
                f"Report interpretation OCR result: {dumpJS(process_ocr_result)}"
            )
            docs += "\n".join(process_ocr_result)
        else:
            logger.error(f"Report interpretation OCR result is empty")
        return docs, raw_result, process_ocr_result

    def __report_interpretation_result__(
        self,
        ocr_result: Union[str, List[str]] = "",
        msg: str = "Unknown Error",
        report_type: str = "Unknown Type",
        remote_image_url: str = "",
    ):
        """报告解读结果

        - Args:
            ocr_result (List[str]): OCR结果
            msg (str): 报告解读内容
            report_type (str): 报告类型

        - Returns:
            Dict: 报告解读结果
        """
        return {
            "ocr_result": ocr_result,
            "report_interpretation": msg,
            "report_type": report_type,
            "image_retc": remote_image_url,
        }

    def __plot_rectangle__(self, tmp_path, file_path, rectangles_with_text):
        """为识别的报告内容画出矩形框"""
        image_io = Image.open(file_path)
        draw = ImageDraw.Draw(image_io)
        for rectangle, text in rectangles_with_text:
            line_value = int(0.002 * sum(image_io.size))
            fontsize = line_value * 6
            image_font = ImageFont.truetype(str(self.image_font_path), fontsize)
            draw.rectangle(rectangle, outline="blue", width=line_value)
            draw.text(
                (rectangle[0] - fontsize * 2, rectangle[1] - fontsize - 15),
                text,
                font=image_font,
                fill="red",
            )
        save_path = tmp_path.joinpath(file_path.stem + "_rect" + file_path.suffix)
        image_io.save(save_path)
        logger.debug(f"Plot rectangle image saved to {save_path}")
        return save_path

    def __upload_image__(self, save_path):
        """上传图片到服务器"""
        url = self.gsr.api_config["ai_backend"] + "/file/uploadFile"
        payload = {"businessType": "reportAnalysis"}
        if save_path.suffix.lower() in [".jpg", ".jpeg"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/jpeg"))]
        elif save_path.suffix.lower() in [".png"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/png"))]
        else:
            files = [
                (
                    "file",
                    (
                        save_path.name,
                        open(save_path, "rb"),
                        f"image/{save_path.suffix.lower()[1:]}",
                    ),
                )
            ]
        resp = self.session.post(url, data=payload, files=files)
        if resp.status_code == 200:
            remote_image_url = resp.json()["data"]
        else:
            logger.error(f"Upload image error: {resp.text}")
            remote_image_url = ""
        return remote_image_url

    async def __report_ocr_classification_make_text_group__(
        self, file_path: Union[str, Path], raw_result, tmp_path, **kwargs
    ) -> str:
        """报告OCR结果分组"""

        # "3. 可选类别有[报告标题,基础信息,影像图片,影像所见,诊断意见,医疗建议,检查方法,检查医生]\n"
        sysprompt = (
            "You are a helpful assistant.\n"
            "# 任务描述\n"
            "1. 下面我将给你报告OCR提取的内容，它是有序的，优先从上到下从左到右\n"
            "2. 请你参考给出的内容的前后信息，按内容的前后顺序对报告的内容进行归类，类别最多5个\n"
            "3. 只给出各类别开始内容和结尾内容对应的index, 所有内容的index都应当被包含\n"
            "# 输出格式要求\n"
            "## 格式参考\n"
            "```json\n"
            '{"分类1": [start_idx_1, end_idx_1], "分类2": [start_idx_2, end_idx_2], "分类3": [start_idx_3, end_idx_3],...}\n'
            "```\n"
            "## 特殊要求\n"
            "其中start_idx_2=end_idx_1+1, start_idx_3=end_idx_2+1\n"
        )
        content_index = {
            idx: text for idx, text in enumerate([i[1] for i in raw_result])
        }
        messages = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": "```json\n" + str(content_index) + "\n```"},
        ]
        model = self.gsr.get_model("aigc_functions_report_interpretation_text_classify")

        logger.debug(f"报告解读文本分组 LLM Input:\n{dumpJS(messages)}")
        response = await self.aclient.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
        )
        content = response.choices[0].message.content
        logger.debug(f"报告解读文本分组 LLM Output: {content}")

        content = re.findall("```json(.*?)```", content, re.S)[0].strip()
        try:
            loc = json.loads(content)
        except:
            loc = {}
        try:
            rectangles_with_text = []
            for topic, index_range in loc.items():
                if index_range[0] > index_range[1]:
                    index_range[0], index_range[1] = index_range[1], index_range[0]
                if index_range[0] < 0:
                    index_range[0] = 0
                elif index_range[0] > len(content_index):
                    continue
                if index_range[1] >= len(content_index):
                    index_range[1] = len(content_index) - 1
                msgs = raw_result[index_range[0] : index_range[1] + 1]
                coordinates = [item[0] for item in msgs]
                left = min([j for i in coordinates for j in [i[0][0], i[3][0]]])
                top = min([j for i in coordinates for j in [i[0][1], i[1][1]]])
                right = max([j for i in coordinates for j in [i[1][0], i[2][0]]])
                bottom = max([j for i in coordinates for j in [i[2][1], i[3][1]]])
                rectangles_with_text.append(((left, top, right, bottom), topic))
            save_path = self.__plot_rectangle__(
                tmp_path, file_path, rectangles_with_text
            )
        except Exception as e:
            logger.exception(f"Report interpretation error: {e}")
            pass
        remote_image_url = self.__upload_image__(save_path)
        return remote_image_url

    def __compose_user_msg__(
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
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "ietary_guidelines":
            if ietary_guidelines:
                for key, value in ietary_guidelines.items():
                    if value and DIETARY_GUIDELINES_KEY_MAP.get(key):
                        content += f"{DIETARY_GUIDELINES_KEY_MAP[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "key_indicators":
            # 创建一个字典来存储按时间聚合的数据
            aggregated_data = {}

            # 遍历数据并聚合
            for item in key_indicators:
                for entry in item["data"]:
                    time = entry["time"].split(" ")[0]
                    value = entry["value"]
                    if time not in aggregated_data:
                        aggregated_data[time] = {}
                    aggregated_data[time][item["key"]] = value

            # 创建Markdown表格
            content = "| 测量时间 | 体重 | 体脂率 | BMI |\n"
            content += "| ------ | ---- | ------ | ----- |\n"

            # 填充表格
            for time, measurements in aggregated_data.items():
                row = f"| {time} | {measurements.get('体重', '')} | {measurements.get('体脂率', '')} | {measurements.get('bmi', '')} |\n"
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

    def regist_aigc_functions(self) -> None:
        self.funcmap = {
            v: getattr(self, v) for k, v in self.gsr.intent_aigcfunc_map.items()
        }
        for obj_str in dir(self):
            if (
                obj_str.startswith("aigc_functions_") or obj_str.startswith("sanji_")
            ) and not self.funcmap.get(obj_str):
                self.funcmap[obj_str] = getattr(self, obj_str)

    def aigc_functions_single_choice(self, prompt: str, options: List[str], **kwargs):
        """单项选择功能

        - Args:
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        model = self.gsr.model_config.get(
            "aigc_functions_single_choice", "Qwen-14B-Chat"
        )
        prompt_template_str = self.gsr.prompt_meta_data["event"][
            "aigc_functions_single_choice"
        ]["description"]
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        query = prompt_template.format(options=options, prompt=prompt)
        messages = [{"role": "user", "content": query}]
        logger.debug(
            f"Single choice LLM Input: {json.dumps(messages, ensure_ascii=False)}"
        )
        content = callLLM(
            history=messages,
            model=model,
            temperature=0,
            repetition_penalty=1.0,
        )
        logger.debug(f"Single choice LLM Output: {content}")
        if content == "选项与要求不符":
            return content
        else:
            if content not in options:
                logger.error(f"Single choice error: {content} not in options")
                return "选项与要求不符"
        return content

    async def aigc_functions_report_interpretation(
        self, options: List[str] = ["口腔报告", "胸部报告", "腹部报告"], **kwargs
    ) -> Dict:
        """报告解读功能

        - Args:
            file_path (str): 报告文件路径

        - Returns:
            str: 报告解读内容
        """

        def prepare_file(**kwargs):

            file_path = None
            image_url = kwargs.get("url")

            if not tmp_path.exists():
                tmp_path.mkdir(parents=True)

            if image_url:
                logger.info(f"start: image_url")
                r = self.session.get(image_url)
                logger.info(f"stop: image_url")
                file_path = tmp_path.joinpath(basename(image_url))
                logger.info(f"start2: image_url")
                with open(file_path, mode="wb") as f:
                    f.write(r.content)
                logger.info(f"stop2: image_url")
            elif kwargs.get("file_path"):
                file_path = kwargs.get("file_path")
                image_url = self.__upload_image__(file_path)
            else:
                logger.error(f"Report interpretation error: file_path or url not found")
            return image_url, file_path

        async def jude_report_type(docs: str, options: List[str]) -> str:
            query = f"{docs}\n\n请你判断以上报告属于哪个类型,从给出的选项中选择: {options}, 要求只输出选项答案, 请不要输出其他内容\n\nOutput:"
            messages = [{"role": "user", "content": query}]
            report_type = callLLM(
                history=messages, model="Qwen1.5-72B-Chat", temperature=0.7, top_p=0.5
            )
            logger.debug(f"Report interpretation report type: {report_type}")
            if report_type not in options:
                if "口腔" in docs and "口腔报告" in options:
                    report_type = "口腔报告"
                elif "胸部" in docs and "胸部报告" in options:
                    report_type = "胸部报告"
                elif "腹部" in docs and "腹部报告" in options:
                    report_type = "腹部报告"
            if report_type not in options:
                report_type = "其他"
            return report_type

        tmp_path = Path(f".tmp/images")
        image_url, file_path = prepare_file(**kwargs)
        if not file_path:
            return self.__report_interpretation_result__(msg="请输入信息源")
        docs, raw_result, process_ocr_result = await self.__ocr_report__(**kwargs)
        if not docs:
            return self.__report_interpretation_result__(
                msg="未识别出报告内容，请重新尝试",
                ocr_result="您的报告内容无法解析，请重新尝试.",
            )
        try:
            remote_image_url = await self.__report_ocr_classification_make_text_group__(
                file_path, raw_result, tmp_path
            )
            if not remote_image_url:
                remote_image_url = image_url
        except Exception as e:
            logger.exception(f"Report interpretation error: {e}")
            remote_image_url = image_url
        # 报告异常信息解读
        # prompt_template_str = "You are a helpful assistant.\n" "# 任务描述\n" "请你为我解读报告中的异常信息"
        # messages = [{"role": "system", "content": prompt_template_str}, {"role": "user", "content": docs}]
        # logger.debug(f"Report interpretation LLM Input: {dumpJS(messages)}")
        # response = callLLM(history=messages, model="Qwen-14B-Chat", temperature=0.7, top_p=0.5, stream=True)
        # content = accept_stream_response(response, verbose=False)
        # logger.debug(f"Report interpretation LLM Output: {content}")

        # 增加报告类型判断
        if options:
            report_type = await jude_report_type(docs, options)
        else:
            report_type = "其他"
        return self.__report_interpretation_result__(
            ocr_result=docs, report_type=report_type, remote_image_url=remote_image_url
        )

    async def aigc_functions_report_summary(self, **kwargs):
        """报告内容总结
        循环
        """
        chunk_size = kwargs.get("chunk_size", 1000)
        assert kwargs["report_content"] is not None, "report_content is None"
        system_prompt = (
            "You are a helpful assistant.\n"
            "# 任务描述\n"
            "你是一个经验丰富的医生,你要清楚了解患者的生命熵检查报告内容，根据报告内容给出总结话术，其中包含生命熵熵值，哪些处于失衡状态，哪些有异常\n"
            "1. 请你根据自身经验，结合生命熵报告内容，给出摘要总结\n"
            "2. 请你根据总结，给出建议，并说明原因\n"
        )
        summary_list = []
        if isinstance(kwargs["report_content"], list):
            report_content = "\n".join(kwargs["report_content"])
        else:
            report_content = kwargs["report_content"]
        for i in range(0, len(report_content), chunk_size):
            chunk_text = kwargs["report_content"][i : i + chunk_size]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text},
            ]
            content = await acallLLM(
                history=messages,
                model="Qwen-14B-Chat",
                temperature=0.7,
                top_p=0.8,
            )
            summary_list.append(content)
        summary = "\n".join(summary_list)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary},
        ]
        content: str = await acallLLM(
            history=messages,
            model="Qwen1.5-72B-Chat",
            temperature=0.7,
            top_p=0.8,
        )
        return {"report_summary": content}

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary(self, **kwargs) -> str:
        """问诊摘要"""

        _event = "问诊摘要"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary_to_group(self, **kwargs) -> str:
        """问诊摘要"""
        _event = "问诊摘要"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary_chief_disease(self, **kwargs) -> str:
        """问诊摘要"""
        _event = "问诊摘要-主诉/现病史"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages", "user_profile"])
    async def aigc_functions_diagnosis(self, **kwargs) -> str:
        """诊断"""
        _event = "诊断"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        model_args = await self.__update_model_args__(
            kwargs, temperature=0, top_p=0.8, repetition_penalty=1.0
        )

        prompt_vars = {"user_profile": user_profile, "messages": messages}
        # 诊断1阶段必须直接返回字符串用于判断下一步逻辑
        content: str = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args={
                "temperature": 0.7,
                "top_p": 0.8,
                "repetition_penalty": 1.0,
            },
            **kwargs,
        )

        if content == "无":
            kwargs["intentCode"] = "aigc_functions_diagnosis_result"
            content: str = await self.aaigc_functions_general(
                _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
            )
        else:
            if model_args.get("stream") is True:
                content: AsyncGenerator = construct_naive_response_generator(content)
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_drug_recommendation(self, **kwargs) -> List[Dict]:
        """用药建议"""
        _event = "用药建议"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0, top_p=1, repetition_penalty=1.0
        )
        response: Union[str, AsyncGenerator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        if isinstance(response, openai.AsyncStream):
            return response
        try:
            content = json5.loads(response)
        except Exception as e:
            try:
                content = re.findall("```json(.*?)```", response, re.DOTALL)[0]
                content = dumpJS(json5.loads(content))
            except Exception as e:
                logger.error(f"AIGC Functions {_event} json5.loads error: {e}")
                content = dumpJS([])
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_food_principle(self, **kwargs) -> str:
        """饮食原则"""
        _event = "饮食原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sport_principle(self, **kwargs) -> str:
        """运动原则"""
        _event = "运动原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_mental_principle(self, **kwargs) -> str:
        """情志原则"""

        _event = "情志原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_diagnosis_generation(self, **kwargs) -> str:
        """西医决策-诊断生成"""

        _event = "西医决策-诊断生成"
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
        content = parse_examination_plan(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_chief_complaint_generation(self, **kwargs) -> str:
        """西医决策-主诉生成"""

        _event = "西医决策-主诉生成"
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

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_present_illness(self, **kwargs) -> str:
        """西医决策-现病史生成"""

        _event = "西医决策-现病史生成"
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

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_past_medical_history(self, **kwargs) -> str:
        """西医决策-既往史生成"""

        _event = "西医决策-既往史生成"
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
        content = parse_examination_plan(content)
        return content

    async def aigc_functions_generate_allergic_history(self, **kwargs) -> str:
        """西医决策-过敏史生成"""

        _event = "西医决策-过敏史生成"
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
        content = parse_examination_plan(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_medication_plan(self, **kwargs) -> str:
        """西药医嘱生成"""

        _event = "西药医嘱生成"
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
        content = parse_examination_plan(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_generate_examination_plan(self, **kwargs) -> str:
        """检查检验医嘱生成"""

        _event = "检查检验医嘱生成"
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
        content = parse_examination_plan(content)
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sjkyn_guideline_generation(self, **kwargs) -> str:
        """三济康养方案总则"""

        _event = "三济康养方案总则"

        # 获取并验证必填字段
        user_profile = kwargs.get("user_profile")
        if not user_profile:
            raise ValueError("用户画像信息缺失")

        required_fields = ["height", "weight", "bmi", "current_diseases"]
        for field in required_fields:
            if field not in user_profile or user_profile[field] is None:
                raise ValueError(f"{USER_PROFILE_KEY_MAP[field]}为必填项，且不能为空")

        # 提取必要的用户信息字段并解析体重和身高
        try:
            weight = parse_measurement(user_profile["weight"], "weight")
            height = parse_measurement(user_profile["height"], "height")
        except ValueError as e:
            raise ValueError(f"体重或身高格式不正确: {str(e)}")

        age = user_profile["age"]
        gender = user_profile["gender"]

        # 计算基础代谢率
        bmr = calculate_bmr(weight, height, age, gender)

        # 组合用户画像信息字符串
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile
        )
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
        user_profile_str = ""

        # 处理用户画像信息
        if user_profile:
            if user_profile.get("weight"):
                weight = parse_measurement(user_profile["weight"], "weight")
            if user_profile.get("height"):
                height = parse_measurement(user_profile["height"], "height")
            if user_profile.get("age") and user_profile.get("gender"):
                age = user_profile["age"]
                gender = user_profile["gender"]
                if weight and height:
                    bmr = calculate_bmr(weight, height, age, gender)
                    user_profile_str += f"基础代谢:\n{bmr}\n"

            # 组合用户画像信息字符串
            user_profile_str += self.__compose_user_msg__(
                "user_profile", user_profile=user_profile
            )

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

    # @param_check(check_params=["messages"])
    async def aigc_functions_dietary_details_generation(self, **kwargs) -> str:
        """饮食调理细则生成"""

        _event = "饮食调理细则生成"

        # 获取并验证必填字段

        user_profile_data = kwargs.get("user_profile")
        if not user_profile_data:
            raise ValueError("用户画像信息缺失")

        # 验证用户画像中的必填字段
        required_fields = [
            "age",
            "gender",
            "height",
            "weight",
            "bmi",
            "daily_physical_labor_intensity",
        ]
        for field in required_fields:
            if field not in user_profile_data or user_profile_data[field] is None:
                raise ValueError(f"{field}为必填项，且不能为空")

        if not (
            user_profile_data.get("current_diseases")
            or user_profile_data.get("management_goals")
        ):
            raise ValueError("现患疾病或管理目标必须至少填写一个")

        user_profile = UserProfile(**user_profile_data)

        # 解析体重和身高
        weight = parse_measurement(user_profile.weight, "weight")
        height = parse_measurement(user_profile.height, "height")

        # 计算基础代谢率 (BMR)
        bmr = calculate_bmr(weight, height, user_profile.age, user_profile.gender)

        # 组合用户画像信息字符串，并添加 BMR 信息
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile.dict()
        )
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
        content = parse_examination_plan(content)

        return content

    # @param_check(check_params=["messages"])
    @async_clock
    async def aigc_functions_meal_plan_generation(self, **kwargs) -> str:
        """生成餐次、食物名称"""

        _event = "生成餐次、食物名称"

        # 获取并验证必填字段
        user_profile_data = kwargs.get("user_profile")
        if not user_profile_data:
            raise ValueError("用户画像信息缺失")

        # 验证用户画像中的必填字段
        required_fields = [
            "age",
            "gender",
            "height",
            "weight",
            "bmi",
            "daily_physical_labor_intensity",
        ]
        for field in required_fields:
            if field not in user_profile_data or user_profile_data[field] is None:
                raise ValueError(f"{field}为必填项，且不能为空")

        if not (
            user_profile_data.get("current_diseases")
            or user_profile_data.get("management_goals")
        ):
            raise ValueError("现患疾病或管理目标必须至少填写一个")

        user_profile = UserProfile(**user_profile_data)

        # 解析体重和身高
        weight = parse_measurement(user_profile.weight, "weight")
        height = parse_measurement(user_profile.height, "height")

        # 计算基础代谢率 (BMR)
        bmr = calculate_bmr(weight, height, user_profile.age, user_profile.gender)

        # 组合用户画像信息字符串，并添加 BMR 信息
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile.dict()
        )
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
        content = parse_examination_plan(content)
        return content

    @async_clock
    async def aigc_functions_generate_food_quality_guidance(self, **kwargs) -> str:
        """生成餐次、食物名称的质量指导"""

        _event = "生成餐次、食物名称的质量指导"

        # 获取并验证必填字段
        user_profile_data = kwargs.get("user_profile")
        if not user_profile_data:
            raise ValueError("用户画像信息缺失")

        # 验证用户画像中的必填字段
        required_fields = [
            "age",
            "gender",
            "height",
            "weight",
            "bmi",
            "daily_physical_labor_intensity",
        ]
        for field in required_fields:
            if field not in user_profile_data or user_profile_data[field] is None:
                raise ValueError(f"{field}为必填项，且不能为空")

        if not (
            user_profile_data.get("current_diseases")
            or user_profile_data.get("management_goals")
        ):
            raise ValueError("现患疾病或管理目标必须至少填写一个")

        ietary_guidelines = kwargs.get("ietary_guidelines")
        if not ietary_guidelines or not ietary_guidelines.get(
            "basic_nutritional_needs"
        ):
            raise ValueError("饮食调理细则中的基础营养需求为必填项，且不能为空")

        basic_nutritional_needs = ietary_guidelines.get("basic_nutritional_needs")

        meal_plan = convert_meal_plan_to_text(kwargs.get("meal_plan"))

        user_profile = UserProfile(**user_profile_data)

        # 解析体重和身高
        weight = parse_measurement(user_profile.weight, "weight")
        height = parse_measurement(user_profile.height, "height")

        # 计算基础代谢率 (BMR)
        bmr = calculate_bmr(weight, height, user_profile.age, user_profile.gender)

        # 组合用户画像信息字符串，并添加 BMR 信息
        user_profile_str = self.__compose_user_msg__(
            "user_profile", user_profile=user_profile.dict()
        )
        user_profile_str += f"基础代谢:\n{bmr}\n"

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
        content = parse_examination_plan(content)
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

        # 参数检查
        ParamTools.check_aigc_functions_sanji_plan_exercise_regimen(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", [])
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

        # 参数检查
        ParamTools.check_aigc_functions_sanji_plan_exercise_plan(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", [])
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
        self, kwargs
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
        ParamTools.check_aigc_functions_body_fat_weight_management_consultation(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        if kwargs["messages"] and len(kwargs["messages"]) >= 6:
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

    def calculate_bmr(weight: float, height: str, age: int, gender: str) -> float:
        """计算基础代谢率 (BMR)"""
        height_cm = (
            float(height.replace("cm", "")) if "cm" in height else float(height) * 100
        )
        if gender == "男":
            return 10 * weight + 6.25 * height_cm - 5 * age + 5
        elif gender == "女":
            return 10 * weight + 6.25 * height_cm - 5 * age - 161
        else:
            raise ValueError("性别必须为 '男' 或 '女'")

    # @param_check(check_params=["messages"])
    async def aigc_functions_chinese_therapy(self, **kwargs) -> str:
        """中医调理"""
        _event = "中医调理"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(kwargs)
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_food_principle_new(self, **kwargs) -> str:
        """饮食原则"""
        _event = "饮食原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     key, values = line.split('：', 1)
        #     if values=='无':
        #         data[key]=[]
        #     else:
        #         data[key] = values
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sport_principle_new(self, **kwargs) -> str:
        """运动原则"""
        _event = "运动原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     if '：' in line:
        #         key, values = line.split('：', 1)
        #         if values=='无':
        #             data[key]=[]
        #         else:
        #             data[key] = values
        #     else:
        #         data['运动课程']=line
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_mental_principle_new(self, **kwargs) -> str:
        """情志原则"""

        _event = "情志原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # lines = content.split('\n')
        # data={}
        # l=['one','two','three','four']
        # for i in range(len(lines)):
        #     data[l[i]]=lines[i]

        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_chinese_therapy_new(self, **kwargs) -> str:
        """中医调理"""
        _event = "中医调理"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(kwargs)
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     if len(line)>0:
        #         key, values = line.split('：', 1)
        #         if values=='无':
        #             data[key]=[]
        #         else:
        #             data[key] = values
        return content

    async def aigc_functions_auxiliary_history_talking(self, **kwargs: object):
        """医生端 - 生成问题"""
        _event = "生成医生问题"

        def __fmt_history(_messages):
            _role_map = {"user": "用户", "assistant": "医生"}
            _tmp_lst = []
            for item in _messages:
                role = _role_map.get(item["role"], "用户")
                content = item["content"]
                _tmp_lst.append(f"{role}: {content}")
            return "\n".join(_tmp_lst)

        duplicate_content = {}
        _messages = []
        for item in kwargs["messages"]:
            _content = item["content"]
            if not duplicate_content.get(_content):
                _messages.append(item)
                duplicate_content[_content] = 1
        history_str = __fmt_history(_messages)
        prompt = (
            f"# 用户与医生的对话记录\n{history_str}\n"
            "# 角色定位\n"
            "请你扮演一个经验丰富的医生,协助我为患者的疾病进行问诊\n"
            "# 任务描述\n"
            "1. 在多轮的对话中会提供患者的个人信息和感受,请你根据自身经验分析,针对个人情况提出相应的 问题\n"
            "2. 问题关键点可以包括:持续时间、发生时机、诱因或症状发生部位等\n"
            "3. 不要重复询问同一个问题,问题尽可能简洁,每次最多提出两个问题\n"
            "4. 纯净模式，只输出要询问患者的问题，不同的问题用｜｜隔开\n"
            "5. 输出示例如下\n"
            "5. 您的发热情况如何，如否有测量体温？｜｜除了嗓子痛和痒，是否有咳嗽或者喉咙有异物感？\n"
        )
        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )

        content: Union[str, Generator] = await acallLLM(
            model="Qwen1.5-72B-Chat",
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: \n{content}")
        # 使用正则表达式找到所有的句子边界（句号或问号）
        try:
            sentences = content.split("｜｜", 1)
        except Exception as err:
            logger.error(err)
            sentences = [content]
        return sentences

    async def aigc_functions_auxiliary_diagnosis(self, **kwargs):
        prompt_template = "# 患者与医生历史会话信息\n{history_str}\n\n"
        user_input = (
            "# 任务描述\n"
            "请你扮演一个经验丰富的医生,协助我进行疾病的诊断,"
            "根据患者与医生的历史会话信息,输出若干个患者最多5个可能的诊断以及对应的概率值\n"
            "格式参考: 疾病-概率,疾病-概率, 以`,`分隔\n"
            "只输出`疾病`-`概率`,避免输入任何其他内容"
        )

        # 2024年4月30日14:44:08 过滤重复的输入
        duplicate_messages, _messages = {}, []
        for item in kwargs["messages"]:
            if not duplicate_messages.get(item["content"]):
                _messages.append(item)
                duplicate_messages[item["content"]] = 1
        history_str = "\n".join([f"{item['content']}" for item in _messages])
        prompt = prompt_template.format(history_str=history_str)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )

        try:
            d_p_pair_str = await acallLLM(
                model="Qwen1.5-72B-Chat",
                query=messages,
                **model_args,
            )
            d_p_pair = [i.strip().split("-") for i in d_p_pair_str.split(",")]
            d_p_pair = [
                {"name": i[0], "prob": i[1].replace("%", "") + "%"} for i in d_p_pair
            ]
        except Exception as err:
            logger.error(repr(err))
            d_p_pair = []
        return d_p_pair

    async def aigc_functions_relevant_inspection(self, **kwargs):
        prompt_template = (
            "# 患者与医生历史会话信息\n{history_str}\n\n"
            "# 任务描述\n"
            "你是一个经验丰富的医生,请你协助我进行疾病的鉴别诊断,输出建议我做的临床辅助检查项目\n"
            "1. 请你根据历史会话信息、初步诊断的结果、鉴别诊断的知识、分析我的疾病，进一步输出能够让我确诊的临床检查项目\n"
            "2. 只输出检查项目的名称，不要其他的内容\n"
            "3. 不同检查项目名称之间用`,`隔开,检查项目不要重复\n\n"
            "# 初步诊断结果\n{diagnosis_str}\n\n"
            "Begins!"
        )
        duplicate_content = {}
        _messages = []
        for item in kwargs["messages"]:
            _content = item["content"]
            if not duplicate_content.get(_content):
                _messages.append(item)
                duplicate_content[_content] = 1
        rolemap: Dict[str, str] = {"user": "患者", "assistant": "医生"}
        history_str = "\n".join(
            [f"{rolemap[item['role']]}: {item['content']}" for item in _messages]
        )
        diagnosis_str = ",".join([i["name"] for i in kwargs["diagnosis"]])
        prompt = prompt_template.format(
            history_str=history_str, diagnosis_str=diagnosis_str
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_str},
        ]
        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        content = await acallLLM(
            model="Qwen1.5-72B-Chat",
            query=messages,
            **model_args,
        )
        return [i.strip() for i in content.split(",")]

    @param_check(check_params=["messages"])
    async def aigc_functions_reason_for_care_plan(self, **kwargs) -> str:
        """康养方案推荐原因"""
        _event = "康养方案推荐原因"
        user_profile = self.__compose_user_msg__(
            "user_profile",
            user_profile=kwargs["user_profile"],
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        drug_plan = self.__compose_user_msg__(
            "drug_plan", drug_plan=kwargs.get("drug_plan", "")
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "drug_plan": drug_plan,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "food_principle": kwargs.get("food_principle", ""),
            "sport_principle": kwargs.get("sport_principle", ""),
            "mental_principle": kwargs["mental_principle"],
            "chinese_therapy": kwargs["chinese_therapy"],
        }
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1)
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return response

    async def sanji_assess_3d_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_3d_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        content = content.replace("：\n", "：")
        content = content.replace("：\n\n", "\n")
        data = {}
        lines = content.split("\n")
        for line in lines:
            if "：" in line:
                key, values = line.split("：", 1)
                if values == "无":
                    data[key] = []
                else:
                    data[key] = values.split("|")
        return data

    async def sanji_assess_keyword_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_keyword_cl"
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        # content = content.replace("：\n", "：")
        # lines = content.split("\n")
        # data = {}
        # for line in lines:
        #     if "：" in line:
        #         key, values = line.split("：", 1)
        #         if values == "无":
        #             data[key] = []
        #         else:
        #             data[key] = values.split(", ")
        content = content.split("||")
        data={}
        data['one']=content
        return data

    async def sanji_assess_3health_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_3health_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        content = content.replace("：\n", "：")
        lines = content.split("\n")
        data = {}
        for line in lines:
            if "：" in line:
                key, values = line.split("：", 1)
                if values == "无;" or values == ";":
                    data[key] = []
                else:
                    data[key] = [values]
        return data


    async def sanji_assess_literature_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_liter_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        if "\n" in content:
            lines = content.split("\n")
        else:
            lines = content.split("||")
        result = {"one": lines}

        return result

    async def sanji_intervene_goal_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_intervene_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )

        data = {}
        data["goal"] = {}
        data["literature"] = {}
        
        content = content.replace("：\n", "：")
        # if '-' in content:
        #     lines = content.split("\n\n")
        # else:
        lines = content.split("\n")
        for line in lines:
            if ":" in line or "：" in line:
                key, values = line.split("：", 1)
                if values == "无":
                    data["goal"][key] = []
                else:
                    my_list = values.split("||")
                    filtered_list = [item for item in my_list if item]
                    data["goal"][key] = filtered_list

        return data

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

        # 参数检查
        ParamTools.check_aigc_functions_sanji_plan_exercise_regimen(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", [])
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

        # 参数检查
        ParamTools.check_aigc_functions_sanji_plan_exercise_plan(kwargs)

        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        medical_records = self.__compose_user_msg__(
            "medical_records", medical_records=kwargs.get("medical_records", [])
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

    @param_check(check_params=["plan_ai", "plan_human"])
    async def aigc_functions_plan_difference_finder(
        self, **kwargs
    ) -> Union[str, Generator]:
        """差异点发现"""
        _event = "差异点发现"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_plan_difference_analysis(
        self, **kwargs
    ) -> Union[str, Generator]:
        """差异能力分析"""
        _event = "差异能力分析"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_plan_adjustment_suggestion(
        self, **kwargs
    ) -> Union[str, Generator]:
        """方案调整建议生成"""
        _event = "方案调整建议生成"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_doctor_recommend(self, **kwargs) -> Union[str, Generator]:
        async def download_and_load_doctor_messages() -> List[Dict]:
            doctor_example_path = Path(".cache/doctor_examples.json")
            if not doctor_example_path.exists():
                download_from_oss(
                    "ai-algorithm-nlp/intelligent-health-manager/data/docter_examples.json",
                    doctor_example_path,
                )
            doctor_examples = json.load(open(".cache/doctor_examples.json", "r"))
            return doctor_examples

        if kwargs.get("model_args") and kwargs["model_args"].get("stream") is True:
            raise ValidationException("医生推荐 model_args.stream can't be True")

        user_demands = self.__compose_user_msg__(
            "messages",
            messages=kwargs["messages"],
            role_map={"assistant": "助手", "user": "用户"},
        )

        # 先判断用户是否需要医生推荐
        prompt_template_assert = f"请你帮我判断用户是否需要推荐医生,需要:`Yes`, 不需要:`No`\n用户:{user_demands}"
        if_need = self.aigc_functions_single_choice(
            prompt_template_assert, options=["Yes", "No"]
        )
        if if_need.lower() != "yes":
            return []

        _event = "医生推荐"

        prompt_template = (
            "# 已知信息\n"
            "1.问诊结果：{diagnosis_result}\n"
            "2.我的诉求：{user_demands}\n"
            "# 医生信息\n"
            "{doctor_message}\n\n"
            "# 任务描述\n"
            "你是一位经验丰富的智能健康助手，请你根据输出要求、我的诉求、已知信息，为我推荐符合我病情的医生。\n"
            "# 输出要求\n"
            "1.根据已知信息、我对医生的诉求、医生信息列表，帮我推荐最符合我情况的备选5个医生信息\n"
            "2.你推荐我的医生，第一需求应该符合我的疾病诊断或者检查检验报告结论\n"
            "3.其他需求你要考虑我对医生擅长领域的需求，我对医生性别的需求等\n"
            "4.推荐医生的顺序按照符合我条件的优先级前后展示，输出格式参考：机构名称1-医生名称1，机构名称2-医生名称2，以`,`隔开\n"
            "Begins~"
        )

        # TODO 从外部加载医生数据
        if not hasattr(self, "docter_message"):
            doctor_examples = await download_and_load_doctor_messages()
            self.docter_message = "\n\n".join(
                [DoctorInfo(**i).__str__() for i in doctor_examples]
            )

        prompt_vars = {
            "doctor_message": self.docter_message,
            "diagnosis_result": kwargs.get("prompt", ""),
            "user_demands": user_demands,
        }
        model_args = await self.__update_model_args__(kwargs, temperature=1, top_p=0.8)
        # kwargs["messages"] = [
        #     {
        #         "role": "user",
        #         "content": "请你综合考虑医生的专业匹配度、医生职称、医生工作年限等信息来帮我推荐医生,,
        #     }
        # ]
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            prompt_template=prompt_template,
            model_args=model_args,
            **kwargs,
        )
        try:
            # raise AssertionError("未定义err")
            result = [i.strip() for i in response.split(",")]
        except Exception as err:
            logger.error(repr(err))
            result = err
        return result

    def aigc_functions_general(
        self,
        _event: str = "",
        prompt_vars: dict = {},
        model_args: dict = {
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1.0,
        },
        **kwargs,
    ) -> str:
        """通用生成"""
        event = kwargs.get("intentCode")
        model = self.gsr.get_model(event)
        prompt_template: str = self.gsr.get_event_item(event)["description"]
        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")
        content: str = callLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def aaigc_functions_general(
        self,
        _event: str = "",
        prompt_vars: dict = {},
        model_args: Dict = {},
        prompt_template: str = "",
        **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""
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

    async def sanji_general(
        self,
        process: int = 1,
        _event: str = "",
        prompt_vars: dict = {},
        model_args: Dict = {},
        prompt_template: str = "",
        **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""
        event = kwargs.get("intentCode")
        model = "Qwen1.5-32B-Chat"
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 0.3,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        des = self.gsr.get_event_item(event)["description"]
        if process == 2:
            des = (
                self.gsr.get_event_item(event)["process"]
                + self.gsr.get_event_item(event)["constraint"]
            )
        if process == 0:
            des += self.gsr.get_event_item(event)["constraint"]
        prompt_template: str = prompt_template if prompt_template else des
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

    async def __preprocess_function_args__(self, kwargs) -> dict:
        """处理aigc functions入参"""
        if not kwargs.get("model_args"):
            kwargs["model_args"] = {}
        if not kwargs.get("user_profile"):
            kwargs["user_profile"] = {}
        if not kwargs.get("diagnosis"):
            kwargs["diagnosis"] = ""
        return kwargs

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
        # kwargs = await self.__preprocess_function_args__(kwargs)
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


if __name__ == "__main__":
    gsr = InitAllResource()
    expert_model = expertModel(gsr)
    agents = Agents(gsr)
    param = testParam.param_dev_report_interpretation
    agents.call_function(**param)
