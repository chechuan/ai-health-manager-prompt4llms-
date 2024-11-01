# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-05 15:14:07
@desc    :   专家模型 & 独立功能
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

# 标准库导入
import random
import re
import sys
from pathlib import Path
import datetime
from datetime import datetime, timedelta
from string import Template
from typing import Dict, Generator, List

# 第三方库导入
from langchain.prompts.prompt import PromptTemplate

# 本地模块导入
sys.path.append(Path(__file__).parents[4].as_posix())
from data.jiahe_util import *
from data.test_param.test import testParam
from src.prompt.model_init import acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.module import (
    accept_stream_response, clock, compute_blood_pressure_level, dumpJS
)
from src.utils.resources import InitAllResource


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
        bmi = round(weight / height ** 2, 1)
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
            t = Template(blood_1)
            pro = kwargs.get("promptParam", {})
            prompt = t.substitute(
                bp_msg=bp_msg,
                age=pro.get("askAge", ''),
                sex=pro.get("askSix", ''),
                height=pro.get("askHeight", ''),
                weight=pro.get("askWeight", ''),
                disease=pro.get("askDisease", ''),
                family_med_history=pro.get("askFamilyHistory", ''),
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
                t = Template(blood_2)
                pro = kwargs.get("promptParam", {})
                prompt = t.substitute(
                    bp_msg=bp_msg,
                    history=hist_s,
                    age=pro.get("askAge", ''),
                    sex=pro.get("askSix", ''),
                    height=pro.get("askHeight", ''),
                    weight=pro.get("askWeight", ''),
                    disease=pro.get("askDisease", ''),
                    family_med_history=pro.get("askFamilyHistory", ''),
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
                    content = content[content.find("？") + 1:]
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

        temp = kwargs.get("promptParam", {}).get("temp", '')
        blood_1 = blood_pressure_risk_advise_prompt
        blood_2 = blood_pressure_scheme_prompt
        if temp == '1':
            blood_1 = blood_pressure_risk_advise_prompt_tem
            blood_2 = blood_pressure_scheme_prompt_tem

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

        model = "Qwen1.5-32B-Chat"
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
        compose_message1 = f"当前血糖{gl},{glucose_type(recent_time, float(gl))}。"
        time_deal = []
        period_content = {}
        for time_period in time_periods:
            count = 0
            t_g = 0
            f_g = 0
            message_f = ""
            for date in data.keys():
                t_e = slot_dict[time_period]
                glucose_val = data[date].get(t_e, '')
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
            if 0 < count < 3:
                message_ = (
                        f"血糖{count}天的记录中，{t_g}天血糖正常，{f_g}天血糖异常。"
                        + message_f
                )
                period_content[time_period] = message_
            if count > 3:
                time_deal.append(time_period)
        glucose_3 = ""
        for i in period_content.keys():
            glucose_3 += i + ":" + period_content[i]

        result_2 = result
        compose_message2 = glucose_3
        if len(time_deal) > 0:
            for time in time_deal:
                result_2 += "|" + time + "|"
                for date in data.keys():
                    t_e = slot_dict[time]
                    result_2 += data[date].get(t_e, '') + "|"
                result_2 += "\n"

            prompt_template = (
                "# 已知信息\n"
                "## 需要分析的血糖状况\n"
                "{glucose_message}\n"
                "# 任务描述\n"
                "你是一个血糖分析助手，请分别按顺序输出近7天不同的血糖测量阶段的最高血糖值、最低血糖值、波动趋势，只分析需要分析的血糖状况里面的时段，输出标明时段，字数少于50\\n"
            )
            prompt_vars = {"glucose_message": result_2}
            sys_prompt = prompt_template.format(**prompt_vars)

            history = []
            history.append({"role": "system", "content": sys_prompt})
            logger.debug(f"血糖趋势分析 LLM Input: {dumpJS(history)}")
            response = callLLM(
                history=history, temperature=0.8, top_p=1, model=model, stream=True
            )
            content = accept_stream_response(response, verbose=False)
            compose_message2 = glucose_3 + content

        if gl_code == "gl_2_pc":
            for time in time_periods:
                result += "|" + time + "|"
                for date in data.keys():
                    t_e = slot_dict[time]
                    result += data[date].get(t_e, '') + "|"
                result += "\n"
            prompt_template_pc = (
                "# 任务描述\n"
                "# 你是一位经验丰富的医师助手，帮助医生对用户进行血糖管理，请你根据用户已知信息，一周的血糖情况，给医生提供专业的处理建议，只提供原则性的建议，包含5个方面：1.检查建议；2.饮食调整；3.运动调整；4.药物治疗；5监测血糖。建议的字数控制在150字以内\n\n"
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
            logger.debug(f"血糖趋势分析 Output: {compose_message2}")
            all_message = compose_message1 + "\n" + compose_message2 + "\n" + pc_message
            return all_message

        # 组装步骤3
        for time in time_periods:
            result += "|" + time + "|"
            for date in data.keys():
                t_e = slot_dict[time]
                result += data[date].get(t_e, '') + "|"
            result += "\n"
        prompt_template_suggest = (
            "# 任务描述\n"
            "# 你是一位经验丰富的医师，请你根据已知信息，针对用户一周的血糖情况，给出合理建议，只提供原则性的建议，包含3个方面：测量建议；饮食建议；运动建议。建议的字数控制在250字以内\n\n"
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

        logger.debug(f"血糖趋势分析 Output: {compose_message2}")
        compose_message3 = compose_message3.replace("**", "")
        all_message = compose_message1 + "\n" + compose_message2 + "\n" + compose_message3
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


if __name__ == "__main__":
    gsr = InitAllResource()
    expert_model = expertModel(gsr)
    param = testParam.param_dev_report_interpretation
    # agents.call_function(**param)