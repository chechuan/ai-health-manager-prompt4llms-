# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-05 15:14:07
@desc    :   专家模型 & 独立功能
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import json
import re
import sys
import time
from email.mime import image
from os.path import basename
from pathlib import Path

import openai
from requests import Session

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from datetime import datetime, timedelta
from typing import Dict, List, Union

from langchain.prompts.prompt import PromptTemplate
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

from data.constrant import *
from data.constrant import DEFAULT_RESTAURANT_MESSAGE, HOSPITAL_MESSAGE
from data.test_param.test import testParam
from src.prompt.model_init import callLLM
from src.utils.Logger import logger
from src.utils.module import (InitAllResource, accept_stream_response, clock,
                              compute_blood_pressure_level, dumpJS, get_intent)


class expertModel:
    indicatorCodeMap = {"收缩压": "lk1589863365641", "舒张压": "lk1589863365791", "心率": "XYZBXY001005"}
    session = Session()
    ocr = RapidOCR()

    def __init__(self, gsr) -> None:
        self.gsr = gsr
        self.gsr.expert_model = self
        self.regist_aigc_functions()
        self.load_image_config()

    def load_image_config(self):
        self.image_font_path = Path(__file__).parent.parent.parent.parent.joinpath("data/font/simsun.ttc")
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
        assert weight > 0 and weight < 300, f"value `weigth`={weight} is not valid ∈ (0, 300)"
        assert height > 0 and height < 2.5, f"value `height`={height} is not valid ∈ (0, 2.5)"
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
        cur_date = kwargs["promptParam"].get("cur_date", "")
        level = kwargs["promptParam"].get("level", "")
        prompt = emotions_prompt.format(cur_date, level)
        messages = [{"role": "user", "content": prompt}]
        logger.debug("压力模型输入:" + json.dumps(messages, ensure_ascii=False))
        generate_text = callLLM(
            history=messages, max_tokens=1024, top_p=0.8, temperature=0.0, do_sample=False, model="Qwen-72B-Chat"
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
            "content": "健康报告显示您的健康处于轻度失衡状态。"
            + content
            + "已为您智能匹配了最适合您的减压方案，帮助您改善睡眠、缓解压力。",
            "scene_ending": True,
        }

    @staticmethod
    def weight_trend(cur_date, weight):
        prompt = weight_trend_prompt.format(cur_date, weight)
        messages = [{"role": "user", "content": prompt}]
        generate_text = callLLM(
            history=messages, max_tokens=1024, top_p=0.8, temperature=0.0, do_sample=False, model="Qwen-72B-Chat"
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

        cur_date = kwargs["promptParam"].get("cur_date", "")
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
            prompt = fat_reduction_prompt.format(cur_date, weight, "减脂效果不好，怎么改善？")
            logger.debug("进入体重出方案流程。。。")
        messages = [{"role": "user", "content": prompt}]
        logger.debug("体重方案/修改模型输入： " + json.dumps(messages, ensure_ascii=False))
        generate_text = callLLM(
            history=messages, max_tokens=1024, top_p=0.8, temperature=0.0, do_sample=False, model="Qwen-72B-Chat"
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
                num = round(float(weight.replace('kg', '')) - 75.4, 1)
                if num < 0:
                    cnt = f"体重较上周减少{num}kg。"
                else:
                    cnt = f"体重较上周增加{num}kg。"

            except Exception as err:
                return {
                    "thought": thought,
                    "contents": [f"您今日体重为{weight}。", "健康报告显示您的健康处于平衡状态。" + content + "这里是您下周的方案，请查收。"],
                    "scene_ending": False,
                    "scheme_gen": 1,
                    "modi_scheme": "scheme_no_change",
                    "weight_trend_gen": True,
                }
            finally:
                return {
                    "thought": thought,
                    "contents": [f"您今日体重为{weight}。", cnt, "健康报告显示您的健康处于平衡状态。" + content + "这里是您下周的方案，请查收。"],
                    "scene_ending": False,
                    "scheme_gen": 2,
                    "modi_scheme": "scheme_no_change",
                    "weight_trend_gen": True,
                }
        else:
            modi_type = get_scheme_modi_type(content)
            return {"thought": thought, 
                    "contents": ['好的，已重新帮您生成了健康方案，请查收。'], 
                    "scene_ending": False, 
                    "scheme_gen": 0, 
                    "modi_scheme":modi_type,
                    "weight_trend_gen":False,
                }

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

        ihm_health_sbp = kwargs["promptParam"]["ihm_health_sbp"]
        ihm_health_dbp = kwargs["promptParam"]["ihm_health_dbp"]
        query = kwargs["promptParam"].get("query", "")

        def inquire_gen(hitory, ihm_health_sbp, ihm_health_dbp):
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
            history = [{"role": role_map.get(str(i["role"]), "user"), "content": i["content"]} for i in hitory]
            # his_prompt = "\n".join([("Doctor" if not i['role'] == "User" else "User") + f": {i['content']}" for i in history])
            # prompt = blood_pressure_inquiry_prompt.format(blood_pressure_inquiry_prompt) + f'Doctor: '
            hist_s = '\n'.join([f"{i['role']}: {i['content']}" for i in history])
            if len(history) >= 7:
                messages = [{"role": "user", "content": blood_pressure_scheme_prompt.format(str(ihm_health_sbp), str(ihm_health_dbp), hist_s)}]
            else:
                messages = [{"role": "user", "content": blood_pressure_inquiry_prompt.format(str(ihm_health_sbp), str(ihm_health_dbp), hist_s)}] #+ history
            logger.debug("血压问诊模型输入： " + json.dumps(messages, ensure_ascii=False))
            generate_text = callLLM(
                history=messages, max_tokens=1024, top_p=0.9, temperature=0.8, do_sample=True, model="Qwen-72B-Chat"
            )
            logger.debug("血压问诊模型输出： " + generate_text)
            return generate_text

        def blood_pressure_inquiry(history, query):
            generate_text = inquire_gen(history, ihm_health_sbp, ihm_health_dbp)
            #while generate_text.count("\nAssistant") != 1 or generate_text.count("Thought") != 1:
                #thought = generate_text
                # generate_text = inquire_gen(bk_history, ihm_health_sbp, ihm_health_dbp)
            #thoughtIdx = generate_text.find("Thought") + 8
            # thoughtIdx = 0
            # thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            # outIdx = generate_text.find("\nassistant") + 11
            # content = generate_text[outIdx:].split("\n")[0].strip()
            if generate_text.find("Thought") == -1:
                lis = ['结合用户个人血压信息，为用户提供帮助。', '结合用户情况，帮助用户降低血压。']
                import random
                thought = random.choice(lis)
            else:
                thoughtIdx = generate_text.find("Thought") + 8
                thought = generate_text[thoughtIdx:].split("\n")[0].strip()
            if generate_text.find("Assistant") == -1:
                content = generate_text
            else:
                outIdx = generate_text.find("Assistant") + 10
                content = generate_text[outIdx:].split("\n")[0].strip()

            return thought, content

        def blood_pressure_pacify(history, query):
            history = [{"role": role_map.get(str(i["role"]), "user"), "content": i["content"]} for i in history]
            his_prompt = "\n".join(
                [("Doctor" if not i["role"] == "user" else "user") + f": {i['content']}" for i in history]
            )
            prompt = blood_pressure_pacify_prompt.format(his_prompt)
            messages = [{"role": "user", "content": prompt}]
            logger.debug("血压安抚模型输入： " + json.dumps(messages, ensure_ascii=False))
            generate_text = callLLM(
                history=messages, max_tokens=1024, top_p=0.8, temperature=0.0, do_sample=False, model="Qwen-72B-Chat"
            )
            logger.debug("血压安抚模型输出： " + generate_text)
            if generate_text.find("\nThought") == -1:
                thought = '在等待医生上门的过程中，我应该安抚患者的情绪，让他保持平静，同时提供一些有助于降低血压的建议。'
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
                    else "尽量保持放松，深呼吸，有助于降低血压。，您可以先尝试静坐，闭上眼睛，缓慢地深呼吸，每次呼吸持续5秒，然后慢慢呼出，也持续5秒。这样可以帮助您放松身心，减轻症状。"
                )
            return thought, content

        def is_visit(history, query):
            if len(history) < 2:
                return False
            if "您需要家庭医生上门帮您服务吗" in history[-2]["content"]:
                prompt = blood_pressure_pd_prompt.format(history[-2]["content"], query)
                messages = [{"role": "user", "content": prompt}]
                if (
                    "是的" in history[-1]["content"]
                    or "好的" in history[-1]["content"]
                    or "需要" in history[-1]["content"]
                    or "嗯" in history[-1]["content"]
                ):
                    return True
                text = callLLM(
                    history=messages,
                    max_tokens=1024,
                    top_p=0.8,
                    temperature=0.0,
                    do_sample=False,
                    model="Qwen-72B-Chat",
                )
                if "YES" in text:
                    return True
                else:
                    return False
            else:
                return False

        def is_pacify(history, query):
            r = [1 for i in history if "您需要家庭医生上门帮您服务吗" in i["content"]]
            return True if sum(r) > 0 else False

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
                thought, content = blood_pressure_inquiry(history, query)
                return {
                    "level": level,
                    "contents": [
                        f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为{get_level(level)}级高血压范围。",
                        "我已经通知了您的女儿和您的家庭医生。",
                        f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                        content,
                    ],
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": True,
                    "call_120": False,
                    "is_visit": False,
                }
            if is_visit(history, query=query):
                thought, content = blood_pressure_pacify(history, query)
                return {
                    "level": level,
                    "contents": ["您的家庭医生回复10分钟后为您上门诊治。同时我也会实时监测您的血压情况。", content],
                    "thought": thought,
                    "idx":1,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": True,
                }
            elif is_pacify(history, query=query):  # 安抚
                thought, content = blood_pressure_pacify(history, query)
                return {
                    "level": level,
                    "contents": [content],
                    "idx":0,
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": True,
                    "blood_trend_gen": False,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                }
            else:  # 问诊
                thought, content = blood_pressure_inquiry(history, query)
                if "？" in content or "?" in content:
                    return {
                        "level": level,
                        "contents": [content],
                        "idx":0,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                    }
                else:  # 出结论
                    return {
                        "level": level,
                        "contents": [content, "您需要家庭医生上门帮您服务吗？"],
                        "idx":0,
                        "thought": thought,
                        "scheme_gen": 0,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                    }

        ihm_health_sbp_list = [116, 118, 132, 121, 128, 123, 128, 117, 132, 134, 124, 120, 80]
        ihm_health_dbp_list = [82, 86, 86, 78, 86, 80, 92, 88, 85, 86, 86, 82, 60]

        # 计算血压波动率,和血压列表的均值对比
        def compute_blood_pressure_trend(x: int, data_list: List) -> float:
            mean_value = sum(data_list) / len(data_list)
            if x > 1.2 * mean_value:
                return 1
            else:
                return 0

        history = kwargs.get("his", [])
        b_history = kwargs.get("backend_history", [])

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
        if ihm_health_sbp > 180 or ihm_health_dbp > 110:  # 三级高血压
            level = 3
            return {
                "level": level,
                "contents": [
                    f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为三级高血压范围",
                    f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                    "我已为您呼叫120。",
                ],
                "idx":-1,
                "thought": "",
                "scheme_gen": -1,
                "scene_ending": True,
                "blood_trend_gen": False,
                "notifi_daughter_doctor": False,
                "call_120": True,
                "is_visit": False,
            }
        elif 179 >= ihm_health_sbp >= 160 or 109 >= ihm_health_dbp >= 100:  # 二级高血压
            level = 2
            return get_second_hypertension(b_history, history, query, level)
        elif 159 >= ihm_health_sbp >= 140 or 99 >= ihm_health_dbp >= 90:  # 一级高血压
            level = 1

            if trend_sbp or trend_dbp:  # 血压波动超过30%
                return get_second_hypertension(b_history, history, query, level=1)
            else:
                if not history:
                    thought, content = blood_pressure_inquiry(history, query)
                    return {
                        "level": level,
                        "contents": [
                            f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为一级高血压范围",
                            f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较大。",
                            content,
                        ],
                        "idx":2,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": True,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                    }
                else:  # 问诊
                    thought, content = blood_pressure_inquiry(history, query)
                    if "？" in content or "?" in content:
                        return {
                            "level": level,
                            "contents": [content],
                            "idx":0,
                            "thought": thought,
                            "scheme_gen": -1,
                            "scene_ending": False,
                            "blood_trend_gen": False,
                            "notifi_daughter_doctor": False,
                            "call_120": False,
                            "is_visit": False,
                        }
                    else:  # 出结论
                        # thought, cont = blood_pressure_pacify(history, query)  #安抚
                        return {
                            "level": level,
                            "contents": [content],
                            "idx":0,
                            "thought": thought,
                            "scheme_gen": 0,
                            "scene_ending": True,
                            "blood_trend_gen": False,
                            "notifi_daughter_doctor": False,
                            "call_120": False,
                            "is_visit": False,
                        }

        elif 139 >= ihm_health_sbp >= 120 or 89 >= ihm_health_dbp >= 80:  # 正常高值
            level = 0
            thought, content = blood_pressure_inquiry(history, query)
            if not history:
                return {
                    "level": level,
                    "contents": [
                        f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为正常高值血压范围",
                        f"健康报告显示您的健康处于为中度失衡状态，本次血压{a}，较日常血压波动较{b}。",
                        content
                    ],
                    "idx":-1,
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                }

            elif "？" in content or "?" in content:
                return {
                    "level": level,
                    "contents": [content],
                    "thought": thought,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "idx":-0,
                }

            else:
                return {
                    "level": level,
                    "contents": [content],
                    "thought": thought,
                    "scheme_gen": 0,
                    "scene_ending": True,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                    "idx":0,
                }
        elif 90 <= ihm_health_sbp < 120 and 80 > ihm_health_dbp >= 60:  # 正常血压
            level = -1
            rules = []
            return {
                "level": 0,
                "contents": [f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为正常血压范围"],
                "thought": "用户血压正常",
                "idx":-1,
                "scheme_gen": -1,
                "scene_ending": True,
                "blood_trend_gen": True,
                "notifi_daughter_doctor": False,
                "call_120": False,
                "is_visit": False,
            }
        else:   # 低血压
            level = -1
            rules = []
            thought, content = blood_pressure_inquiry(history, query)
            if not history:
                return {
                    "level": -1,
                    "contents": [f"您本次血压{ihm_health_sbp}/{ihm_health_dbp}，为低血压范围", "健康报告显示您的健康处于为中度失衡状态，本次血压偏低。", content],
                    "thought": thought,
                    "idx":1,
                    "scheme_gen": -1,
                    "scene_ending": False,
                    "blood_trend_gen": True,
                    "notifi_daughter_doctor": False,
                    "call_120": False,
                    "is_visit": False,
                }
            else:
                  
                if "？" in content or "?" in content:   # 问诊
                    return {
                        "level": level,
                        "contents": [content],
                        "idx":0,
                        "thought": thought,
                        "scheme_gen": -1,
                        "scene_ending": False,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
                    }
                else:  # 出结论
                    # thought, cont = blood_pressure_pacify(history, query)  #安抚
                    return {
                        "level": level,
                        "contents": [content],
                        "idx":0,
                        "thought": thought,
                        "scheme_gen": 0,
                        "scene_ending": True,
                        "blood_trend_gen": False,
                        "notifi_daughter_doctor": False,
                        "call_120": False,
                        "is_visit": False,
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
        history = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]

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
        sys_prompt = self.gsr.prompt_meta_data["event"]["blood_pressure_trend_analysis"]["description"]
        history.append({"role": "system", "content": sys_prompt})

        tst = param["ihm_health_sbp"][0]["date"]
        ted = param["ihm_health_sbp"][-1]["date"]
        content = f"从{tst}至{ted}期间\n"
        if param.get("ihm_health_sbp"):
            content += self.__bpta_compose_value_prompt("收缩压测量数据: ", param["ihm_health_sbp"])
        if param.get("ihm_health_dbp"):
            content += self.__bpta_compose_value_prompt("舒张压测量数据: ", param["ihm_health_dbp"])
        if param.get("ihm_health_hr"):
            content += self.__bpta_compose_value_prompt("心率测量数据: ", param["ihm_health_hr"])
        history.append({"role": "user", "content": content})
        logger.debug(f"血压趋势分析\n{history}")
        response = callLLM(history=history, temperature=0.8, top_p=1, model=model, stream=True)
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"趋势分析结果 length - {len(content)}")
        return content

    def __health_warning_solutions_early_continuous_check__(self, indicatorData: List[Dict]) -> bool:
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
        if return_str:
            valuemap = {-1: "低血压", 0: "正常", 1: "高血压一级", 2: "高血压二级", 3: "高血压三级"}
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

        time_range = {i["date"][:10] for i in param["indicatorData"][0]["data"]}  # 当前的时间范围
        bpl = []
        ihm_health_sbp, ihm_health_dbp, ihm_health_hr = [], [], []
        if is_continuous:
            prompt_str = self.gsr.prompt_meta_data["event"]["warning_solutions_early_continuous"]["description"]
            prompt_template = PromptTemplate.from_template(prompt_str)
            for i in param["indicatorData"]:
                if i["code"] == self.indicatorCodeMap["收缩压"]:  # 收缩压
                    ihm_health_sbp = [j["value"] for j in i["data"]]
                    bpl = self.__health_warning_update_blood_pressure_level__(bpl, ihm_health_sbp)
                elif i["code"] == self.indicatorCodeMap["舒张压"]:  # 舒张压
                    ihm_health_dbp = [j["value"] for j in i["data"]]
                    bpl = self.__health_warning_update_blood_pressure_level__(bpl, ihm_health_dbp)
                elif i["code"] == self.indicatorCodeMap["心率"]:  # 心率
                    ihm_health_hr = [j["value"] for j in i["data"]]
            ihm_health_blood_pressure_level = self.__health_warning_update_blood_pressure_level__(bpl, return_str=True)
            prompt = prompt_template.format(
                time_start=min(time_range),
                time_end=max(time_range),
                ihm_health_sbp=ihm_health_sbp,
                ihm_health_dbp=ihm_health_dbp,
                ihm_health_blood_pressure_level=ihm_health_blood_pressure_level,
                ihm_health_hr=ihm_health_hr,
            )
        else:  # 非连续，只取当日指标
            prompt_str = self.gsr.prompt_meta_data["event"]["warning_solutions_early_not_continuous"]["description"]
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
        response = callLLM(history=history, temperature=0.7, top_p=0.8, model=model, stream=True)
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
            first_match_content = re.findall("```json(.*?)```", content, re.S)[0].strip()
            ret = json.loads(first_match_content)
            reply, purchasing_list = ret["content"], ret["purchasing_list"]
            return reply, purchasing_list

        purchasing_list = kwds.get("purchasing_list")
        prompt = kwds.get("prompt")
        intentCode = "food_purchasing_list_management"
        event_msg = self.gsr.prompt_meta_data["event"][intentCode]
        sys_prompt = event_msg["description"] + event_msg["process"]
        model = self.gsr.model_config["food_purchasing_list_management"]
        sys_prompt = sys_prompt.replace("{purchasing_list}", json.dumps(purchasing_list, ensure_ascii=False))
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
            "intentDesc": self.gsr.intent_desc_map.get(intentCode, "食材采购清单管理-unknown intentCode desc error"),
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

    def food_purchasing_list_generate_by_content(self, query: str, *args, **kwargs) -> Dict:
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
        prompt = prompt_template_str.replace("{{ example_item_js }}", example_item_js).replace("{{ prompt }}", query)
        history = [{"role": "user", "content": prompt}]
        logger.debug(f"根据用户输入生成采购清单 LLM Input: {json.dumps(history, ensure_ascii=False)}")
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

    def rec_diet_reunion_meals_restaurant_selection(self, history=[], backend_history: List = [], **kwds) -> str:
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
                self.gsr.prompt_meta_data["event"]["reunion_meals_restaurant_selection"]["description"]
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
                name, role, content = item.get("name"), item.get("role"), item.get("content")
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
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.9,
                top_p=0.8,
                top_k=-1,
                repetition_penalty=1.1,
                stream=True,
            )

            t_st = time.time()
            ret_content = ""
            for i in response:
                msg = i.choices[0].delta.to_dict()
                text_stream = msg.get("content")
                if text_stream:
                    ret_content += text_stream
                    print(text_stream, end="", flush=True)
                    yield make_ret_item(text_stream, False, [])
            messages.append({"role": "assistant", "content": ret_content})

            time_cost = round(time.time() - t_st, 1)
            logger.debug(f"共策回复: {ret_content}")
            logger.success(
                f"Model {model} generate costs summary: " + f"total_texts:{len(ret_content)}, "
                f"complete cost: {time_cost}s"
            )
            yield make_ret_item("", True, messages[1:])
        except openai.APIError as e:
            logger.error(f"Model {model} generate error: {e}")
            yield make_ret_item("内容过长,超出模型处理氛围", True, [])
        except Exception as err:
            logger.error(f"Model {model} generate error: {err}")
            yield make_ret_item(repr(err), True, [])

    def regist_aigc_functions(self):
        self.funcmap = {}
        self.funcmap["aigc_functions_single_choice"] = self.__single_choice__
        self.funcmap["aigc_functions_report_interpretation"] = self.__report_ocr_classification__

    def __single_choice__(self, prompt: str, options: List[str], **kwargs):
        """单项选择功能

        - Args:
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        model = self.gsr.model_config.get("aigc_functions_single_choice", "Qwen-14B-Chat")
        prompt_template_str = self.gsr.prompt_meta_data["event"]["aigc_functions_single_choice"]["description"]
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        query = prompt_template.format(options=options, prompt=prompt)
        messages = [{"role": "user", "content": query}]
        logger.debug(f"Single choice LLM Input: {json.dumps(messages, ensure_ascii=False)}")
        response = callLLM(history=messages, model=model, temperature=0.7, top_p=0.5, stream=True)
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"Single choice LLM Output: {content}")
        if content == "选项与要求不符":
            return content
        else:
            if content not in options:
                logger.error(f"Single choice error: {content} not in options")
                return "选项与要求不符"
        return content

    def __ocr_report__(self, file_path):
        """报告OCR功能"""
        raw_result, _ = self.ocr(file_path)
        docs = ""
        if raw_result:
            process_ocr_result = [line[1] for line in raw_result]
            logger.debug(f"Report interpretation OCR result: {dumpJS(process_ocr_result)}")
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

    def __plot_rectangle(self, tmp_path, file_path, rectangles_with_text):
        """为识别的报告内容画出矩形框"""
        image_io = Image.open(file_path)
        draw = ImageDraw.Draw(image_io)
        for rectangle, text in rectangles_with_text:
            line_value = int(0.002 * sum(image_io.size))
            fontsize = line_value * 6
            image_font = ImageFont.truetype(str(self.image_font_path), fontsize)
            draw.rectangle(rectangle, outline="blue", width=line_value)
            draw.text(
                (rectangle[0] - fontsize*2, rectangle[1] - fontsize - 15), text, font=image_font, fill="red"
            )
        save_path = tmp_path.joinpath(file_path.stem + "_rect" + file_path.suffix)
        image_io.save(save_path)
        logger.debug(f"Plot rectangle image saved to {save_path}")
        return save_path

    def __upload_image(self, save_path):
        """上传图片到服务器"""
        url = self.gsr.api_config["ai_backend"] + "/file/uploadFile"
        payload = {"businessType": "reportAnalysis"}
        if save_path.suffix.lower() in [".jpg", ".jpeg"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/jpeg"))]
        elif save_path.suffix.lower() in [".png"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/png"))]
        else:
            files = [("file", (save_path.name, open(save_path, "rb"), f"image/{save_path.suffix.lower()[1:]}"))]
        resp = self.session.post(url, data=payload, files=files)
        if resp.status_code == 200:
            remote_image_url = resp.json()["data"]
        else:
            logger.error(f"Upload image error: {resp.text}")
            remote_image_url = ""
        return remote_image_url

    def __report_ocr_classification_make_text_group__(
        self, file_path: Union[str, Path], raw_result, tmp_path, **kwargs
    ) -> str:
        """报告OCR结果分组"""
        sysprompt = (
            "You are a helpful assistant.\n"
            "# 任务描述\n"
            "1. 下面我将给你报告OCR提取的内容，它是有序的，优先从上到下从左到右\n"
            "2. 请你参考给出的内容的前后信息，按内容的前后顺序对报告的内容进行归类，类别最多5个\n"
            # "3. 可选类别有[报告标题,基础信息,影像图片,影像所见,诊断意见,医疗建议,检查方法,检查医生]\n"
            "3. 只给出各类别开始内容和结尾内容对应的index, 所有内容的index都应当被包含\n"
            '4. 输出格式参考:\n```json\n{"分类1": [start_idx_1, end_idx_1], "分类2": [start_idx_2, end_idx_2], "分类3": [start_idx_3, end_idx_3],...}\n```其中start_idx_2=end_idx_1+1, start_idx_3=end_idx_2+1'
        )
        content_index = {idx: text for idx, text in enumerate([i[1] for i in raw_result])}
        messages = [{"role": "system", "content": sysprompt}, {"role": "user", "content": str(content_index)}]

        logger.debug(f"报告解读文本分组 LLM Input:\n{dumpJS(messages)}")
        response = openai.ChatCompletion.create(
            model="Qwen-72B-Chat",
            messages=messages,
            temperature=0.7,
            n=1,
            top_p=0.3,
            top_k=-1,
            presence_penalty=0,
            frequency_penalty=0.5,
            stream=True,
        )
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"报告解读文本分组: {content}")

        content = re.findall("```json(.*?)```", content, re.S)[0].strip()
        try:
            loc = json.loads(content)
        except:
            loc = {}
        try:
            rectangles_with_text = []
            for topic, index_range in loc.items():
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
            save_path = self.__plot_rectangle(tmp_path, file_path, rectangles_with_text)
        except Exception as e:
            logger.exception(f"Report interpretation error: {e}")
        remote_image_url = self.__upload_image(save_path)
        return remote_image_url

    def __report_ocr_classification__(
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
                r = self.session.get(image_url)
                file_path = tmp_path.joinpath(basename(image_url))
                with open(file_path, mode="wb") as f:
                    f.write(r.content)
            elif kwargs.get("file_path"):
                file_path = kwargs.get("file_path")
                image_url = self.__upload_image(file_path)
            else:
                logger.error(f"Report interpretation error: file_path or url not found")
            return image_url, file_path

        def jude_report_type(docs: str, options: List[str]) -> str:
            query = f"{docs}\n\n请你判断以上报告属于哪个类型,从给出的选项中选择: {options}, 要求只输出选项答案, 请不要输出其他内容\n\nOutput:"
            messages = [{"role": "user", "content": query}]
            response = callLLM(history=messages, model="Qwen-72B-Chat", temperature=0.7, top_p=0.5, stream=True)
            report_type = accept_stream_response(response, verbose=False)
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
        docs, raw_result, process_ocr_result = self.__ocr_report__(file_path)
        if not docs:
            return self.__report_interpretation_result__(
                msg="未识别出报告内容，请重新尝试", ocr_result="您的报告内容无法解析，请重新尝试."
            )
        try:
            remote_image_url = self.__report_ocr_classification_make_text_group__(file_path, raw_result, tmp_path)
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
            report_type = jude_report_type(docs, options)
        else:
            report_type = "其他"
        return self.__report_interpretation_result__(
            ocr_result=docs, report_type=report_type, remote_image_url=remote_image_url
        )

    def call_function(self, **kwargs):
        """调用函数

        - Args Example:
            ```json
            {
                "intentCode": "",
                "prompt": "",
                "options": []
            }
            ```
        - Args:
            intentCode (str): 意图代码
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        intent_code = kwargs.get("intentCode")
        # TODO intentCode -> funcCode
        func_code = (
            self.gsr.intent_aigcfunc_map.get(intent_code)
            if self.gsr.intent_aigcfunc_map.get(intent_code)
            else intent_code
        )
        if not self.funcmap.get(func_code):
            logger.error(f"intentCode {func_code} not found in funcmap")
            raise RuntimeError(f"Code not supported.")
        try:
            content = self.funcmap.get(func_code)(**kwargs)
        except Exception as e:
            logger.exception(f"call function {func_code} error: {e}")
            raise RuntimeError(f"Call function error.")
        return content


if __name__ == "__main__":
    expert_model = expertModel(InitAllResource())
    # expert_model.__rec_diet_eval__(param)sss

    # param = testParam.param_pressure_trend
    # expert_model.__blood_pressure_trend_analysis__(param)

    # param = testParam.param_rec_diet_reunion_meals_restaurant_selection
    # generator = expert_model.__rec_diet_reunion_meals_restaurant_selection__(**param)
    # while True:
    #     yield_item = next(generator)
    #     print(yield_item)

    # param = testParam.param_dev_single_choice
    param = testParam.param_dev_report_interpretation
    expert_model.call_function(**param)
    # param = testParam.param_dev_tool_compute_blood_pressure
    # expert_model.tool_compute_blood_pressure(**param)
