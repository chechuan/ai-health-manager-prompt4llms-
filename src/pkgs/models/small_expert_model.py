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
from os.path import basename
from pathlib import Path

import openai
from requests import Session

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from datetime import datetime, timedelta
from typing import Dict, List, Union

from langchain.prompts.prompt import PromptTemplate
from rapidocr_onnxruntime import RapidOCR

from data.constrant import DEFAULT_RESTAURANT_MESSAGE, HOSPITAL_MESSAGE
from data.test_param.test import testParam
from src.prompt.model_init import callLLM
from src.utils.Logger import logger
from src.utils.module import (InitAllResource, accept_stream_response, clock,
                              compute_blood_pressure_level, dumpJS, get_intent)
from data.constrant import *


class expertModel:
    indicatorCodeMap = {"收缩压": "lk1589863365641", "舒张压": "lk1589863365791", "心率": "XYZBXY001005"}
    session = Session()
    ocr = RapidOCR()

    def __init__(self, gsr) -> None:
        self.gsr = gsr
        self.gsr.expert_model = self
        self.regist_aigc_functions()

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
    def emotions(cur_date, level):
        prompt = emotions_prompt.format(cur_date, level)
        messages = [{"role": "user", "content": prompt}]
        generate_text = callLLM(history=messages, max_tokens=1024, top_p=0.8,
                temperature=0.0, do_sample=False, model='Qwen-72B-Chat')
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        outIdx = generate_text.find("\nOutput") + 8
        content = generate_text[outIdx:].split("\n")[0].strip()
        return {'thought': thought, 'content': content}
    
    @staticmethod
    def weight_trend(cur_date, weight):
        prompt = weight_trend_prompt.format(cur_date, weight)
        messages = [{"role": "user", "content": prompt}]
        generate_text = callLLM(history=messages, max_tokens=1024, top_p=0.8,
                temperature=0.0, do_sample=False, model='Qwen-72B-Chat')
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        outIdx = generate_text.find("\nOutput") + 8
        content = generate_text[outIdx:].split("\n")[0].strip()
        return {'thought': thought, 'content': content}
    
    @staticmethod
    def fat_reduction(history, cur_date, weight):
        if not history:
            return f'您今日体重为{weight}'
        prompt = weight_trend_prompt.format(cur_date, weight)
        messages = [{"role": "user", "content": prompt}]
        generate_text = callLLM(history=messages, max_tokens=1024, top_p=0.8,
                temperature=0.0, do_sample=False, model='Qwen-72B-Chat')
        thoughtIdx = generate_text.find("\nThought") + 9
        thought = generate_text[thoughtIdx:].split("\n")[0].strip()
        outIdx = generate_text.find("\nOutput") + 8
        content = generate_text[outIdx:].split("\n")[0].strip()
        return {'thought': thought, 'content': content}



    @staticmethod
    def tool_rules_blood_pressure_level(ihm_health_sbp: int, ihm_health_dbp: int, **kwargs) -> dict:
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
        ihm_health_sbp_list = [134, 123, 142, 114, 173, 164, 121]
        ihm_health_dbp_list = [88, 66, 78, 59, 100, 90, 60]

        # 计算血压波动率,和血压列表的均值对比
        def compute_blood_pressure_trend(x: int, data_list: List) -> float:
            mean_value = sum(data_list) / len(data_list)
            rate = (x - mean_value) / mean_value
            if rate > 0.3:
                return 1
            else:
                return 0
        
        history = kwargs.get('history', [])
        if ihm_health_sbp > 180 or ihm_health_dbp > 110:
            level = 3
            rules = ["呼叫救护车"]
        elif 179 >= ihm_health_sbp >= 160 or 109 >= ihm_health_dbp >= 100:
            level = 2
            if not history:
                return {'level':2, 'rules':[], 'contents': [f'张叔叔，发现您刚刚的血压是{ihm_health_sbp}/{ihm_health_dbp_list},血压偏高']}
            
            rules = ["预问诊", "是否通知家人", "是否通知家庭医师"]
        elif 159 >= ihm_health_sbp >= 140 or 99 >= ihm_health_dbp >= 90:
            level = 1
            trend_sbp = compute_blood_pressure_trend(ihm_health_sbp, ihm_health_sbp_list)
            trend_dbp = compute_blood_pressure_trend(ihm_health_dbp, ihm_health_dbp_list)
            if trend_sbp or trend_dbp:
                rules = ["预问诊", "智能呼叫工具"]
            else:
                rules = ["预问诊", "嘱托"]
        elif 139 >= ihm_health_sbp >= 120 or 89 >= ihm_health_dbp >= 80:
            level = 0
            rules = ["预问诊", "嘱托"]
        else:
            level = -1
            rules = []
        return {"level": level, "rules": rules}
    
    def blood_pressure_inquiry(history)

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
        self.funcmap["aigc_functions_report_interpretation"] = self.__report_interpretation__

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
        result, _ = self.ocr(file_path)
        docs = ""
        if result:
            ocr_result = [line[1] for line in result]
            logger.debug(f"Report interpretation OCR result: {dumpJS(ocr_result)}")
            docs += "\n".join(ocr_result)
        else:
            logger.error(f"Report interpretation OCR result is empty")
        return docs, ocr_result

    def __report_interpretation_result__(
        self,
        ocr_result: List[str] = [],
        msg: str = "Unknown Error",
        report_type: str = "Unknown Type",
    ):
        """报告解读结果

        - Args:
            ocr_result (List[str]): OCR结果
            msg (str): 报告解读内容
            report_type (str): 报告类型

        - Returns:
            Dict: 报告解读结果
        """
        return {"ocr_result": ocr_result, "report_interpretation": msg, "report_type": report_type}

    def __report_interpretation__(self, **kwargs) -> str:
        """报告解读功能

        - Args:
            file_path (str): 报告文件路径

        - Returns:
            str: 报告解读内容
        """

        def prepare_file(**kwargs):
            tmp_path = Path(f".tmp/images")
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
            else:
                logger.error(f"Report interpretation error: file_path or url not found")

            return file_path

        file_path = prepare_file(**kwargs)
        if not file_path:
            return self.__report_interpretation_result__(msg="请输入信息源")
        docs, ocr_result = self.__ocr_report__(file_path)
        if not docs:
            return self.__report_interpretation_result__(msg="未识别出报告内容，请重新尝试")

        # 报告异常信息解读
        prompt_template_str = "You are a helpful assistant.\n" "# 任务描述\n" "请你为我解读报告中的异常信息"
        messages = [{"role": "system", "content": prompt_template_str}, {"role": "user", "content": docs}]
        logger.debug(f"Report interpretation LLM Input: {dumpJS(messages)}")
        response = callLLM(history=messages, model="Qwen-14B-Chat", temperature=0.7, top_p=0.5, stream=True)
        content = accept_stream_response(response, verbose=False)
        logger.debug(f"Report interpretation LLM Output: {content}")

        # 增加报告类型判断
        if kwargs.get("options"):
            report_type = self.__single_choice__(docs, kwargs["options"] + ["其他"])
            if report_type not in kwargs["options"]:
                report_type = "其他"
        else:
            report_type = "其他"
        return self.__report_interpretation_result__(ocr_result=ocr_result, msg=content, report_type=report_type)

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
