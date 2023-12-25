# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-05 15:14:07
@desc    :   小专家模型
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import re
import sys
from pathlib import Path

from sympy import EX

import chat
from chat import qwen_chat

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from typing import Dict, List

from config.constrant import DEFAULT_RESTAURANT_MESSAGE
from data.test_param.test import testParam
from src.prompt.model_init import chat_qwen
from src.utils.Logger import logger
from src.utils.module import clock, initAllResource


class expertModel:
    def __init__(self, gsr) -> None:
        self.gsr = gsr
        
    def check_number(x: str or int or float, key: str):
        try:
            x = float(x)
            return float(x)
        except Exception as err:
            raise f"{key} must can be trans to number."

    @staticmethod
    def tool_compute_bmi(weight: float or int, height: float or int) -> float():
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
    def tool_compute_max_heart_rate(age: float or int) -> int:
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
    def tool_compute_exercise_target_heart_rate_for_old(age: float or int) -> int:
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
    def tool_assert_body_status(age: float or int, bmi: float or int) -> str:
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
            elif 20 <= bmi < 26.9 :
                status = "体重正常"
            elif 26.9 <= bmi < 28:
                status = "超重"
        if not status and bmi >= 28:
            status = "肥胖"
        else:
            status = "体重正常"
        return status

    @clock
    def __rec_diet_eval__(self, param):
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
        prompt = (
            f"本餐建议热量: {param['recommend_heat_target']}\n"
            "实际吃的:\n"
        )

        recipes_prompt_list = []
        for recipe in param["recipes"]:
            tmp = f"{recipe['name']}"
            if recipe['weight']:
                tmp += f": {recipe['weight']}g"
            elif recipe['unit']:
                tmp += f": {recipe['unit']}"
            recipes_prompt_list.append(tmp)
        
        recipes_prompt = "\n".join(recipes_prompt_list)
        prompt += recipes_prompt
        history = [{"role":"system", "content": sys_prompt},{"role":"user", "content": prompt}]

        logger.debug(f"饮食点评 Prompt:\n{json.dumps(history, ensure_ascii=False)}")
        
        content = chat_qwen(history=history, temperature=0.7, top_p=0.8)
        
        logger.debug(f"饮食点评 Result:\n{content}")
        
        return content
    
    def __bpta_compose_value_prompt(self, key: str = "对值的解释", data: List = []):
        """血压趋势分析, 拼接值"""
        value_list = [i['value'] for i in data]
        content = f"{key}{value_list}\n"
        return content
    
    @clock
    def __blood_pressure_trend_analysis__(self, param: dict) -> str:
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
        history = []
        sys_prompt = "请你扮演一个专业的家庭医师,结合提供的信息帮助用户分析血压和心率变化整体趋势并给出健康建议,不超过200字."
        history.append({"role":"system", "content": sys_prompt})
        
        tst = param['ihm_health_sbp'][0]['date']
        ted = param['ihm_health_sbp'][-1]['date']
        content = f"从{tst}至{ted}期间\n"
        if param.get('ihm_health_sbp'):
            content += self.__bpta_compose_value_prompt("收缩压测量数据: ", param['ihm_health_sbp'])
        if param.get('ihm_health_dbp'):
            content += self.__bpta_compose_value_prompt("舒张压测量数据: ", param['ihm_health_dbp'])
        if param.get('ihm_health_hr'):
            content += self.__bpta_compose_value_prompt("心率测量数据: ", param['ihm_health_hr'])
        history.append({"role":"user", "content": content})
        logger.debug(f"血压趋势分析\n{history}")
        content = chat_qwen(history=history, temperature=0.7, top_p=0.8, model="Qwen-1_8B-Chat")
        logger.debug(f"趋势分析结果: {content}")
        return content

    def __food_purchasing_list_intent_(self, content):
        """食材采购清单过程中的意图识别

        - Args
            content [Str] 清单页面说的话

        - Return
            intentCode [Str] code: 清单管理or关闭清单/提交

        """
        code = "manage" or "quit"
        return code

    def __food_purchasing_list_manage__(self, **kwds):
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
        intentCode = kwds.get("intentCode", "create_food_purchasing_list")
        purchasing_list = kwds.get("purchasing_list")
        prompt = kwds.get("prompt")

        if kwds['intentCode'] == "create_food_purchasing_list":     # 固定内容
            ret = [
                {"name": "鸡蛋", "quantity": 500, "unit": "g"},
                {"name": "鸡胸肉", "quantity": 500, "unit": "g"},
                {"name": "酱油", "quantity": 1, "unit": "瓶"},
                {"name": "牛腩", "quantity": 200, "unit": "g"},
                {"name": "菠菜", "quantity": 500, "unit": "g"}
            ]
        else:
            intentCode = self.__food_purchasing_list_intent_(prompt)
            purchasing_list = []
            if intentCode == "quit":
                reply = "好的,已为您提交"
            elif intentCode == "manage":
                # TODO 增加意图判断 是管理还是结束
                reply = "订单修改成功"
                event_msg = self.gsr.prompt_meta_data['event'][intentCode]
                sys_prompt = event_msg['description'] + event_msg['process']
                model = self.gsr.model_config['food_purchasing_list_management']
                sys_prompt = sys_prompt.replace("{purchasing_list}", json.dumps(purchasing_list, ensure_ascii=False))
                query = sys_prompt + f"\n\n用户说: {prompt}\n" + f"现采购清单:\n```json\n"
                content = chat_qwen(query=query, temperature=0.7, model=model, top_p=0.8, max_tokens=200)
                try:
                    first_match_content = re.findall("(.*?)```", content, re.S)[0].strip()
                    purchasing_list = json.loads(first_match_content)
                except Exception as err:
                    logger.exception(content)
                    content = chat_qwen(query=query, temperature=0.7, model=model, top_p=0.8, max_tokens=300)
                    try:
                        first_match_content = re.findall("(.*?)```", content, re.S)[0].strip()
                        purchasing_list = json.loads(first_match_content)
                    except Exception as err:
                        logger.exception(err)
                        logger.critical(content)
                        purchasing_list = [
                            {"name": "鸡蛋", "quantity": "500", "unit": "g"},
                            {"name": "鸡胸肉", "quantity": "500", "unit": "g"},
                            {"name": "酱油", "quantity": "1", "unit": "瓶"},
                            {"name": "牛腩", "quantity": "200", "unit": "g"},
                            {"name": "菠菜", "quantity": "500", "unit": "g"}
                        ]
        ret = {"purchasing_list": purchasing_list, "content": reply, "intentCode": intentCode}
        return ret

    def __rec_diet_reunion_meals_restaurant_selection__(self, history=[], backend_history=[], **kwds) -> str:
        """聚餐场景
        提供各餐厅信息
        群组中各角色聊天内容
        推荐满足角色提出条件的餐厅, 并给出推荐理由

        Args
            history List[Dict]: 群组对话信息
        
        Example
            ```json
            {
                "restaurant_message":"候选餐厅信息",
                "history": [
                    {"name":"爸爸", "content": "咱们每年年夜饭都在家吃，咱们今年下馆子吧！大家有什么意见？"},
                    {"name":"妈妈", "content":"年夜饭我姐他们一家过来一起，咱们一共10个人，得找一个能坐10个人的包间，预算一两千吧"},
                    {"name":"爷爷", "content":"年夜饭得有鱼，找一家中餐厅，做鱼比较好吃的，孩子奶奶腿脚不太好，离家近点吧"},
                    {"name":"奶奶", "content":"我没什么意见，环境好一点有孩子活动空间就可以。"},
                    {"name":"大儿子", "content":"我想吃海鲜！吃海鲜！"}
                ],
                "backend_history": []
            }
            ```
        """
        model = self.gsr.model_config['reunion_meals_restaurant_selection']
        restaurant_message = kwds.get("restaurant_message", DEFAULT_RESTAURANT_MESSAGE)
        event_msg = self.gsr.prompt_meta_data['event']['reunion_meals_restaurant_selection']
        query = ""
        sys_prompt = event_msg['description'].replace("{RESTAURANT_MESSAGE}", restaurant_message)
        all_user_input = "群里的聊天信息:\n"
        for item in history:
            name, role, content = item.get("name"), item.get("role"), item.get("content")
            user_input = ""
            user_input += name
            if role:
                user_input += f" {role}"
            user_input += f": {content}\n"
            all_user_input += user_input
        query = sys_prompt + "\n\n" + all_user_input
        input_history = [{"role": "user", "content": query}]
        content = chat_qwen(history=input_history, temperature=0.7, top_p=0.8, model=model)
        logger.debug("餐厅推荐:\n" + content)
        return content

if __name__ == "__main__":
    param = testParam.param_pressure_trend
    initAllResource()
    expert_model = expertModel()
    # expert_model.__rec_diet_eval__(param)
    expert_model.__blood_pressure_trend_analysis__(param)