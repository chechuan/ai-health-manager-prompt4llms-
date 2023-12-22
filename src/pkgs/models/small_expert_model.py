# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-05 15:14:07
@desc    :   小专家模型
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from typing import List

from data.test_param.test import testParam
from src.prompt.model_init import chat_qwen
from src.utils.Logger import logger
from src.utils.module import clock, initAllResource


class expertModel:
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

    def __rec_diet_reunion_meals_restaurant_selection__(self, param) -> str:
        """聚餐场景
        提供各餐厅信息
        群组中各角色聊天内容
        推荐满足角色提出条件的餐厅, 并给出推荐理由

        Args
            history List[Dict]: 群组对话信息
        
        Example
            ```json
            {
                "history": [
                    {"role":"小明", "content":"我想吃炸薯条"},
                    {"role":"妈妈", "content":"下次再带你吃炸薯条,我们现在讨论过年晚上去哪儿吃,要不吃中餐吧,川菜怎么样,咱家人都能吃辣"},
                    {"role":"爸爸", "content": "好啊,那去吃川菜,川菜确实很好吃"},
                    {"role":"奶奶", "content":"川菜听起来不错，但是你们年轻人啊，吃东西总是太辣了，对肠胃不好。要不我们还是去吃个团圆饭，清淡一点的好。"},
                    {"role":"姑姑", "content":"奶奶说得对，我们还是要照顾到大家的口味。不如我们去吃粤菜，既有海鲜又有炖汤，健康又美味。"},
                    {"role":"小姨", "content": "粤菜是个好主意，我听说附近开了一家新的粤菜馆，评价很不错，我们晚上可以去试试。"}
                ]
            }
            ```
        """
        content = "根据提供的信息, 推荐你们去元善家宴餐厅, 理由是: 环境优雅, 菜品丰富, 服务热情, 价格亲民, 适合聚餐."
        return content

if __name__ == "__main__":
    param = testParam.param_pressure_trend
    initAllResource()
    expert_model = expertModel()
    # expert_model.__rec_diet_eval__(param)
    expert_model.__blood_pressure_trend_analysis__(param)