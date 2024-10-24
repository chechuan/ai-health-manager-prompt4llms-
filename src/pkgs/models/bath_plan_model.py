# -*- encoding: utf-8 -*-
"""
@Time    :   2024-10-22 10:01:21
@desc    :   泡浴方案功能
@Author  :   车川
@Contact :   1163317515@qq.com
"""

import random
import json
from src.utils.database import MysqlConnector
import re
from src.utils.resources import InitAllResource
from src.utils.Logger import logger


class BathPlanModel:
    def __init__(self, gsr):
        self.gsr = gsr
        self.mysql_conn = MysqlConnector(**self.gsr.mysql_config)
        self.data = self.load_data()

    def load_data(self):
        """
        从数据库中加载泡浴方案和温泉作用表的数据
        """
        tables = [
            "bath_category_effects", "cleaned_bath_plan"
        ]
        data = {table: self.mysql_conn.query(f"select * from {table}") for table in tables}
        return data

    # 读取泡浴方案和温泉作用的数据
    def load_bath_plan_data(self):
        bath_plan_data = self.data['cleaned_bath_plan']
        spring_effects_data = self.data['bath_category_effects']

        # 将温泉作用数据转换为字典
        spring_effects_dict = {row['category']: row['effect'] for row in spring_effects_data}
        # 将泡浴方案数据转换为字典
        bath_plan_dict = {row['problem_combination']: row['bath_plan'] for row in bath_plan_data}

        return bath_plan_dict, spring_effects_dict

    def parse_bath_plan(self, bath_plan_string, spring_effects_data, gender, is_default=False):
        """
        解析泡浴方案字符串，将温泉名和作用提取出来
        """
        main_pool_time = "建议15-20分钟"
        secondary_pool_time = "建议10-15分钟"

        bath_plan = []
        pattern = r'(\w+泉)(?:（(主池)）)?'
        matches = re.findall(pattern, bath_plan_string)

        for match in matches:
            spring_name = match[0]
            pool_type = "主池" if match[1] == '主池' else "副池"

            suggested_time = main_pool_time if pool_type == "主池" else secondary_pool_time

            effect = spring_effects_data.get(spring_name, "未知作用")

            bath_plan.append({
                "spring_name": spring_name,
                "pool_type": pool_type,
                "suggested_time": suggested_time,
                "effect": effect
            })

        if not is_default:
            if random.choice([True, False]):  # 随机决定是否添加男士泉或女士泉
                extra_spring = "男士泉" if gender == "男" else "女士泉"
                pool_type = "主池" if random.choice([True, False]) else "副池"
                suggested_time = main_pool_time if pool_type == "主池" else secondary_pool_time
                bath_plan.append({
                    "spring_name": extra_spring,
                    "pool_type": pool_type,
                    "suggested_time": suggested_time,
                    "effect": spring_effects_data.get(extra_spring, "未知作用")
                })

        return bath_plan

    def generate_bath_plan(self, user_data):
        """
        根据用户输入的健康问题生成泡浴方案
        """
        bath_plan_data, spring_effects_data = self.load_bath_plan_data()

        full_plan = []
        user_problems = []

        if user_data.get('skin_problems'):
            user_problems.append('皮肤问题')
        if user_data.get('pain_problems'):
            user_problems.append('疼痛问题')
        if user_data.get('fatigue_problems'):
            user_problems.append('疲劳问题')
        if user_data.get('sleep_problems'):
            user_problems.append('睡眠问题')

        if len(user_problems) == 1:
            combination_key = user_problems[0]
        elif len(user_problems) > 1:
            combination_key = '+'.join(user_problems)
        else:
            return self.default_bath_plan(user_data['gender'], spring_effects_data, bath_plan_data)

        if combination_key in bath_plan_data:
            bath_plan_string = bath_plan_data[combination_key]
            full_plan = self.parse_bath_plan(bath_plan_string, spring_effects_data, user_data['gender'], is_default=False)
            health_analysis = self.get_health_analysis(combination_key, user_data['gender'])
        else:
            return {"msg": "未找到匹配的泡浴方案"}

        return {
            "head": 200,
            "items": {
                "bath_plan": full_plan,
                "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                "output_basis": "根据您的问卷结果，结合特色温泉的不同功效，从整体有效性、合理性、提升效果的角度为您推荐以下泡浴方案，仅供参考",
                "health_analysis": health_analysis
            },
            "msg": ""
        }

    def get_health_analysis(self, combination_key, gender):
        """
        根据问题组合键和性别获取健康分析
        """
        bath_plan_data = self.data['cleaned_bath_plan']
        health_analysis_data = next(
            (row['health_analysis'] for row in bath_plan_data if row['problem_combination'] == combination_key), None)

        if health_analysis_data:
            try:
                health_analysis_dict = json.loads(health_analysis_data)
                return health_analysis_dict.get(gender, "未找到对应性别的健康分析")
            except Exception as e:
                return f"健康分析解析出错: {e}"
        else:
            return "未找到健康分析"

    def default_bath_plan(self, gender, spring_effects_data, bath_plan_data):
        """
        当用户未选择问题时，返回默认方案
        """
        if "未选择问题" in bath_plan_data:
            default_plan_string = bath_plan_data["未选择问题"]
            default_plan_string = default_plan_string.replace("专享泉", "男士泉" if gender == "男" else "女士泉")
            default_plan = self.parse_bath_plan(default_plan_string, spring_effects_data, gender, is_default=True)
            health_analysis = self.get_health_analysis("未选择问题", gender)

            return {
                "head": 200,
                "items": {
                    "bath_plan": default_plan,
                    "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                    "output_basis": "根据您的问卷结果，结合特色温泉的不同功效，从整体有效性、合理性、提升效果的角度为您推荐以下泡浴方案，仅供参考",
                    "health_analysis": health_analysis
                },
                "msg": ""
            }
        else:
            return {"msg": "未找到默认的泡浴方案"}


if __name__ == '__main__':
    # 测试输入数据
    input_data = {
      "intentcode_bath_plan": "aigc_functions_generate_bath_plan",
      "gender": "男",
      "skin_problems": False,
      "pain_problems": False,
      "fatigue_problems": True,
      "sleep_problems": False
    }

    # 实例化 BathPlanModel 类并生成泡浴方案
    gsr = InitAllResource()
    generator = BathPlanModel(gsr)
    recommended_bath_plan = generator.generate_bath_plan(input_data)
    print(recommended_bath_plan)