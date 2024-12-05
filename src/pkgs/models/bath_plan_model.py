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
from src.utils.module import run_in_executor, wrap_content_for_frontend



class BathPlanModel:
    # 将常量提取为类变量
    NOTICE = "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适"
    OUTPUT_BASIS = "根据您的测评结果，结合温泉功效特色，从整体有效性角度为您推荐泡汤方案，具体内容如下："

    def __init__(self, gsr):
        self.gsr = gsr
        self.mysql_conn = MysqlConnector(**self.gsr.mysql_config)
        self.data = self.load_data()

    def load_data(self):
        """
        从数据库中加载泡浴方案和温泉作用表的数据
        """
        tables = [
            "bath_category_effects", "cleaned_bath_plan", "tweet_articles"
        ]
        data = {table: self.mysql_conn.query(f"select * from {table}") for table in tables}
        return data

    # 读取泡浴方案和温泉作用的数据
    async def load_bath_plan_data(self):

        bath_plan_data = self.data['cleaned_bath_plan']
        spring_effects_data = self.data['bath_category_effects']

        # 将温泉作用数据转换为字典
        spring_effects_dict = {row['category']: row['effect'] for row in spring_effects_data}
        # 将泡浴方案数据转换为字典
        bath_plan_dict = {row['problem_combination']: row['bath_plan'] for row in bath_plan_data}

        return bath_plan_dict, spring_effects_dict

    async def parse_bath_plan(self, bath_plan_string, spring_effects_data, gender, is_default=False):
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

    async def generate_bath_plan(self, user_data):
        """
        根据用户输入的健康问题生成泡浴方案
        """
        bath_plan_data, spring_effects_data = await self.load_bath_plan_data()

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
            return await self.default_bath_plan(user_data.get("gender", "男"), spring_effects_data, bath_plan_data)

        if combination_key in bath_plan_data:
            bath_plan_string = bath_plan_data[combination_key]
            full_plan = await self.parse_bath_plan(bath_plan_string, spring_effects_data, user_data.get("gender", "男"), is_default=False)
            health_analysis = await self.get_health_analysis(combination_key, user_data.get("gender", "男"))
        else:
            return {"msg": "未找到匹配的泡浴方案"}

        return {
                "bath_plan": full_plan,
                "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                "output_basis": "根据健康问卷测评结果，结合温泉功效特色，从整体有效性角度为您推荐泡浴方案，仅供参考",
                "health_analysis": health_analysis
            }

    async def get_health_analysis(self, combination_key, gender):
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

    async def default_bath_plan(self, gender, spring_effects_data, bath_plan_data, version=None):
        """
        当用户未选择问题时，返回默认方案
        根据版本返回不同的数据结构
        """
        if "未选择问题" in bath_plan_data:
            default_plan_string = bath_plan_data["未选择问题"]
            default_plan_string = default_plan_string.replace("专享泉", "男士泉" if gender == "男" else "女士泉")
            default_plan = await self.parse_bath_plan(default_plan_string, spring_effects_data, gender, is_default=True)
            health_analysis = await self.get_health_analysis("未选择问题", gender)

            # 根据版本判断返回的结构
            if version == "v_1.1.0":
                # v_1.1.0 版本只返回 full_plan 和 health_analysis
                return default_plan, health_analysis
            else:
                # 其他版本的默认返回结构
                return {
                    "head": 200,
                    "items": {
                        "bath_plan": default_plan,
                        "notice": self.NOTICE,
                        "output_basis": self.OUTPUT_BASIS,
                        "health_analysis": health_analysis
                    },
                    "msg": ""
                }
        else:
            return {"msg": "未找到默认的泡浴方案"}

    async def generate_markdown_bath_plan(self, full_plan, health_analysis):
        """
        根据泡浴方案和健康分析生成 Markdown 格式内容
        """
        # 健康分析部分
        markdown = f"根据您的症状描述，{health_analysis}\n\n"

        # 添加 output_basis 说明
        markdown += f"{self.OUTPUT_BASIS}\n\n"

        markdown += "###### &nbsp;\n"

        # 判断是否有温泉池，并生成方案部分
        for idx, step in enumerate(full_plan):
            if idx == 0:  # 第一条为重点推荐
                markdown += f"- **{step['spring_name']}（重点推荐）**\n"
            else:  # 其余为普通推荐，没有括号内容
                markdown += f"- **{step['spring_name']}**\n"
            markdown += f"  - **时间建议**: {step['suggested_time']}\n"
            markdown += f"  - **疗愈效果**: {step['effect']}\n\n"

        markdown += "###### &nbsp;\n"
        # 添加温馨提示部分
        markdown += f"**智小伴建议**：{self.NOTICE}\n\n"

        return markdown

    async def get_bath_plan(self, combination_key, gender, spring_effects_data, bath_plan_data):
        """
        获取泡浴方案：如果没有选择健康问题，返回默认方案；否则，返回自定义方案。
        """
        # 如果是没有选择健康问题的情况，获取默认方案
        if combination_key == "未选择问题":
            return await self.default_bath_plan(gender, spring_effects_data, bath_plan_data, version="v_1.1.0")

        # 如果有健康问题，查找匹配的方案
        if combination_key in bath_plan_data:
            bath_plan_string = bath_plan_data[combination_key]
            full_plan = await self.parse_bath_plan(bath_plan_string, spring_effects_data, gender, is_default=False)
            health_analysis = await self.get_health_analysis(combination_key, gender)
            return full_plan, health_analysis

        # 如果没有找到匹配的方案
        return [], {"msg": "未找到匹配的泡浴方案"}

    async def generate_bath_plan_v1_1_0(self, user_data):
        """
        根据用户输入的健康问题生成泡浴方案
        """
        bath_plan_data, spring_effects_data = await self.load_bath_plan_data()
        gender = user_data.get('gender', "男")

        # 收集用户的健康问题
        user_problems = []
        if user_data.get('skin_problems'):
            user_problems.append('皮肤问题')
        if user_data.get('pain_problems'):
            user_problems.append('疼痛问题')
        if user_data.get('fatigue_problems'):
            user_problems.append('疲劳问题')
        if user_data.get('sleep_problems'):
            user_problems.append('睡眠问题')

        # 根据健康问题选择组合的key
        combination_key = '+'.join(user_problems) if user_problems else "未选择问题"

        # 获取泡浴方案
        full_plan, health_analysis = await self.get_bath_plan(combination_key, gender, spring_effects_data,
                                                              bath_plan_data)

        # 生成 Markdown 格式
        markdown_output = await self.generate_markdown_bath_plan(full_plan, health_analysis)

        # 转换为前端需要的结构化格式
        frontend_contents = await wrap_content_for_frontend(markdown_output, content_type="MARKDOWN")

        business_category = list(
            set(
                article.get("business_category", "")
                for article in self.data.get("tweet_articles", [])
                if article.get("category") in ["温泉", "酒店"]
            )
        )
        if not business_category:
            business_category = ["温泉", "酒店"]  # 默认值

        return {
                "plan": full_plan,
                "contents": frontend_contents,
                "cates": business_category,
                "plan_text": markdown_output,
                "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                "output_basis": "根据健康问卷测评结果，结合温泉功效特色，从整体有效性角度为您推荐泡浴方案，仅供参考",
                "health_analysis": health_analysis
            }


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
    # print(recommended_bath_plan)