from fastapi import FastAPI, Request
from typing import List, Optional
import random
from datetime import datetime, timedelta
import pandas as pd
import re
from src.utils.resources import InitAllResource
from src.utils.database import MysqlConnector
from typing import Generator
from src.utils.Logger import logger




app = FastAPI()


class ItineraryModel:
    def __init__(self, gsr: InitAllResource) -> None:
        # 初始化实例属性
        self.gsr = gsr
        self.mysql_conn = MysqlConnector(**self.gsr.mysql_config)
        self.data = self.load_data()

    def load_data(self):
        """
        从数据库中加载9张表格的数据并存储在类属性中
        """
        # 使用统一的SQL查询加载所有表的数据
        tables = [
            "cleaned_accommodation", "cleaned_activities", "cleaned_agricultural_products",
            "cleaned_agricultural_services", "cleaned_dining", "cleaned_health_projects",
            "cleaned_packages", "cleaned_secondary_products", "cleaned_study_tour_products"
        ]
        data = {table: self.mysql_conn.query(f"select * from {table}") for table in tables}
        return data

    def is_within_time_range(self, service_time, best_time, duration, opening_hours):
        """
        检查活动的时间段和时长是否在用户的服务时间范围内，并根据活动开放时间筛选
        :param service_time: 用户输入的服务时间范围
        :param best_time: 活动的最佳时段
        :param duration: 活动的时长
        :param opening_hours: 活动的开放时间 (weekday_opening_hours, saturday_opening_hours, sunday_opening_hours)
        :return: 是否符合时间要求
        """
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

        # 解析活动的开放时间段
        def get_open_hours(day_of_week):
            if day_of_week < 5:  # 周一到周五
                return opening_hours.get("weekday_opening_hours")
            elif day_of_week == 5:  # 周六
                return opening_hours.get("saturday_opening_hours")
            else:  # 周日
                return opening_hours.get("sunday_opening_hours")

        # 检查多个日期范围内的开放时间
        current_date = start_date
        while current_date <= end_date:
            day_of_week = current_date.weekday()
            open_hours = get_open_hours(day_of_week)

            if open_hours:
                open_start_str, open_end_str = open_hours.split('-')
                open_start_time = datetime.strptime(open_start_str, "%H:%M")
                open_end_time = datetime.strptime(open_end_str, "%H:%M")

                # 处理活动时长
                if '-' in duration:
                    min_duration = float(duration.split('-')[0])  # 最短时长
                else:
                    min_duration = float(duration)  # 单一时长

                # 检查活动的时段 (best_time)，例如 "12:30-15:00" 或 "12:30,15:00"
                time_slots = best_time.split(',')
                for time_slot in time_slots:
                    # 如果是范围，如 "12:30-15:00"
                    if '-' in time_slot:
                        slot_start_str, slot_end_str = time_slot.split('-')
                        slot_start_time = datetime.strptime(slot_start_str, "%H:%M")
                        slot_end_time = datetime.strptime(slot_end_str, "%H:%M")
                    else:
                        # 单个时间点，加上活动时长作为结束时间
                        slot_start_time = datetime.strptime(time_slot, "%H:%M")
                        slot_end_time = slot_start_time + timedelta(hours=min_duration)

                    # 检查活动时间是否在开放时间内，并且时间段是否在用户的服务时间范围内
                    if open_start_time <= slot_start_time <= open_end_time and open_start_time <= slot_end_time <= open_end_time:
                        return True  # 如果某天的时段符合开放时间，返回 True

            # 移动到下一天
            current_date += timedelta(days=1)

        # 如果没有任何一天符合时间段，返回 False
        return False

    def matches_season(self, service_time, suitable_season):
        """
        检查活动是否适合当前季节或服务时间
        :param service_time: 用户输入的服务时间范围
        :param suitable_season: 活动适合的季节（例如：“春季、夏季”）
        :return: 是否符合季节要求
        """
        # 将用户输入的开始和结束时间转换为月份
        start_month = datetime.strptime(service_time["start_date"], "%Y-%m-%d").month
        end_month = datetime.strptime(service_time["end_date"], "%Y-%m-%d").month

        # 定义季节与月份的映射关系
        season_mapping = {
            "春季": [3, 4, 5],
            "夏季": [6, 7, 8],
            "秋季": [9, 10, 11],
            "冬季": [12, 1, 2]
        }

        # 如果活动适合全年，则直接返回 True
        if suitable_season == "全年":
            return True

        # 将活动适合的季节用顿号拆分为列表
        suitable_seasons = suitable_season.split("、")

        # 获取活动适合的月份集合
        suitable_months = set()
        for season in suitable_seasons:
            suitable_months.update(season_mapping.get(season, []))

        # 检查用户服务时间的月份范围是否与活动的适合月份有重合
        for month in range(start_month, end_month + 1):
            if month in suitable_months:
                return True

        # 如果没有任何重合，返回 False
        return False

    def filter_activities(self, user_data):
        """
        筛选符合用户偏好、年龄段、预算、时间和季节的活动
        :param user_data: 用户输入的数据，包括偏好、出行人员和预算
        :return: 符合条件的活动列表
        """
        activities = self.data["cleaned_activities"]
        filtered_activities = []

        preferences = user_data.get("service_preference", [])
        travelers = user_data.get("travelers", [])
        budget = user_data.get("budget", 0)
        service_time = user_data.get("service_time", {})

        # 1. 进行偏好、价格、季节、时间的初步筛选
        for activity in activities:
            activity_preferences = activity["preference"].split('、')
            applicable_people = activity["applicable_people"]

            # 1.1 筛选偏好匹配
            if not any(pref in activity_preferences for pref in preferences):
                continue  # 如果偏好不匹配，跳过该活动

            # 1.2 获取活动价格
            activity_price = self.get_activity_price(activity, service_time)
            if activity_price is not None and activity_price > budget:
                continue  # 如果价格超过预算，跳过该活动

            # 1.3 检查季节是否符合
            if not self.matches_season(service_time, activity["suitable_season"]):
                continue  # 如果季节不符合，跳过该活动

            # 1.4 筛选时间和时长
            if not self.is_within_time_range(service_time, activity["best_time"], activity["duration"]):
                continue  # 如果时间和时长不符合，跳过该活动

            # 2. 检查适用人群
            all_people_fit = True
            for traveler in travelers:
                traveler_age_group = traveler["age_group"]
                traveler_gender = traveler["gender"]

                # 检查每一个出行人员是否符合该活动的适用人群
                if not any(
                        person["age_group"] == traveler_age_group and
                        (person["gender"] == "不限" or person["gender"] == traveler_gender)
                        for person in applicable_people
                ):
                    all_people_fit = False
                    break  # 如果有一个人不符合，跳出循环

            if not all_people_fit:
                continue  # 如果有不符合的出行人员，跳过该活动

            # 如果通过所有条件筛选，将活动添加到结果列表
            filtered_activities.append(activity)

        return filtered_activities

    def get_activity_price(self, activity, service_time):
        """
        根据活动的时间和套餐获取活动的价格（考虑跨越多个日期）
        :param activity: 活动数据
        :param service_time: 用户的服务时间（包含多个日期）
        :return: 活动的总价格
        """
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

        total_price = 0  # 用于累加跨越多日的价格
        current_date = start_date

        while current_date <= end_date:
            # 1. 判断当天是否免费
            if activity["price_info"]["is_free"]:
                # 免费的活动视为该天价格为0
                day_price = 0
            else:
                # 2. 判断当前日期是工作日还是周末
                if current_date.weekday() < 5:  # 周一到周五是工作日
                    day_prices = activity["price_info"].get("weekday", {})
                else:  # 周末
                    day_prices = activity["price_info"].get("weekend", {})

                # 3. 获取有无餐的价格，优先选择包含餐的价格
                price_with_meal = day_prices.get("with_meal")
                price_without_meal = day_prices.get("without_meal")
                day_price = price_with_meal if price_with_meal is not None else price_without_meal

                # 如果当天的价格不存在（比如周末或工作日都没有设定），跳过该天
                if day_price is None:
                    day_price = 0

            # 累加每一天的价格
            total_price += day_price
            current_date += timedelta(days=1)

        return total_price if total_price > 0 else None  # 如果没有价格或活动全程免费，返回None或0

    def generate(self, user_data):
        """
        根据用户数据生成行程清单
        :param user_data: 用户的输入数据，包括偏好、需求等
        :return: 行程清单的响应字典
        """
        filtered_activities = self.filter_activities(user_data)

        # 可以继续补充对其他表格（如住宿、餐饮等）的筛选逻辑
        # 这里我们先占位
        selected_accommodation = None  # placeholder for accommodation selection
        selected_dining = None  # placeholder for dining selection

        # 创建行程的逻辑（简化版）
        itinerary = []
        for i, activity in enumerate(filtered_activities):
            itinerary.append({
                "day": i + 1,
                "date": f"2024-10-{27 + i}",
                "time_slots": [
                    {
                        "period": "上午",
                        "activities": [
                            {
                                "name": activity["activity_name"],
                                "location": activity["activity_type"],
                                "extra_info": {
                                    "description": activity["description"],
                                    "operation_tips": activity.get("reservation_note", "无"),
                                    "activity_link": "待定"
                                }
                            }
                        ]
                    }
                ]
            })

        return {
            "head": 200,
            "items": {
                "hotel": {
                    "name": "汤泉逸墅",
                    "extra_info": {
                        "location": "具体地址信息待定",
                        "description": "房型丰富、设施齐全、中医理疗特色",
                        "link": "待定"
                    }
                },
                "recommendation_basis": "推荐内容基于用户输入的健康状况、偏好及其他条件，匹配合适的酒店和行程方案。",
                "itinerary": itinerary,
                "msg": "行程生成成功"
            }
        }


    def get_time_slots(self, day):
      """
      获取示例活动（根据天数写死的简单活动）
      :param day: 当前是第几天
      :return: 活动列表
      """
      if day == 1:
        return [
          {
            "period": "下午",
            "activities": [
              {"name": "办理入住", "location": "汤泉逸墅", "extra_info": {"note": "需提前预约"}},
              {"name": "温泉疗愈", "location": "汤泉逸墅", "extra_info": {"description": "放松身心"}}
            ]
          }
        ]
      elif day == 2:
        return [
          {
            "period": "中午",
            "activities": [
              {"name": "岩洞氧吧体验", "location": "岩洞氧吧", "extra_info": {"description": "舒缓压力"}}
            ]
          },
          {
            "period": "下午",
            "activities": [
              {"name": "果蔬采摘", "location": "来康郡庄园", "extra_info": {"description": "体验乡村劳作"}},
              {"name": "劳作体验", "location": "来康郡庄园", "extra_info": {"description": "农业体验"}}
            ]
          }
        ]
      else:
        return [{"period": "全天", "activities": [{"name": "自由活动", "location": "酒店", "extra_info": {}}]}]

    # 读取温泉方案 Excel 文件
    def load_bath_plan_data(self):
        # 文件路径
        file_path = 'doc/bath_plan/温泉方案_updated.xlsx'  # 更新为你的文件路径

        # 读取泡浴方案和温泉作用
        df_bath_plan = pd.read_excel(file_path, sheet_name='温泉方案')
        df_spring_effects = pd.read_excel(file_path, sheet_name='温泉作用')

        # 创建一个字典来存储问题组合和方案
        bath_plan_data = {}
        # 创建一个字典来存储温泉作用
        spring_effects_data = {}

        # 遍历温泉作用数据，存储每个温泉的作用
        for index, row in df_spring_effects.iterrows():
            spring_name = row['分类']  # 温泉名称
            effect = row['作用']  # 温泉的作用
            if pd.notna(spring_name) and pd.notna(effect):
                spring_effects_data[spring_name] = effect

        # 遍历泡浴方案数据，存储每个问题组合对应的泡浴方案
        for index, row in df_bath_plan.iterrows():
            question_combination = row['问题组合']  # 确保 Excel 中这一列是问题组合
            bath_plan = row['泡浴方案']  # 确保 Excel 中这一列是泡浴方案

            # 只处理非空组合和方案
            if pd.notna(question_combination) and pd.notna(bath_plan):
                bath_plan_data[question_combination] = bath_plan

        return bath_plan_data, spring_effects_data

    def parse_bath_plan(self, bath_plan_string, spring_effects_data):
        # 固定主池和副池的时间
        main_pool_time = "15-20分钟"
        secondary_pool_time = "10-15分钟"

        # 使用正则表达式提取温泉名和主池标记
        bath_plan = []
        pattern = r'(\w+泉)(?:（(主池)）)?'
        matches = re.findall(pattern, bath_plan_string)

        # 遍历匹配到的温泉名和池类型
        for match in matches:
            spring_name = match[0]
            pool_type = "主池" if match[1] == '主池' else "副池"

            # 设置固定的时间
            suggested_time = main_pool_time if pool_type == "主池" else secondary_pool_time

            # 获取温泉的作用
            effect = spring_effects_data.get(spring_name, "未知作用")

            # 构建温泉对象，包含作用
            bath_plan.append({
                "spring_name": spring_name,
                "pool_type": pool_type,
                "suggested_time": suggested_time,
                "effect": effect
            })

        return bath_plan

    def generate_bath_plan(self, user_data):
        bath_plan_data, spring_effects_data = self.load_bath_plan_data()

        full_plan = []
        user_problems = []

        # 根据用户健康问题生成组合
        if user_data.get('skin_problems'):
            user_problems.append('皮肤问题')

        if user_data.get('pain_problems'):
            user_problems.append('疼痛问题')

        if user_data.get('fatigue_problems'):
            user_problems.append('疲劳问题')

        if user_data.get('sleep_problems'):
            user_problems.append('睡眠问题')

        # 生成问题组合的键
        if len(user_problems) == 1:
            combination_key = user_problems[0]
        elif len(user_problems) > 1:
            combination_key = '+'.join(user_problems)
        else:
            return {"msg": "无健康问题数据"}

        # 根据组合键查找方案
        if combination_key in bath_plan_data:
            bath_plan_string = bath_plan_data[combination_key]
            # 解析泡浴方案字符串并关联温泉作用
            full_plan = self.parse_bath_plan(bath_plan_string, spring_effects_data)
        else:
            return {"msg": "未找到匹配的泡浴方案"}

        return {
            "head": 200,
            "item": {
                "bath_plan": full_plan,
                "notice": "建议累计泡浴时长40-50分钟/次，避免长时间泡浴导致疲劳、低血糖等不适",
                "output_basis": "根据您的问卷结果，结合特色温泉的不同功效，从整体有效性、合理性、提升效果的角度为您推荐以下泡浴方案，仅供参考",
                "health_analysis": "待定"
            },
            "msg": ""
        }


if __name__ == '__main__':
    # 测试输入数据
    input_data = {
        "service_theme": "旅游度假",
        "service_time": {
            "start_date": "2024-10-27",
            "end_date": "2024-10-28"
        },
        "travelers": [
            {"age_group": "adult", "count": 2},
            {"age_group": "elderly", "count": 1}
        ],
        "service_preference": ["休闲", "文化"],
        "remarks": "行程不要太紧凑，喜欢温泉和文化体验",
        "preferred_package": "豪华家庭套餐",
        "budget_range": 1000
    }

    # 实例化 ItineraryGenerator 类并生成行程
    gsr = InitAllResource()
    generator = ItineraryModel(gsr)
    recommended_itinerary = generator.generate(input_data)
