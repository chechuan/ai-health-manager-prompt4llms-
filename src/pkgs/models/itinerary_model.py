# -*- encoding: utf-8 -*-
"""
@Time    :   2024-10-22 10:01:51
@desc    :   行程推荐功能
@Author  :   车川
@Contact :   1163317515@qq.com
"""

from fastapi import FastAPI, Request
from datetime import datetime, timedelta
from src.utils.resources import InitAllResource
from src.utils.database import MysqlConnector
import math
import random
import json
from typing import Generator
from src.prompt.model_init import acallLLM, callLikangLLM
from src.utils.Logger import logger
from src.utils.api_protocal import *
from src.utils.module import (run_in_executor, wrap_content_for_frontend, parse_generic_content,
                              assemble_frontend_format_with_fixed_items, extract_clean_output)
import asyncio
import json5

app = FastAPI()


class ItineraryModel:
    def __init__(self, gsr: InitAllResource) -> None:
        """
        初始化行程推荐模型
        :param gsr: 全局资源对象，包含数据库配置等
        """
        self.gsr = gsr
        self.mysql_conn = MysqlConnector(**self.gsr.mysql_config)
        self.data = self.load_data()
        self.regist_aigc_functions()

    def load_data(self):
        """
        从数据库中加载行程推荐相关的表格数据，存储在类属性中。
        :return: 包含各表数据的数据字典
        """
        logger.info("正在从数据库加载行程推荐数据...")
        tables = [
            "cleaned_accommodation", "cleaned_activities", "tweet_articles"
            # "cleaned_agricultural_products",
            # "cleaned_agricultural_services", "cleaned_dining", "cleaned_health_projects",
            # "cleaned_packages", "cleaned_secondary_products", "cleaned_study_tour_products"
        ]
        data = {table: self.mysql_conn.query(f"select * from {table}") for table in tables}
        logger.info("行程推荐数据加载成功。")
        return data

    async def filter_data(self, table_name, user_data):
        """
        通用数据过滤入口，根据表名分发到不同的处理函数
        :param table_name: 要过滤的表名
        :param user_data: 用户输入的数据
        :return: 筛选结果
        """
        logger.info(f"开始筛选表 {table_name} 中的数据")


        if table_name == "cleaned_activities":
            return await self.filter_activities(user_data, self.data.get("cleaned_activities", []))
        elif table_name == "cleaned_accommodation":
            return await self.filter_accommodation(user_data, self.data.get("cleaned_accommodation", []))
        else:
            logger.warning(f"未知的表名：{table_name}")
            return []

    def get_accommodation_price(self, price_info, service_time):
        """
        计算住宿的总价格，按天计算预算
        :param price_info: 住宿的价格信息
        :param service_time: 用户的服务时间（包含多个日期）
        :return: 住宿的总价格
        """
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

        total_price = 0  # 用于累加跨越多日的价格
        current_date = start_date
        while current_date <= end_date:
            # 根据是周末还是工作日，决定当天的价格
            if current_date.weekday() >= 5:  # 周六和周日
                day_price = price_info.get("weekend", {}).get("withMeal", 0)
            else:  # 周一到周五
                day_price = price_info.get("weekday", {}).get("withMeal", 0)

            total_price += day_price
            current_date += timedelta(days=1)

        return total_price if total_price > 0 else None  # 返回总价或None

    async def filter_accommodation(self, user_data, accommodations):
        """
        根据用户输入筛选符合条件的住宿
        :param user_data: 用户输入的数据，包括出行人员和预算
        :return: 符合条件的住宿列表
        """
        logger.info("根据用户输入筛选住宿。")
        # 确保传入的 `activities` 是列表
        if not isinstance(accommodations, list):
            raise ValueError("Expected 'accommodations' to be a list.")
        accommodations = accommodations[:6]
        filtered_accommodations = []

        travelers = user_data.get("travelers", [])
        budget = user_data.get("budget", 0)
        service_time = user_data.get("service_time", {})

        # 1. 根据适用人群和预算进行筛选
        for accommodation in accommodations:
            applicable_people = json.loads(accommodation.get("applicable_people", []))
            price_info = json.loads(accommodation.get("price_info", {}))  # 解析价格字段

            # 1.1 筛选适用人群是否匹配
            if not self.is_applicable_for_travelers(travelers, applicable_people):
                # logger.info(f"住宿 {accommodation['name']} 不符合适用人群。")
                continue

            # # 1.2 计算住宿的总价格并筛选
            # accommodation_price = self.get_accommodation_price(price_info, service_time)
            # if accommodation_price is not None and accommodation_price > budget:
            #     logger.debug(f"由于预算超出，跳过住宿 {accommodation['name']}")
            #     continue

            # 如果通过所有条件筛选，将住宿添加到结果列表
            filtered_accommodations.append(accommodation)
            # logger.info(f"住宿 {accommodation['name']} 符合用户条件。")

        return filtered_accommodations

    async def filter_activities(self, user_data, activities):
        """
        筛选符合用户偏好、年龄段、预算、时间和季节的活动
        :param user_data: 用户输入的数据，包括偏好、出行人员和预算
        :return: 符合条件的活动列表
        """
        logger.info("根据用户输入筛选活动。")
        # 确保传入的 `activities` 是列表
        if not isinstance(activities, list):
            raise ValueError("Expected 'activities' to be a list.")

        activities = activities[:18]
        filtered_activities = []

        preferences = user_data.get("service_preference", [])
        travelers = user_data.get("travelers", [])
        budget = user_data.get("budget", 0)
        service_time = user_data.get("service_time", {})

        # 1. 进行偏好、价格、季节、时间的初步筛选
        for activity in activities:
            activity_preferences = activity["preference"].split('、')
            applicable_people = json.loads(activity.get("applicable_people", []))
            price_info = json.loads(activity.get("price_info", []))  # 将字符串转为字典

            # 1.1 筛选偏好匹配
            if not any(pref in activity_preferences for pref in preferences):
                # logger.info(f"活动 {activity['activity_name']} 不符合用户偏好。")
                continue

            # # 1.2 获取活动价格并筛选
            # activity_price = self.get_activity_price(price_info, service_time, activity["duration"])
            # if activity_price is not None and activity_price > budget:
            #     logger.debug(f"由于预算超出，跳过活动 {activity['activity_name']}")
            #     continue

            # 1.3 检查季节是否符合
            if not self.matches_season(service_time, activity["suitable_season"]):
                # logger.info(f"活动 {activity['activity_name']} 不适合当前季节。")
                continue

            start_date = datetime.strptime(service_time.get("start_date"), "%Y-%m-%d")
            end_date = datetime.strptime(service_time.get("end_date"), "%Y-%m-%d")
            best_time = activity.get("best_time", "")
            reservation_days = self.normalize_value(activity.get("reservation_days"))
            opening_hours = {
                "weekday": activity.get("weekday_opening_hours", None),
                "saturday": activity.get("saturday_opening_hours", None),
                "sunday": activity.get("sunday_opening_hours", None)
            }

            # 2. 检查是否在用户时间范围内开放
            if not self.is_open_in_user_range(start_date, end_date, opening_hours):
                # logger.debug(f"由于开放时间不符，跳过活动 {activity['activity_name']}")
                continue

            # 3. 检查最佳时段是否在开放时间内
            if not self.is_best_time_available(best_time, opening_hours):
                # logger.debug(f"由于最佳时段不在开放时间内，跳过活动 {activity['activity_name']}")
                continue

            # 4. 检查是否满足预约要求
            if not self.check_reservation_requirements(reservation_days, start_date):
                # logger.debug(f"由于预约要求不满足，跳过活动 {activity['activity_name']}")
                continue

            # 5. 检查适用人群是否匹配
            if not self.is_applicable_for_travelers(travelers, applicable_people):
                # logger.debug(f"由于适用人群不匹配，跳过活动 {activity['activity_name']}")
                continue

            # 如果通过所有条件筛选，将活动添加到结果列表
            filtered_activities.append(activity)
            # logger.info(f"活动 {activity['activity_name']} 符合用户偏好和条件。")

        return filtered_activities

    def matches_season(self, service_time, suitable_season):
        """
        检查活动是否适合当前季节或服务时间
        :param service_time: 用户输入的服务时间范围
        :param suitable_season: 活动适合的季节（例如：“春季、夏季”）
        :return: 是否符合季节要求
        """
        start_month = datetime.strptime(service_time.get("start_date"), "%Y-%m-%d").month
        end_month = datetime.strptime(service_time.get("end_date"), "%Y-%m-%d").month

        season_mapping = {
            "春季": [3, 4, 5],
            "夏季": [6, 7, 8],
            "秋季": [9, 10, 11],
            "冬季": [12, 1, 2]
        }

        if suitable_season == "全年":
            return True

        suitable_seasons = suitable_season.split("、")
        suitable_months = {month for season in suitable_seasons for month in season_mapping.get(season, [])}

        logger.debug(f"检查服务月份 ({start_month}-{end_month}) 是否匹配活动适合月份。")
        return any(month in suitable_months for month in range(start_month, end_month + 1))

    def is_applicable_for_travelers(self, travelers, applicable_people):
        """
        检查出行人员是否符合活动的适用人群
        :param travelers: 出行人员列表
        :param applicable_people: 活动的适用人群，包含每个人群的 age_group, gender 和 required 字段
        :return: 如果出行人员符合条件，返回True；否则返回False
        """
        # 定义英文到中文的年龄组映射
        age_group_translation = {
            "elderly": "老人",
            "adult": "成人",
            "teenager": "青年",
            "children": "儿童"
        }

        # 提取适用人群中的年龄组和必需人群
        required_age_groups = {person.get("ageGroup") for person in applicable_people if person.get("required", False)}
        applicable_age_groups = {person.get("ageGroup") for person in applicable_people}

        # 初始化字典以跟踪出行人员的年龄组
        travelers_age_groups = {age_group: False for age_group in applicable_age_groups}

        # 遍历出行人员，检查是否满足所需的每个年龄组要求
        for traveler in travelers:
            traveler_age_group_en = traveler.get("age_group", "")
            traveler_age_group = age_group_translation.get(traveler_age_group_en, "")
            count = traveler.get("count", 0)

            # 如果count为0，跳过该出行人员
            if count == 0:
                continue

            # 标记该年龄组是否包含在出行人员中
            if traveler_age_group in travelers_age_groups:
                travelers_age_groups[traveler_age_group] = True

        # 检查是否包含所有必需的年龄组
        if not all(travelers_age_groups[age] for age in required_age_groups):
            return False  # 必需的年龄组没有全部满足

        # 确保出行人员中没有不在适用人群中的年龄组
        for traveler in travelers:
            traveler_age_group_en = traveler.get("age_group", "")
            traveler_age_group = age_group_translation.get(traveler_age_group_en, "")
            count = traveler.get("count", 0)

            # 如果count为0，跳过该出行人员
            if count == 0:
                continue

            # 如果出行人员包含不在适用人群中的年龄组，则不符合条件
            if traveler_age_group not in applicable_age_groups:
                return False

        return True

    def normalize_value(self, value):
        """
        通用方法，将 NaN、None 或无效值统一处理为 None
        :param value: 任意数据值，可能为 NaN 或其他类型
        :return: 如果值为 NaN 或 None，返回 None；否则返回原值
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value

    def is_open_in_user_range(self, start_date, end_date, opening_hours):
        """
        检查活动是否在用户提供的时间范围内开放
        :param start_date: 用户行程开始日期
        :param end_date: 用户行程结束日期
        :param opening_hours: 活动的开放时间
        :return: 是否在时间范围内开放
        """
        current_date = start_date

        while current_date <= end_date:
            day_of_week = current_date.weekday()

            open_hours = opening_hours.get("weekday" if day_of_week < 5 else "saturday" if day_of_week == 5 else "sunday")
            if not open_hours:
                return False
            current_date += timedelta(days=1)
        return True

    def is_best_time_available(self, best_time, opening_hours):
        """
        检查活动的最佳时段是否在开放时间内
        :param best_time: 活动的最佳时段
        :param opening_hours: 活动的开放时间
        :return: 是否符合最佳时段
        """
        if not best_time:
            return True

        time_slots = best_time.split(',')
        for time_slot in time_slots:
            if '-' in time_slot:
                slot_start, slot_end = time_slot.split('-')
                slot_start_time = datetime.strptime(slot_start, "%H:%M")
                slot_end_time = datetime.strptime(slot_end, "%H:%M")
                if self.is_within_opening_hours(slot_start_time, slot_end_time, opening_hours):
                    return True
        return False

    def is_within_opening_hours(self, slot_start_time, slot_end_time, opening_hours):
        """
        检查给定的时间段是否与活动的开放时间有重叠
        :param slot_start_time: 活动开始时间
        :param slot_end_time: 活动结束时间
        :param opening_hours: 活动的开放时间 (weekday_opening_hours, saturday_opening_hours, sunday_opening_hours)
        :return: 时间段是否在开放时间范围内
        """
        if slot_start_time.weekday() < 5:
            open_hours_str = opening_hours.get("weekday", None)
        elif slot_start_time.weekday() == 5:
            open_hours_str = opening_hours.get("saturday", None)
        else:
            open_hours_str = opening_hours.get("sunday", None)

        if open_hours_str:
            open_start_str, open_end_str = open_hours_str.split('-')
            open_start_time = datetime.strptime(open_start_str, "%H:%M")
            open_end_time = datetime.strptime(open_end_str, "%H:%M")

            # 检查活动的时间是否与开放时间有重叠
            if open_start_time <= slot_end_time and slot_start_time <= open_end_time:
                return True
        return False

    def check_reservation_requirements(self, reservation_days, start_date):
        """
        检查活动是否满足提前预约要求
        :param reservation_days: 预约要求的天数
        :param start_date: 用户行程开始日期
        :return: 是否符合预约要求
        """
        if reservation_days:
            today = datetime.today()
            delta_days = (start_date - today).days
            return delta_days >= reservation_days  # 如果在预约期限内，返回True
        return True  # 如果没有预约要求，返回True

    async def get_activity_price(self, price_info, service_time, duration):
        """
        根据活动的时间和套餐获取活动的价格（考虑跨越多个日期）
        :param price_info: 活动的价格信息
        :param service_time: 用户的服务时间（包含多个日期）
        :param duration: 活动的时长（小时）
        :return: 活动的总价格
        """

        async def calculate_price():
            start_date = datetime.strptime(service_time.get("start_date"), "%Y-%m-%d")
            end_date = datetime.strptime(service_time.get("end_date"), "%Y-%m-%d")

            total_price = 0  # 用于累加跨越多日的价格
            current_date = start_date

            while current_date <= end_date:
                # 判断当天是否免费
                if price_info.get("isFree", False):
                    day_price = 0
                else:
                    day_price = price_info.get("defaultValue", 0)

                    # 如果有按时长收费，计算总价
                    if price_info.get("unitPrice"):
                        unit_price = price_info.get("unitPrice").get("amount")
                        day_price += unit_price * float(duration)

                total_price += day_price
                current_date += timedelta(days=1)

            return total_price if total_price > 0 else None  # 返回总价或 None

        # 使用 asyncio.to_thread 将计算逻辑移到线程池中运行
        return await run_in_executor(calculate_price)

    async def generate_recommendation_basis(self, user_data, selected_accommodation, spa_activities):
        """
        根据用户输入和推荐结果生成推荐依据
        :param user_data: 用户输入的数据
        :param selected_accommodation: 选中的酒店
        :param spa_activities: 选中的温泉活动
        :return: 推荐依据的描述
        """
        # 用户偏好与年龄组的映射
        age_group_translation = {
            "elderly": "老人",
            "adult": "成人",
            "teenager": "青年",
            "children": "儿童"
        }

        # 根据用户的偏好生成推荐依据的基础描述
        preferences = user_data.get("service_preference", [])
        travelers = user_data.get("travelers", [])
        budget = user_data.get("budget", 0)

        basis_parts = []

        # 用户偏好描述
        if preferences:
            preferences_desc = "、".join(preferences)
            basis_parts.append(f"用户的偏好包括{preferences_desc}")
        else:
            basis_parts.append("用户喜欢多样化的活动")

        # 检查是否选择了温泉活动
        if spa_activities:
            basis_parts.append("并且喜欢温泉活动")

        # 检查用户的预算 (可选)
        # if budget > 0:
        #     basis_parts.append(f"推荐内容基于预算{budget}元")

        # 检查出行人员的类型并翻译
        if travelers:
            traveler_types = set([age_group_translation.get(traveler.get("age_group"), traveler.get("age_group"))
                                  for traveler in travelers if traveler.get("count", 0) > 0])
            traveler_types_desc = "、".join(traveler_types)
            basis_parts.append(f"适合出行人员为{traveler_types_desc}")
        else:
            basis_parts.append("适合全家出行")

        # 使用选中的酒店信息
        if selected_accommodation:
            hotel_name = selected_accommodation.get("name", "温泉酒店")
            basis_parts.append(f"推荐入住的酒店为{hotel_name}")

        # 拼接所有描述部分
        basis = "，".join(basis_parts) + "。"

        return basis

    def process_extra_info(self, activity):
        """
        将活动的 location、prescription 和 reservation_note（作为 operation_tips）添加到 extra_info 中。
        :param activity: 活动数据字典
        :return: 修改后的活动数据
        """
        # 构建新的活动结构
        activity_info = {
            "name": activity.get("activity_name", ""),
            "location": activity.get("location", ""),
            "activity_code": activity.get("activity_code", ""),
            "external_id": activity.get("external_id", ""),
            "activity_link": activity.get("activity_link", ""),
            "extra_info": {
                "description": activity.get("description", ""),
                "operation_tips": activity.get("reservation_note", "")  # 如果没有 reservation_note 默认是 "无"
            }
        }

        return activity_info

    async def generate_default_itinerary(self, user_data, selected_hotel=None):
        """
        根据用户的服务时间，生成通用的行程方案，适合大多数人的默认安排。
        所有活动都从现有的活动列表中选择，确保每一天都有安排。
        :param user_data: 用户输入的数据
        :return: 默认行程的列表
        """
        service_time = user_data.get("service_time", {})
        # 使用 get() 获取时间，避免可能的 KeyError
        start_date = datetime.strptime(service_time.get("start_date", ""), "%Y-%m-%d")
        end_date = datetime.strptime(service_time.get("end_date", ""), "%Y-%m-%d")

        total_days = (end_date - start_date).days + 1
        current_date = start_date
        itinerary = []

        # 如果没有提供酒店信息，使用默认的酒店
        if not selected_hotel:
            available_hotels = self.data.get("cleaned_accommodation", [])  # 假设从数据库加载的酒店数据
            selected_hotel = random.choice(available_hotels)

        # 现有的活动列表，用于随机选择活动
        available_activities = self.data.get('cleaned_activities', [])

        # 为每一天生成行程
        for day in range(1, total_days + 1):
            if day == 1:
                # 第一天下午安排温泉活动
                day_activities = [
                    {
                        "period": "下午",
                        "activities": [
                            {
                                "name": "办理入住",
                                "location": selected_hotel.get("name", "汤泉逸墅 院线房"),
                                "activity_code": selected_hotel.get("activity_code", "ACC992657"),
                                "external_id": selected_hotel.get("external_id", ""),
                                "activity_link": selected_hotel.get("activity_link", ""),
                                "extra_info": {
                                    # "description": selected_hotel.get("extra_info", {}).get("description", "请提前确认入住时间，提醒需要预约等"),
                                    "description": "请提前确认入住时间，提醒需要预约等",
                                    "room_description": selected_hotel.get("extra_info", {}).get("room_description",
                                                                                                 ""),
                                    "operation_tips": "请提前确认入住时间，提醒需要预约等"
                                }
                            },
                            self.process_extra_info(random.choice(available_activities))  # 随机选择一个活动

                        ]
                    }
                ]
            else:
                # 每天上午和下午安排两个活动，随机从现有的活动中选择
                day_activities = [
                    {
                        "period": "上午",
                        "activities": [self.process_extra_info(random.choice(available_activities))]
                    },
                    {
                        "period": "下午",
                        "activities": [self.process_extra_info(random.choice(available_activities))]
                    }
                ]

            itinerary.append({
                "day": day,
                "date": current_date.strftime("%Y-%m-%d"),
                "time_slots": day_activities
            })

            current_date += timedelta(days=1)

        return itinerary

    async def select_random_hotel(self, selected_accommodation):
        """
        随机选择一个符合条件的酒店
        :param selected_accommodation: 经过筛选的符合条件的住宿列表
        :return: 随机选择的酒店信息
        """
        if selected_accommodation:
            hotel = await self.select_random_activity(selected_accommodation)
            return {
                "name": hotel.get("name", ""),
                "location": hotel.get("name", ""),
                "activity_code": hotel.get("activity_code", ""),
                "external_id": hotel.get("external_id", ""),
                "activity_link": hotel.get("activity_link", ""),
                "extra_info": {
                    "description": hotel.get("hotel_description", "房型丰富、设施齐全、中医理疗特色"),
                    "room_description": hotel.get("room_description", "")
                }
            }

        return {
            "name": "无合适酒店",
            "location": "",
            "activity_code": "",
            "external_id": "",
            "activity_link": "",
            "extra_info": {
                "description": "无"
            }
        }

    async def select_random_activity(self, activities):
        return await asyncio.to_thread(random.choice, activities)

    async def create_itinerary(self, user_data, filtered_activities, spa_activities, selected_hotel):
        """
        创建用户的行程安排，确保每天都有活动，不允许出现空白天
        :param user_data: 用户输入的数据
        :param filtered_activities: 筛选后的活动列表
        :param spa_activities: 筛选出的温泉类活动
        :param selected_hotel: 随机选择的酒店信息
        :return: 构建好的行程列表
        """
        itinerary = []
        service_time = user_data.get("service_time", {})
        start_date = datetime.strptime(service_time.get("start_date", ""), "%Y-%m-%d")
        end_date = datetime.strptime(service_time.get("end_date", ""), "%Y-%m-%d")

        total_days = (end_date - start_date).days + 1
        current_date = start_date

        activity_index = 0
        max_spa_activities = 2  # 整个行程中最多2次温泉活动
        spa_activity_count = 0  # 记录温泉体验的数量
        recent_activities = []  # 用于跟踪最近的活动，防止重复

        def is_activity_recent(activity_name, recent_activities, min_gap_days=7):
            """
            检查活动是否最近已经安排过
            :param activity_name: 活动名称
            :param recent_activities: 最近活动的记录
            :param min_gap_days: 活动之间的最小间隔天数
            :return: 是否最近安排过
            """
            for recent_activity in recent_activities:
                if recent_activity.get("name") == activity_name and (
                        current_date - recent_activity.get("date")).days < min_gap_days:
                    return True
            return False

        # 为每一天生成行程
        for day in range(1, total_days + 1):
            day_activities = []
            daily_spa_activity_flag = False  # 每天是否安排了温泉活动

            if day == 1:
                # 第一日安排办理入住
                day_activities.append({
                    "period": "下午",
                    "activities": [
                        {
                            "name": "办理入住",
                            "location": selected_hotel.get("name", ""),
                            "activity_code": selected_hotel.get("activity_code", ""),
                            "external_id": selected_hotel.get("external_id", ""),
                            "activity_link": selected_hotel.get("activity_link"),
                            "extra_info": {
                                "description": "请提前确认入住时间，提醒需要预约等",
                                "room_description": selected_hotel["extra_info"].get("room_description", ""),
                                "operation_tips": "请提前确认入住时间，提醒需要预约等"
                            }
                        }
                    ]
                })

                # 如果有温泉活动并且温泉次数未达到上限，安排温泉活动
                if spa_activities and spa_activity_count < max_spa_activities:
                    random_spa_activity = await self.select_random_activity(spa_activities)
                    day_activities[-1]["activities"].append({
                        "name": random_spa_activity.get("activity_name", ""),
                        "location": random_spa_activity.get("location", ""),
                        "activity_code": random_spa_activity.get("activity_code", ""),
                        "external_id": random_spa_activity.get("external_id", ""),
                        "activity_link": random_spa_activity.get("activity_link", ""),
                        "extra_info": {
                            "description": random_spa_activity.get("description", ""),
                            "operation_tips": random_spa_activity.get("reservation_note", "")
                        }
                    })
                    spa_activity_count += 1

            else:
                # 安排其他天的活动，确保活动不重复
                available_periods = ["上午", "下午"]
                for period in available_periods:
                    if activity_index < len(filtered_activities):
                        # 避免重复安排活动
                        activity = filtered_activities[activity_index]
                        if not is_activity_recent(activity["activity_name"], recent_activities):
                            day_activities.append({
                                "period": period,
                                "activities": [
                                    {
                                        "name": activity.get("activity_name", ""),
                                        "location": activity.get("location", ""),
                                        "activity_code": activity.get("activity_code", ""),
                                        "external_id": activity.get("external_id", ""),
                                        "activity_link": activity.get("activity_link", ""),
                                        "extra_info": {
                                            "description": activity.get("description", ""),
                                            "operation_tips": activity.get("reservation_note", "无")
                                        }
                                    }
                                ]
                            })
                            recent_activities.append({"name": activity["activity_name"], "date": current_date})
                            activity_index += 1

                # 如果当日没有活动，随机安排一个非温泉类活动
                if not day_activities:
                    random_activity = await self.select_random_activity(filtered_activities)
                    day_activities.append({
                        "period": "上午",
                        "activities": [
                            {
                                "name": random_activity.get("activity_name", ""),
                                "location": random_activity.get("location", ""),
                                "activity_code": random_activity.get("activity_code", ""),
                                "external_id": random_activity.get("external_id", ""),
                                "activity_link": random_activity.get("activity_link", ""),
                                "extra_info": {
                                    "description": random_activity.get("description", ""),
                                    "operation_tips": random_activity.get("reservation_note", "无")
                                }
                            }
                        ]
                    })

            itinerary.append({
                "day": day,
                "date": current_date.strftime("%Y-%m-%d"),
                "time_slots": day_activities
            })

            current_date += timedelta(days=1)

        return itinerary

    async def generate_itinerary(self, user_data):
        """
        根据用户数据生成行程清单
        :param user_data: 用户输入的数据，包括偏好、需求等
        :return: 行程清单的响应字典
        """
        # 1. 筛选符合条件的活动
        filtered_activities = await self.filter_data("cleaned_activities", user_data)

        # 筛选出温泉类的活动
        spa_activities = [activity for activity in filtered_activities if
                          "泉" in activity.get("activity_name", "") or "泉" in activity.get("activity_category",
                                                                                                "")]
        # 2. 随机选择一个符合条件的酒店
        selected_accommodation = await self.filter_data("cleaned_accommodation", user_data)
        hotel = await self.select_random_hotel(selected_accommodation)

        # 3. 创建行程，将选中的酒店和温泉活动作为参数传递
        if not filtered_activities:
            itinerary = await self.generate_default_itinerary(user_data, hotel)
        else:
            itinerary = await self.create_itinerary(user_data, filtered_activities, spa_activities, hotel)
        # 4. 生成推荐依据
        recommendation_basis = await self.generate_recommendation_basis(user_data, hotel, spa_activities)

        # 5. 构建最终的响应结构
        response = {
            "hotel": {
                "name": hotel.get("name", ""),
                "location": hotel.get("name", ""),
                "extra_info": hotel.get("extra_info", {}),
                "activity_code": hotel.get("activity_code", ""),
                "external_id": hotel.get("external_id", ""),
                "activity_link": hotel.get("activity_link", "")
            },
            "recommendation_basis": recommendation_basis,
            "itinerary": itinerary,
            "msg": "行程生成成功"
        }

        return response

    async def format_itinerary_to_text(self, itinerary_data: list) -> str:
        """
        将行程数据格式化为指定的文本格式
        :param itinerary_data: JSON格式的行程数据列表
        :return: 格式化后的字符串
        """
        result = ""

        try:
            for day_entry in itinerary_data:
                # 确保day和date存在
                day = day_entry.get("day", "未知天数")
                date = day_entry.get("date", "未知日期")
                result += f"第{day}天（{date}）\n"

                time_slots = day_entry.get("time_slots", [])
                for time_slot in time_slots:
                    # 确保时间段存在
                    period = time_slot.get("period", "未知时间段")
                    activities = time_slot.get("activities", [])

                    result += f"{period}：\n"
                    for activity in activities:
                        # 确保活动信息存在
                        name = activity.get("name", "未知活动")
                        location = activity.get("location", "未知地点")
                        description = activity.get("description", "无描述")

                        result += f"{name} - {location}（{description}）\n"

                result += "\n"  # 每天之间空行

        except Exception as e:
            # 捕获所有异常并记录错误信息，返回一个安全的提示
            result += "\n行程数据格式化时发生错误，请检查输入数据。\n"
            result += f"错误详情：{str(e)}"

        return result.strip()  # 去除末尾的多余空行

    async def transform_itinerary(self, response_data):
        """
        转换行程数据结构为大模型所需的简化格式
        :param response_data: 原始响应数据
        :return: 转换后的行程数据列表
        """
        try:
            # 从响应中提取行程数据
            itinerary = response_data.get("itinerary", [])
            if not isinstance(itinerary, list):
                raise ValueError("Invalid itinerary data format: Expected a list.")

            simplified_itinerary = []

            # 遍历每一天的行程
            for day_entry in itinerary:
                # 提取基础信息
                day = day_entry.get("day")
                date = day_entry.get("date")
                time_slots = day_entry.get("time_slots", [])

                if not day or not date:
                    raise ValueError(f"Missing 'day' or 'date' in day entry: {day_entry}")

                transformed_day = {
                    "day": day,
                    "date": date,
                    "time_slots": []
                }

                # 遍历时间段
                for time_slot in time_slots:
                    period = time_slot.get("period")
                    activities = time_slot.get("activities", [])

                    if not period:
                        raise ValueError(f"Missing 'period' in time slot: {time_slot}")

                    transformed_slot = {
                        "period": period,
                        "activities": []
                    }

                    # 遍历活动
                    for activity in activities:
                        if not isinstance(activity, dict):
                            raise ValueError(f"Invalid activity format: Expected a dict, got {type(activity)}")

                        transformed_activity = {
                            "name": activity.get("name", None),
                            "location": activity.get("location", None),
                            "description": activity.get("extra_info", {}).get("description", None),
                            "external_id": activity.get("external_id", None),
                            "activity_link": activity.get("activity_link", None)
                        }
                        transformed_slot["activities"].append(transformed_activity)

                    transformed_day["time_slots"].append(transformed_slot)

                simplified_itinerary.append(transformed_day)

            return simplified_itinerary

        except Exception as e:
            # 日志记录异常，确保系统不崩溃
            import logging
            logging.error(f"Error transforming itinerary: {e}")
            # 返回空列表或其他合理的默认值
            return []

    async def __update_model_args__(self, kwargs, **args) -> Dict:
        if "model_args" in kwargs:
            if kwargs.get("model_args"):
                args = {
                    **args,
                    **kwargs["model_args"],
                }
            del kwargs["model_args"]
        return args

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

    def regist_aigc_functions(self) -> None:
        self.funcmap = {}
        for obj_str in dir(self):
            if obj_str.startswith("aigc_functions_") and not self.funcmap.get(obj_str):
                self.funcmap[obj_str] = getattr(self, obj_str)


    async def aigc_functions_itinerary_description(self, day_itinerary: dict, **kwargs) -> dict:
        """
        根据单天行程生成活动描述
        :param day_itinerary: 单天行程数据
        :return: 包含更新后的活动描述的单天行程数据
        """
        _event = kwargs.get("intentCode", "aigc_functions_itinerary_description")

        # 构建大模型参数
        prompt_vars = {"itinerary": day_itinerary}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用大模型
        content = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        try:
            # 解析生成的内容
            parsed_content = await parse_generic_content(content)
            return parsed_content
        except Exception as e:
            logger.error(f"活动描述生成失败: {str(e)}")
            return day_itinerary  # 返回原始数据，确保稳定性

    async def aigc_functions_itinerary_summary(self, itinerary_text: str, **kwargs) -> dict:
        """
        根据完整行程生成总结语和实用建议
        :param itinerary_text: 行程文本数据
        :return: 包含 intro、tips 和 closing 的字典
        """
        _event = kwargs.get("intentCode", "aigc_functions_itinerary_summary")

        # 构建大模型参数
        prompt_vars = {"itinerary": itinerary_text}
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )

        # 调用大模型
        content = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )

        try:
            # 尝试解析生成的内容
            return await parse_generic_content(content)
        except Exception as e:
            logger.error(f"解析行程总结失败: {str(e)}")
            raise  # 抛出异常，让上层函数处理

    # Helper Functions
    async def _filter_activities_and_hotel(self, user_data: dict):
        """
        筛选活动和酒店
        """
        filtered_activities = await self.filter_data("cleaned_activities", user_data)
        spa_activities = [
            activity for activity in filtered_activities if
            "温泉" in activity["activity_name"] or "温泉" in activity["activity_category"]
        ]
        selected_accommodation = await self.filter_data("cleaned_accommodation", user_data)
        hotel = await self.select_random_hotel(selected_accommodation)
        return filtered_activities, spa_activities, hotel

    async def _create_itinerary(self, user_data: dict, filtered_activities, spa_activities, hotel):
        """
        根据活动和酒店生成初步行程
        """
        if not filtered_activities:
            return await self.generate_default_itinerary(user_data, hotel)
        return await self.create_itinerary(user_data, filtered_activities, spa_activities, hotel)

    async def _generate_descriptions_and_summary(self, transformed_itinerary: list):
        """
        并发生成活动描述和行程总结
        :param transformed_itinerary: 标准化行程数据
        :return: 更新后的行程数据和总结
        """

        async def process_day(day_entry):
            """
            调用生成活动描述的方法
            """
            try:
                return await self.aigc_functions_itinerary_description(
                    day_entry, intentCode="aigc_functions_itinerary_description"
                )
            except Exception as e:
                logger.error(f"生成活动描述失败（第{day_entry.get('day', '未知天数')}天）：{e}")
                # 为该天活动生成默认描述
                for time_slot in day_entry.get("time_slots", []):
                    for activity in time_slot.get("activities", []):
                        activity["description"] = "无法生成个性化描述，请联系客服解决问题。"
                return day_entry

        async def generate_summary():
            """
            调用生成行程总结的方法
            """
            try:
                itinerary_text = await self.format_itinerary_to_text(transformed_itinerary)
                return await self.aigc_functions_itinerary_summary(
                    itinerary_text, intentCode="aigc_functions_itinerary_summary"
                )
            except Exception as e:
                logger.error(f"生成行程总结失败: {e}")
                # 返回默认值
                return {
                    "intro": "欢迎来到我们的行程体验！",
                    "tips": ["请携带合适衣物", "注意预约事项", "确保旅途中安全"],
                    "closing": "感谢选择我们的行程方案，祝您旅途愉快！"
                }

        # 并发处理活动描述和总结生成
        try:
            updated_itinerary, summary = await asyncio.gather(
                asyncio.gather(*(process_day(day) for day in transformed_itinerary)),
                generate_summary()
            )
            return updated_itinerary, summary
        except Exception as e:
            logger.error(f"并发处理活动描述和总结失败: {e}")
            raise

    async def generate_markdown_from_items(self, items: dict) -> str:
        """
        根据 items 数据生成 Markdown 文本
        :param items: 包含行程相关信息的字典
        :return: 生成的 Markdown 文本
        """
        markdown = []

        # 添加欢迎语
        intro = items.get("intro", "欢迎参加我们的行程！")
        markdown.append(f"{intro}\n ###### &nbsp;")  # 在欢迎语后添加空行

        # 添加每日行程
        itinerary = items.get("itinerary", [])
        for day in itinerary:
            day_number = day.get("day", "未知天数")
            date = day.get("date", "未知日期")
            markdown.append(f"### 第{day_number}天：")

            for time_slot in day.get("time_slots", []):
                period = time_slot.get("period", "未知时间段")
                markdown.append(f"- **{period}**：")

                for activity in time_slot.get("activities", []):
                    name = activity.get("name", "未知活动")
                    location = activity.get("location", "未知地点")
                    description = activity.get("description", "无描述")

                    # 特殊处理 "办理入住"
                    if name == "办理入住":
                        markdown.append(f"  - 到达来康都并办理入住，建议入住{location}，能够享受便捷的服务。")
                    else:
                        markdown.append(f"  - **{name}**：")
                        markdown.append(f"    - {description}")
                        markdown.append(f"    - 地点：{location}")

            # 检查并添加“返程”到最后一天最后一个时间段
            if day == itinerary[-1]:  # 检查是否为最后一天
                if time_slot == day.get("time_slots", [])[-1]:  # 检查是否为最后一个时间段
                    markdown.append("  - **返程**")

            markdown.append("")  # 空行分隔天数

        # 添加温馨提示
        tips = items.get("tips", [])
        if tips:
            markdown.append("### 温馨小贴士：")
            for tip in tips:
                markdown.append(f"- {tip}")
            markdown.append("")  # 空行分隔
            markdown.append("###### &nbsp;")  # 空行分隔

        # 添加总结
        closing = items.get("closing", "感谢参加我们的行程，祝您旅途愉快！")
        markdown.append(f"{closing}")

        # 合并所有内容
        return "\n".join(markdown)

    async def clean_itinerary(self, itinerary: list) -> list:
        """
        清理行程数据，确保日期格式和其他字段正确
        :param itinerary: 行程列表
        :return: 清理后的行程列表
        """
        for day_entry in itinerary:
            # 清理日期格式
            day_entry["date"] = day_entry["date"].replace(" ", "").strip()

            # # 其他字段的校验或清理可以在这里加入
            # for time_slot in day_entry.get("time_slots", []):
            #     for activity in time_slot.get("activities", []):
            #         activity["name"] = activity.get("name", "未知活动").strip()
            #         activity["location"] = activity.get("location", "未知地点").strip()
            #         activity["description"] = activity.get("description", "无描述").strip()

        return itinerary

    async def __compose_user_msg__(
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
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "ietary_guidelines":
            if ietary_guidelines:
                for key, value in ietary_guidelines.items():
                    if value and DIETARY_GUIDELINES_KEY_MAP.get(key):
                        content += f"{DIETARY_GUIDELINES_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "key_indicators":
            # 创建一个字典来存储按日期聚合的数据
            aggregated_data = {}

            # 遍历数据并聚合
            for item in key_indicators:
                for entry in item["data"]:
                    date, time = entry["time"].split(" ")
                    value = entry["value"]
                    if date not in aggregated_data:
                        aggregated_data[date] = {}
                    aggregated_data[date][item["key"]] = {"time": time, "value": value}

            # 创建 Markdown 表格
            content = "| 测量日期 | 测量时间 | 体重 | BMI | 体脂率 |\n"
            content += "| ------ | ------ | ---- | ----- | ------ |\n"

            # 填充表格
            for date, measurements in aggregated_data.items():
                time = measurements.get("体重", {}).get("time", "")
                weight = measurements.get("体重", {}).get("value", "")
                bmi = measurements.get("bmi", {}).get("value", "")
                body_fat_rate = measurements.get("体脂率", {}).get("value", "")
                row = f"| {date} | {time} | {weight} | {bmi} | {body_fat_rate} |\n"
                content += row
        else:
            logger.error(f"Compose user profile error: mode {mode} not supported")
        return content

    async def aigc_functions_likang_introduction(self, **kwargs) -> dict:
        """
        根据用户对话和背景信息生成固安来康郡的介绍内容
        :param kwargs: 包含会话记录(messages)和背景信息(background_info)的动态参数
        :return: 生成的介绍内容，返回标准化结构
        """
        # 获取事件码
        _event = kwargs.get("intentCode", "aigc_functions_likang_introduction")

        # 动态获取系统提示
        system_prompt_template = self.gsr.get_event_item(_event)["description"]
        if not system_prompt_template:
            raise ValueError(f"无法从事件 {_event} 获取有效的描述，请检查配置！")

        # 动态插入当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = system_prompt_template.format(datetime=current_time)

        # 组合用户消息
        messages = kwargs.get("messages", [])
        messages = [{"role": "system", "content": system_prompt}] + messages

        # 调试日志，记录完整消息内容
        logger.debug(f"生成固安来康郡介绍 LLM Input: {repr(messages)}")

        try:
            # 调用大模型直接生成内容
            content = await callLikangLLM(
                history=messages,
                model="Qwen2-72B-Instruct",
                temperature=0.7,
                top_p=0.7,
                repetition_penalty=1.0,
                stream=False
            )

            # 添加日志记录输出内容
            logger.debug(f"生成固安来康郡介绍 LLM Output: {content}")

        except Exception as e:
            logger.error(f"调用大模型时发生错误: {repr(e)}")
            raise

        # 处理业务分类
        business_category = list(
            set(
                article.get("business_category", "")
                for article in self.data.get("tweet_articles", [])
                if article.get("category") in ["温泉", "酒店"]
            )
        )
        if not business_category:
            business_category = ["温泉", "酒店"]  # 设置默认值

        # 清理生成的内容
        content = await extract_clean_output(content)

        # 格式化内容供前端使用
        frontend_contents = await wrap_content_for_frontend(content)

        # 构建返回结果
        res = {
                "contents": frontend_contents,
                "cates": business_category
            }

        return res

    async def generate_itinerary_v1_1_0(self, user_data: dict) -> dict:
        """
        根据用户数据生成完整行程，并包括活动描述和行程总结
        :param user_data: 用户输入的数据，包括偏好、需求等
        :return: 包含行程详情的响应字典
        """
        # 1. 筛选符合条件的活动和酒店
        filtered_activities, spa_activities, hotel = await self._filter_activities_and_hotel(user_data)

        # 2. 创建初步行程
        itinerary = await self._create_itinerary(user_data, filtered_activities, spa_activities, hotel)

        # 3. 标准化行程数据
        transformed_itinerary = await self.transform_itinerary({"itinerary": itinerary})

        # logger.debug(
        #     f"Transformed itinerary before processing: {json.dumps(transformed_itinerary, indent=4, ensure_ascii=False)}")

        # 4. 并发生成活动描述和总结
        updated_itinerary, summary = await self._generate_descriptions_and_summary(transformed_itinerary)
        # logger.debug(
        #     f"Transformed itinerary before processing: {json.dumps(updated_itinerary, indent=4, ensure_ascii=False)}")

        # 5. 对 updated_itinerary 进行清理和格式化
        updated_itinerary = await self.clean_itinerary(updated_itinerary)
        # logger.debug(
        #     f"Transformed itinerary before processing: {json.dumps(updated_itinerary, indent=4, ensure_ascii=False)}")

        # 6. 构建 overview 字典
        overview = {
            "intro": summary.get("intro", ""),
            "itinerary": updated_itinerary,
            "tips": summary.get("tips", []),
            "closing": summary.get("closing", ""),
        }

        # 7. 判断行程天数并生成附加数据
        if len(updated_itinerary) <= 3:
            # 如果行程小于等于三天，生成 Markdown
            markdown = await self.generate_markdown_from_items(overview)
            frontend_contents = await wrap_content_for_frontend(markdown)

        else:
            # 如果行程大于三天，生成 frontend_contents
            markdown = await self.generate_markdown_from_items(overview)
            frontend_contents = await assemble_frontend_format_with_fixed_items(overview)

        business_category = list(
            set(
                article.get("business_category", "")
                for article in self.data.get("tweet_articles", [])
                if article.get("category") in ["温泉", "酒店"]
            )
        )
        if not business_category:
            business_category = ["温泉", "酒店"]  # 默认值

        # 9. 构建完整的响应结构
        response = {
                "plan": overview,
                "contents": frontend_contents,
                "cates": business_category,
                "plan_text": markdown
            }
        return response

    async def aigc_functions_health_analysis_advice_generation(self, **kwargs) -> str:
        """
        生成健康分析与健康建议

        功能描述：
        根据用户画像和检测数据生成健康分析报告及生活建议，帮助用户理解健康状况并提供改善建议。
        """
        _event = "健康分析与健康建议生成"

        # 获取用户基本信息和检测数据
        user_profile = kwargs.get("user_profile", {})
        health_data = kwargs.get("health_data", [])

        # 修改空腹血糖参考范围和名称
        for item in health_data:
            if item.get("name") == "空腹血糖":
                item["name"] = "血糖"
                item["reference_range"] = "血糖过低：<3.9；正常：3.9~11.1；血糖过高>11.1"

        # 日志打印原始数据
        logger.info(f"User Profile: {user_profile}")
        logger.info(f"Raw Health Data: {health_data}")

        # 转换 HealthDataItem 对象为带中文键名的格式
        def map_health_data_to_chinese_format(health_data):
            """
            将健康数据映射为带中文键名的格式
            """
            mapped_data = []
            for item in health_data:
                # 构造映射后的数据
                mapped_item = {
                    "名称": item.get("name", ""),
                    "结果": item.get("value", ""),
                    "单位": item.get("unit", ""),
                    "参考范围": item.get("reference_range", ""),
                    "结果状态": item.get("status", "")
                }
                # 过滤掉完全空或仅包含空字符串的项目
                if all(value == "" for value in mapped_item.values()):
                    continue
                mapped_data.append({k: v for k, v in mapped_item.items() if v != ""})
            return mapped_data

        # 转换健康数据
        processed_health_data = map_health_data_to_chinese_format(health_data)

        # 日志记录转换后的健康数据
        logger.info(f"Processed Health Data: {processed_health_data}")

        # 构建健康分析与建议生成提示变量
        prompt_vars_for_health_analysis = {
            "user_profile": json.dumps(user_profile, ensure_ascii=False, indent=4),
            "health_data": json.dumps(processed_health_data, ensure_ascii=False, indent=4)
        }

        # 更新模型参数
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.5, repetition_penalty=1.0
        )

        # 调用AIGC函数生成健康分析与健康建议
        content: str = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars_for_health_analysis,
            model_args=model_args,
            **kwargs
        )

        # 打印生成的内容
        logger.info(f"Generated Health Analysis and Advice: {content}")

        return content


if __name__ == '__main__':
    # 测试输入数据
    input_data = {
        "service_theme": "旅游度假",
        "service_time": {
            "start_date": "2024-10-27",
            "end_date": "2024-10-28"
        },
        "travelers": [
            {"age_group": "成人", "count": 2},
            {"age_group": "老年", "count": 1}
        ],
        "service_preference": ["休闲", "文化"],
        "remarks": "行程不要太紧凑，喜欢温泉和文化体验",
        "preferred_package": "豪华家庭套餐",
        "budget_range": 1000
    }

    # 实例化 ItineraryModel 类并生成行程
    gsr = InitAllResource()
    generator = ItineraryModel(gsr)
    recommended_itinerary = generator.generate_itinerary(input_data)
    print(recommended_itinerary)






