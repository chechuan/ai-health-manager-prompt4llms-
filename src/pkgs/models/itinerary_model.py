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
from src.utils.Logger import logger
import json
import math
import random

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

    def load_data(self):
        """
        从数据库中加载多张表格的数据并存储在类属性中
        :return: 各表数据字典
        """
        logger.info("正在从数据库加载数据...")
        tables = [
            "cleaned_accommodation", "cleaned_activities",
            # "cleaned_agricultural_products",
            # "cleaned_agricultural_services", "cleaned_dining", "cleaned_health_projects",
            # "cleaned_packages", "cleaned_secondary_products", "cleaned_study_tour_products"
        ]
        data = {table: self.mysql_conn.query(f"select * from {table}") for table in tables}
        logger.info("所有表的数据加载成功。")
        return data

    def filter_data(self, table_name, user_data):
        """
        通用数据过滤入口，根据表名分发到不同的处理函数
        :param table_name: 要过滤的表名
        :param user_data: 用户输入的数据
        :return: 筛选结果
        """
        logger.info(f"开始筛选表 {table_name} 中的数据")

        if table_name == "cleaned_activities":
            return self.filter_activities(user_data)
        elif table_name == "cleaned_accommodation":
            return self.filter_accommodation(user_data)
        # 继续添加其他表的分支
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
                day_price = price_info.get("weekend", {}).get("with_meal", 0)
            else:  # 周一到周五
                day_price = price_info.get("weekday", {}).get("with_meal", 0)

            total_price += day_price
            current_date += timedelta(days=1)

        return total_price if total_price > 0 else None  # 返回总价或None

    def filter_accommodation(self, user_data):
        """
        根据用户输入筛选符合条件的住宿
        :param user_data: 用户输入的数据，包括出行人员和预算
        :return: 符合条件的住宿列表
        """
        logger.info("根据用户输入筛选住宿。")
        accommodations = self.data["cleaned_accommodation"]
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
                logger.info(f"住宿 {accommodation['name']} 不符合适用人群。")
                continue

            # # 1.2 计算住宿的总价格并筛选
            # accommodation_price = self.get_accommodation_price(price_info, service_time)
            # if accommodation_price is not None and accommodation_price > budget:
            #     logger.debug(f"由于预算超出，跳过住宿 {accommodation['name']}")
            #     continue

            # 如果通过所有条件筛选，将住宿添加到结果列表
            filtered_accommodations.append(accommodation)
            logger.info(f"住宿 {accommodation['name']} 符合用户条件。")

        return filtered_accommodations

    def filter_activities(self, user_data):
        """
        筛选符合用户偏好、年龄段、预算、时间和季节的活动
        :param user_data: 用户输入的数据，包括偏好、出行人员和预算
        :return: 符合条件的活动列表
        """
        logger.info("根据用户输入筛选活动。")
        activities = self.data["cleaned_activities"]
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
                logger.info(f"活动 {activity['activity_name']} 不符合用户偏好。")
                continue

            # # 1.2 获取活动价格并筛选
            # activity_price = self.get_activity_price(price_info, service_time, activity["duration"])
            # if activity_price is not None and activity_price > budget:
            #     logger.debug(f"由于预算超出，跳过活动 {activity['activity_name']}")
            #     continue

            # 1.3 检查季节是否符合
            if not self.matches_season(service_time, activity["suitable_season"]):
                logger.info(f"活动 {activity['activity_name']} 不适合当前季节。")
                continue

            start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")
            best_time = activity.get("best_time", "")
            reservation_days = self.normalize_value(activity.get("reservation_days"))
            opening_hours = {
                "weekday": activity.get("weekday_opening_hours", None),
                "saturday": activity.get("saturday_opening_hours", None),
                "sunday": activity.get("sunday_opening_hours", None)
            }

            # 2. 检查是否在用户时间范围内开放
            if not self.is_open_in_user_range(start_date, end_date, opening_hours):
                logger.debug(f"由于开放时间不符，跳过活动 {activity['activity_name']}")
                continue

            # 3. 检查最佳时段是否在开放时间内
            if not self.is_best_time_available(best_time, opening_hours):
                logger.debug(f"由于最佳时段不在开放时间内，跳过活动 {activity['activity_name']}")
                continue

            # 4. 检查是否满足预约要求
            if not self.check_reservation_requirements(reservation_days, start_date):
                logger.debug(f"由于预约要求不满足，跳过活动 {activity['activity_name']}")
                continue

            # 5. 检查适用人群是否匹配
            if not self.is_applicable_for_travelers(travelers, applicable_people):
                logger.debug(f"由于适用人群不匹配，跳过活动 {activity['activity_name']}")
                continue

            # 如果通过所有条件筛选，将活动添加到结果列表
            filtered_activities.append(activity)
            logger.info(f"活动 {activity['activity_name']} 符合用户偏好和条件。")

        return filtered_activities

    def matches_season(self, service_time, suitable_season):
        """
        检查活动是否适合当前季节或服务时间
        :param service_time: 用户输入的服务时间范围
        :param suitable_season: 活动适合的季节（例如：“春季、夏季”）
        :return: 是否符合季节要求
        """
        start_month = datetime.strptime(service_time["start_date"], "%Y-%m-%d").month
        end_month = datetime.strptime(service_time["end_date"], "%Y-%m-%d").month

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
        :param applicable_people: 活动的适用人群
        :return: 如果所有出行人员符合条件，返回True；否则返回False
        """
        # 定义英文到中文的年龄组映射
        age_group_translation = {
            "elderly": "老人",
            "adult": "成人",
            "teenagers": "青少年",
            "children": "儿童"
        }

        for traveler in travelers:
            traveler_age_group_en = traveler.get("age_group", "")
            traveler_age_group = age_group_translation.get(traveler_age_group_en, "")
            traveler_gender = traveler.get("gender", "")
            count = traveler.get("count", 0)

            # 如果count为0，跳过该出行人员
            if count == 0:
                continue

            # 检查是否符合适用人群
            if not any(person["age_group"] == traveler_age_group for person in applicable_people):
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

    def get_activity_price(self, price_info, service_time, duration):
        """
        根据活动的时间和套餐获取活动的价格（考虑跨越多个日期）
        :param price_info: 活动的价格信息
        :param service_time: 用户的服务时间（包含多个日期）
        :param duration: 活动的时长（小时）
        :return: 活动的总价格
        """
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

        total_price = 0  # 用于累加跨越多日的价格
        current_date = start_date

        while current_date <= end_date:
            # 判断当天是否免费
            if price_info.get("is_free", False):
                day_price = 0
            else:
                day_price = price_info.get("default", 0)

                # 如果有按时长收费，计算总价
                if price_info.get("unit_price"):
                    unit_price = price_info["unit_price"]["amount"]
                    day_price += unit_price * float(duration)

            total_price += day_price
            current_date += timedelta(days=1)

        return total_price if total_price > 0 else None  # 返回总价或None

    def generate_recommendation_basis(self, user_data, selected_accommodation, spa_activities):
        """
        根据用户输入和推荐结果生成推荐依据
        :param user_data: 用户输入的数据
        :param selected_accommodation: 选中的酒店
        :param spa_activities: 选中的温泉活动
        :return: 推荐依据的描述
        """
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

        # 检查出行人员的类型
        if travelers:
            traveler_types = set([traveler["age_group"] for traveler in travelers if traveler["count"] > 0])
            traveler_types_desc = "、".join(traveler_types)
            basis_parts.append(f"适合出行人员为{traveler_types_desc}")
        else:
            basis_parts.append("适合全家出行")

        # 使用选中的酒店信息
        if selected_accommodation:
            hotel_name = selected_accommodation.get("name", "默认酒店")
            basis_parts.append(f"推荐入住的酒店为{hotel_name}")

        # 拼接所有描述部分
        basis = "，".join(basis_parts) + "。"

        return basis

    def generate_default_itinerary(self, user_data, selected_hotel=None):
        """
        根据用户的服务时间，生成通用的行程方案，适合大多数人的默认安排。
        所有活动都从现有的活动列表中选择，确保每一天都有安排。
        :param user_data: 用户输入的数据
        :return: 默认行程的列表
        """
        service_time = user_data.get("service_time", {})
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

        total_days = (end_date - start_date).days + 1
        current_date = start_date
        itinerary = []

        # 如果没有提供酒店信息，使用默认的酒店
        if not selected_hotel:
            selected_hotel = {
                "name": "汤泉逸墅 院线房",
                "extra_info": {
                    "description": None,
                    "room_description": "院线房 61.66㎡-71.82㎡\n小院私汤，卧室有投影幕布，含双早"
                },
                "activity_code": "ACC992657"
            }

        # 已有的活动列表，用于随机选取
        available_activities = [
            {
                "name": "盐房",
                "location": "来康温泉",
                "activity_code": "ACT173738",
                "extra_info": {
                    "description": "经过加热以后盐会释放大量的钠离子成分,当人体置身在高温盐房环境里,毛孔全部打开,钠离子由毛孔进入人体的微循环,补充人体所需要的大量的微量元素。盐疗对于哮喘、气管炎、咽炎、肺炎、鼻炎等有一定缓解作用。",
                    "operation_tips": "建议泡汤时间不超过30分钟。"
                }
            },
            {
                "name": "岩洞氧吧",
                "location": "来康温泉",
                "activity_code": "ACT268949",
                "extra_info": {
                    "description": "抗氧化防衰老,可促进人体新陈代谢,具有提神、消除疲劳、提高免疫力的功效,从而增强抗病能力。",
                    "operation_tips": "体验时保持放松。"
                }
            },
            {
                "name": "香修体验",
                "location": "七修书院",
                "activity_code": "ACT698000",
                "extra_info": {
                    "description": "了解香学文化及制作合香香品",
                    "operation_tips": "请在老师指导下操作。"
                }
            },
            {
                "name": "花修体验",
                "location": "七修书院",
                "activity_code": "ACT098692",
                "extra_info": {
                    "description": "学习中式插花，享受宁静时光",
                    "operation_tips": "请提前预约花材。"
                }
            },
            {
                "name": "功修体验",
                "location": "七修书院",
                "activity_code": "ACT193928",
                "extra_info": {
                    "description": "导引养生功，讲解带练每一招式要点",
                    "operation_tips": "适合所有年龄段，活动轻松。"
                }
            },
            {
                "name": "书修体验",
                "location": "七修书院",
                "activity_code": "ACT796407",
                "extra_info": {
                    "description": "拓印、静心抄经，体验传统书法的静心力量",
                    "operation_tips": "建议提前预约，选择书法材料。"
                }
            },
            {
                "name": "食修体验",
                "location": "七修书院",
                "activity_code": "ACT875638",
                "extra_info": {
                    "description": "学习制作药食同源的小点心，了解传统健康饮食文化",
                    "operation_tips": "活动时间约1.5小时，请提前预约。"
                }
            }
        ]

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
                                "extra_info": {
                                    # "description": selected_hotel.get("extra_info", {}).get("description", "请提前确认入住时间，提醒需要预约等"),
                                    "description": "请提前确认入住时间，提醒需要预约等",
                                    "room_description": selected_hotel.get("extra_info", {}).get("room_description",
                                                                                                 ""),
                                    "operation_tips": "请提前确认入住时间，提醒需要预约等"
                                }
                            },
                            {
                                "name": "盐房",
                                "location": "来康温泉",
                                "activity_code": "ACT173738",
                                "extra_info": {
                                    "description": "经过加热以后盐会释放大量的钠离子成分,当人体置身在高温盐房环境里,毛孔全部打开,钠离子由毛孔进入人体的微循环,补充人体所需要的大量的微量元素。",
                                    "operation_tips": "建议泡汤时间不超过30分钟。"
                                }
                            }
                        ]
                    }
                ]
            else:
                # 每天上午和下午安排两个活动，随机从现有的活动中选择
                day_activities = [
                    {
                        "period": "上午",
                        "activities": [random.choice(available_activities)]
                    },
                    {
                        "period": "下午",
                        "activities": [random.choice(available_activities)]
                    }
                ]

            itinerary.append({
                "day": day,
                "date": current_date.strftime("%Y-%m-%d"),
                "time_slots": day_activities
            })

            current_date += timedelta(days=1)

        return itinerary

    def generate_itinerary(self, user_data):
        """
        根据用户数据生成行程清单
        :param user_data: 用户输入的数据，包括偏好、需求等
        :return: 行程清单的响应字典
        """
        # 1. 筛选符合条件的活动
        filtered_activities = self.filter_data("cleaned_activities", user_data)

        # 筛选出温泉类的活动
        spa_activities = [activity for activity in filtered_activities if
                          "温泉" in activity["activity_name"] or "温泉" in activity["activity_category"]]

        # 2. 随机选择一个符合条件的酒店
        selected_accommodation = self.filter_data("cleaned_accommodation", user_data)
        hotel = self.select_random_hotel(selected_accommodation)

        # 3. 创建行程，将选中的酒店和温泉活动作为参数传递
        if not filtered_activities:
            itinerary = self.generate_default_itinerary(user_data, hotel)
        else:
            itinerary = self.create_itinerary(user_data, filtered_activities, spa_activities, hotel)
        # 4. 生成推荐依据
        recommendation_basis = self.generate_recommendation_basis(user_data, hotel, spa_activities)

        # 5. 构建最终的响应结构
        response = {
            "head": 200,
            "items": {
                "hotel": {
                    "name": hotel["name"],
                    "extra_info": hotel["extra_info"],
                    "activity_code": hotel["activity_code"]
                },
                "recommendation_basis": recommendation_basis,
                "itinerary": itinerary,
                "msg": "行程生成成功"
            }
        }

        return response

    def select_random_hotel(self, selected_accommodation):
        """
        随机选择一个符合条件的酒店
        :param selected_accommodation: 经过筛选的符合条件的住宿列表
        :return: 随机选择的酒店信息
        """
        if selected_accommodation:
            hotel = random.choice(selected_accommodation)
            return {
                "name": hotel["name"],
                "location": "待定",
                "activity_code": hotel["activity_code"],
                "extra_info": {
                    "description": hotel.get("hotel_description", "房型丰富、设施齐全、中医理疗特色"),
                    "room_description": hotel.get("room_description", "")
                }
            }
        return {
            "name": "无合适酒店",
            "location": "",
            "activity_code": "",
            "extra_info": {
                "description": "无"
            }
        }

    def create_itinerary(self, user_data, filtered_activities, spa_activities, selected_hotel):
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
        start_date = datetime.strptime(service_time["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(service_time["end_date"], "%Y-%m-%d")

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
                if recent_activity["name"] == activity_name and (
                        current_date - recent_activity["date"]).days < min_gap_days:
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
                            "activity_code": selected_hotel["activity_code"],
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
                    random_spa_activity = random.choice(spa_activities)
                    day_activities[-1]["activities"].append({
                        "name": random_spa_activity["activity_name"],
                        "location": random_spa_activity["activity_category"],
                        "activity_code": random_spa_activity["activity_code"],
                        "extra_info": {
                            "description": random_spa_activity["description"],
                            "operation_tips": "建议泡汤时间不超过30分钟。"
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
                                        "name": activity["activity_name"],
                                        "location": activity["activity_category"],
                                        "activity_code": activity["activity_code"],
                                        "extra_info": {
                                            "description": activity["description"],
                                            "operation_tips": activity.get("reservation_note", "无")
                                        }
                                    }
                                ]
                            })
                            recent_activities.append({"name": activity["activity_name"], "date": current_date})
                            activity_index += 1

                # 如果当日没有活动，随机安排一个非温泉类活动
                if not day_activities:
                    random_activity = random.choice(filtered_activities)
                    day_activities.append({
                        "period": "上午",
                        "activities": [
                            {
                                "name": random_activity["activity_name"],
                                "location": random_activity["activity_category"],
                                "activity_code": random_activity["activity_code"],
                                "extra_info": {
                                    "description": random_activity["description"],
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