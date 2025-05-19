import requests
import time
import hashlib
from typing import Dict
from datetime import datetime, timedelta

# 管理群组映射关系字典
SCENE_MAPPING = {
    "BloodGlucoseManagement": "血糖管理",
    "BloodPressureManagement": "血压管理",
    "BodyFatAndWeightManagement": "体脂体重管理"
}

class ParameterFetcher:
    def __init__(self, **kwargs):
        """
        初始化ParameterFetcher类，加载配置项。

        :param config: 配置字典，包含api_key, api_secret和api_endpoints
        """
        self.api_key = kwargs.get("api_key")
        self.api_secret = kwargs.get("api_secret")
        self.host = kwargs.get("host")
        self.api_endpoints = kwargs.get("api_endpoints", {})

    def _generate_headers(self):
        """生成请求头信息"""
        req_time = str(int(time.time()))  # 当前时间戳（秒）
        sign_src = self.api_secret + req_time
        sign = hashlib.sha256(sign_src.encode('utf-8')).hexdigest()

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'authorization': f'Bearer {self.api_key}',
            'reqTime': req_time,
            'sign': sign
        }
        return headers

    def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像数据，并转换为 UserProfile 对象"""
        url = self.host + self.api_endpoints.get("获取用户画像")
        headers = self._generate_headers()
        params = {'userId': user_id}
        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"获取用户信息失败, 状态码: {response.status_code}, 错误信息: {response.text}")

        raw = response.json()
        if not raw.get("success") or "data" not in raw:
            return {}

        data = raw.get("data")

        # 使用映射函数将原始数据转换为需要的格式
        mapped_data = self.map_user_profile_data(data)

        res = {
            "user_profile": mapped_data,
            "expert_system": data.get("smartPlatform", "1"),
        }

        return res

    def map_user_profile_data(self, data: Dict) -> Dict:
        """将获取到的用户数据映射到UserProfile需要的格式"""
        health = data.get("healthRecord", {})

        mapped = {
            "age": data.get("age"),
            "gender": data.get("genderDesc") if data.get("genderDesc") else None,
            "height": str(data.get("height") or health.get("height")) if data.get("height") or health.get("height") else None,
            "weight": str(data.get("weight") or health.get("weight")) if data.get("weight") or health.get("weight") else None,
            "bmi": data.get("bmi") or health.get("bmi"),
            "current_diseases": data.get("currentDisease") or ",".join(health.get("currentDisease", [])),
            "disease_history": health.get("personalMedicalHistory", []),
            "allergic_history": health.get("drugAllergy", []) + health.get("foodAllergy", []),
            "surgery_history": [health.get("surgeryHistory")] if health.get("surgeryHistory") else [],
            "dietary_habits": health.get("eatingHabits"),
            "taste_preferences": ",".join(health.get("tastePreference", [])) if health.get("tastePreference") else None,
            "exercise_habits": health.get("sportType"),
            "sleep_quality": health.get("sleepQuality"),
            "recommended_caloric_intake": health.get("standardCalories"),
            "weight_status": health.get("weightStatus"),
            "daily_physical_labor_intensity": health.get("physicalLaborIntensity"),
            "management_goals": health.get("managementGoal"),
            "family_history": ",".join(health.get("familyMedicalHistory", [])) if health.get("familyMedicalHistory") else None,
            "food_allergies": ",".join(health.get("foodAllergy", [])) if health.get("foodAllergy") else None,
            "city": data.get("city"),
            "diabetes_medication": health.get("drugHistory", []),
        }

        return mapped

    def get_meals_info(self, group_id: str, start_time: str, end_time: str):
        """获取群内最近三小时的饮食识别记录，仅返回 data 列表"""
        url = self.host + self.api_endpoints.get("获取饮食信息")
        headers = self._generate_headers()

        # 确保请求头中有正确的 Content-Type
        headers["Content-Type"] = "application/json"

        payload = {
            "groupId": group_id,
            "startTime": start_time,
            "endTime": end_time
        }

        response = requests.post(url=url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"获取饮食信息失败, 状态码: {response.status_code}, 错误信息: {response.text}")

        raw = response.json()
        if not raw.get("success") or "data" not in raw:
            return []

        data = raw.get("data")
        return data  # 只返回 data 字段

    def get_nutritionist_feedback(self, user_id: str, time_range='3_hours'):
        """获取营养师点评"""
        url = self.host + self.api_endpoints.get("获取营养师点评")
        headers = self._generate_headers()
        params = {'userId': user_id, 'timeRange': time_range}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取营养师点评失败, 状态码: {response.status_code}, 错误信息: {response.text}")

    def get_group_schedule(self, group_id: str, is_new: bool):
        """根据群ID获取群日程，只返回 details 字段中的日程数据"""
        url = self.host + self.api_endpoints.get("获取群日程")
        headers = self._generate_headers()
        params = {'groupId': group_id, 'isNew': is_new}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"获取群日程失败, 状态码: {response.status_code}, 错误信息: {response.text}")

        raw = response.json()
        if not raw.get("success") or "data" not in raw:
            return {}

        # 只返回 details 中的日程数据
        return raw["data"].get("details", {})

    def get_group_info(self, group_id: str):
        """根据群组ID获取群组信息"""
        url = self.host + self.api_endpoints.get("获取群日程")
        headers = self._generate_headers()
        params = {'imGroupId': group_id}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"获取群组信息失败, 状态码: {response.status_code}, 错误信息: {response.text}")

        raw = response.json()
        if not raw.get("success") or "data" not in raw:
            return None

        group_data = raw.get("data")
        group_setting = group_data.get("groupSetting", {})

        # 提取 groupSceneTagCode
        group_scene_tag_code = group_setting.get("groupSceneTagCode", "")

        # 根据 groupSceneTagCode 进行映射
        management_type = SCENE_MAPPING.get(group_scene_tag_code, "未知管理类型")

        return management_type

    def get_all_group_chat_records_recent_3_hours(self, group_id: str):
        """
        自动分页拉取过去3小时内的所有有效群聊消息。
        仅保留 msgContent 存在且非 '[群日程消息]' 的文本，提取 userNick 和 content。
        """
        url = self.host + self.api_endpoints.get("获取群组聊天记录")
        headers = self._generate_headers()
        headers["Content-Type"] = "application/json"

        all_records = []
        last_msg_id = None
        last_msg_seq = None
        page_size = 100

        # 去除微秒部分的时间格式
        from_date = "2025-04-28 00:00:00"
        # from_date = (datetime.now() - timedelta(hours=3)).replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

        print(f"[DEBUG] 拉取聊天记录 groupId = {group_id}")
        print(f"[DEBUG] fromDate = {from_date}")

        while True:
            payload = {
                "groupId": group_id,
                "pageSize": page_size,
                "fromDate": from_date
            }

            if last_msg_id:
                payload["lastMsgId"] = last_msg_id
            if last_msg_seq:
                payload["lastMsgSeq"] = last_msg_seq

            request_json = {"groupChatRecordReq": payload}
            print(f"[DEBUG] 请求 payload = {request_json}")

            response = requests.post(url=url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(f"获取聊天记录失败: {response.status_code} - {response.text}")

            raw = response.json()
            if not raw.get("success"):
                raise Exception(f"聊天记录请求失败: {raw.get('message')}")

            page_data = raw.get("data", [])
            if not page_data:
                break

            # 只保留有意义的聊天内容
            for msg in page_data:
                content = msg.get("msgContent", "")
                if content and isinstance(content, str) \
                        and "[群日程消息]" not in content \
                        and "此文本不应该显示" not in content \
                        and "[图片]" not in content:
                    base_info = msg.get("baseUserInfo", {})
                    user_nick = base_info.get("userNick", "未知用户")
                    all_records.append({
                        "role": user_nick,
                        "content": content,
                        "timestamp": msg.get("msgTime")
                    })

            if len(page_data) < page_size:
                break

            last_msg_id = page_data[-1].get("id")
            last_msg_seq = page_data[-1].get("msgSeq")

        print(f"[DEBUG] 有效聊天记录条数：{len(all_records)}")
        return all_records

    def get_parameters(self, user_id: str, group_id: str, start_time: str, end_time: str):
        """
        获取用户所有必要的参数，并根据任务需要调整返回的参数

        :param user_id: 用户ID
        :param group_id: 群组ID
        :param start_time: 起始时间，格式：yyyy-MM-dd HH:mm:ss
        :param end_time: 结束时间，格式：yyyy-MM-dd HH:mm:ss
        :return: 汇总所有获取到的参数
        """
        user_profile = self.get_user_profile(user_id)
        # 更新为根据 group_id 和时间范围获取饮食信息
        meals_info = self.get_meals_info(group_id, start_time, end_time)

        # 汇总所有参数
        params = {
            "user_profile": user_profile,
            "meals_info": meals_info
        }

        return params

