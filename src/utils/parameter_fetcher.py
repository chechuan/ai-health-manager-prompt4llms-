import requests
import time
import hashlib
from typing import Dict


class ParameterFetcher:
    def __init__(self, **kwargs):
        """
        初始化ParameterFetcher类，加载配置项。

        :param config: 配置字典，包含api_key, api_secret和api_endpoints
        """
        self.api_key = kwargs.get("api_key")
        self.api_secret = kwargs.get("api_secret")
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
        url = self.api_endpoints.get("获取用户画像")
        headers = self._generate_headers()
        params = {'userId': user_id}
        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"获取用户信息失败, 状态码: {response.status_code}, 错误信息: {response.text}")

        raw = response.json()
        if not raw.get("success") or "data" not in raw:
            return None

        data = raw.get("data")

        # 使用映射函数将原始数据转换为需要的格式
        mapped_data = self.map_user_profile_data(data)

        # 返回 UserProfile 实例
        return mapped_data

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

    def get_meals_info(self, user_id: str, time_range='3_hours'):
        """获取用户饮食信息，time_range 可为 '3_hours' 或 'any_time'"""
        url = self.api_endpoints.get("获取饮食信息")
        headers = self._generate_headers()
        params = {'userId': user_id, 'timeRange': time_range}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取饮食信息失败, 状态码: {response.status_code}, 错误信息: {response.text}")

    def get_nutritionist_feedback(self, user_id: str, time_range='3_hours'):
        """获取营养师点评"""
        url = self.api_endpoints.get("获取营养师点评")
        headers = self._generate_headers()
        params = {'userId': user_id, 'timeRange': time_range}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取营养师点评失败, 状态码: {response.status_code}, 错误信息: {response.text}")

    def get_warning_indicators(self, user_id: str):
        """获取预警指标数据"""
        url = self.api_endpoints.get("获取预警指标")
        headers = self._generate_headers()
        params = {'userId': user_id}

        response = requests.get(url=url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"获取预警指标失败, 状态码: {response.status_code}, 错误信息: {response.text}")

    def get_parameters(self, user_id: str, time_range='3_hours'):
        """
        获取用户所有必要的参数，并根据任务需要调整返回的参数

        :param user_id: 用户ID
        :param time_range: 获取饮食信息和点评的时间范围
        :return: 汇总所有获取到的参数
        """
        user_profile = self.get_user_profile(user_id)
        meals_info = self.get_meals_info(user_id, time_range)
        nutritionist_feedback = self.get_nutritionist_feedback(user_id, time_range)
        warning_indicators = self.get_warning_indicators(user_id)

        # 汇总所有参数
        params = {
            "user_profile": user_profile,
            "meals_info": meals_info,
            "nutritionist_feedback": nutritionist_feedback,
            "warning_indicators": warning_indicators
        }

        return params
