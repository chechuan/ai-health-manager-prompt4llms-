# -*- encoding: utf-8 -*-
"""
@Time    :   2024-06-27 11:24:57
@desc    :   Tools for Agents models
@Author  :   ticoAg, chechuan
@Contact :   1627635056@qq.com 1163317515@qq.com
"""


from typing import Dict, List
from datetime import datetime

class ParamTools:

    @classmethod
    # @async_clock
    async def check_required_fields(cls, params: dict, required_fields: dict, at_least_one: list = []):
        """
        检查参数中的必填字段，包括多层嵌套和“或”的逻辑。

        参数:
            params (dict): 参数字典。
            required_fields (dict): 必填字段字典，键为参数名，值为该参数的必填字段列表、字典或元组（表示“或”逻辑）。
            field_names (dict): 字段中文释义字典。
            at_least_one (list): 至少需要有一项的参数列表（可选）。

        抛出:
            ValueError: 如果必填字段缺失或至少一项参数缺失，抛出错误。
        """
        missing = []

        async def add_missing(field, prefix=""):
            _prefix = prefix.rstrip(".")
            missing.append(f"{_prefix} 缺少 {field}")

        async def recursive_check(current_params, current_fields, prefix=""):
            """
            递归检查参数中的必填字段。

            参数:
                current_params (dict): 当前层级的参数字典。
                current_fields (dict or list): 当前层级的必填字段字典或列表。
                prefix (str): 当前字段的前缀，用于构建错误信息中的完整路径。
            """
            _prefix = prefix.rstrip(".")
            if not current_params:
                if _prefix not in at_least_one:
                    raise AssertionError(f"{_prefix} 不能为空")

            if isinstance(current_fields, dict):
                for param, fields in current_fields.items():
                    if param not in current_params or not current_params[param]:
                        # 如果参数不存在或为空，则记录缺失的参数
                        await add_missing(param, prefix)
                    elif isinstance(fields, (dict, list)):
                        await recursive_check(current_params[param], fields, prefix=f"{prefix}{param}.")
            elif isinstance(current_fields, list):
                for field in current_fields:
                    if isinstance(field, tuple):
                        # 如果字段是元组，表示“或”的逻辑，检查至少一个字段存在
                        if not any(f in current_params and current_params[f] is not None for f in field):
                            field_names_str = '或'.join(field for field in field)
                            await add_missing(field_names_str, prefix)
                    elif isinstance(field, str):
                        if field not in current_params or current_params[field] is None:
                            # 如果必填字段不存在或为空，则记录缺失的字段
                            await add_missing(field, prefix)

        # 检查至少一项参数存在的条件
        if at_least_one and not any(params.get(p) for p in at_least_one):
            # 打印执行时间日志
            raise AssertionError(f"至少需要提供以下一项参数：{', '.join(at_least_one)}")
        # 检查所有必填字段
        for key, fields in required_fields.items():
            if key in params:
                await recursive_check(params[key], fields, prefix=f"{key}.")
            else:
                await add_missing(key)

        # 如果有缺失的必填字段，抛出错误
        # print(missing)
        if missing:
            raise AssertionError(f"缺少必要的字段：{'; '.join(missing)}")

    @classmethod
    async def check_aigc_functions_body_fat_weight_management_consultation(
            cls, params: dict
    ) -> List:
        """检查参数是否满足需求

    - Args
        intentCode: str
            - aigc_functions_body_fat_weight_management_consultation
            - aigc_functions_weight_data_analysis
            - aigc_functions_body_fat_weight_data_analysis
    """
        stats_records = {"user_profile": [], "key_indicators": []}
        intentCode = params.get("intentCode")
        if intentCode == "aigc_functions_body_fat_weight_management_consultation":
            # 用户画像
            if (
                    not params.get("user_profile")
                    or not params["user_profile"].get("age")
                    or not params["user_profile"].get("gender")
                    or not params["user_profile"].get("height")
            ):
                stats_records["user_profile"].append("用户画像必填项缺失")
                if not params["user_profile"].get("age"):
                    stats_records["user_profile"].append("age")
                if not params["user_profile"].get("gender"):
                    stats_records["user_profile"].append("gender")
                if not params["user_profile"].get("height"):
                    stats_records["user_profile"].append("height")
            if not params.get("key_indicators"):
                stats_records["key_indicators"].append("缺少关键指标数据")
            else:
                key_list = [i["key"] for i in params["key_indicators"]]
                if "体重" not in key_list:
                    stats_records["key_indicators"].append("体重")
                if "bmi" not in key_list:
                    stats_records["key_indicators"].append("bmi")
                for item in params["key_indicators"]:
                    if item["key"] == "体重":
                        if not item.get("data"):
                            stats_records["key_indicators"].append("体重数据缺失")
                        elif not isinstance(item["data"], list):
                            stats_records["key_indicators"].append("体重数据格式不符")
                    elif item["key"] == "bmi":
                        if not item.get("data"):
                            stats_records["key_indicators"].append("BMI数据缺失")
                        elif not isinstance(item["data"], list):
                            stats_records["key_indicators"].append("BMI数据格式不符")

        elif intentCode == "aigc_functions_weight_data_analysis":
            # 用户画像检查
            if (
                    not params.get("user_profile")
                    or not params["user_profile"].get("age")
                    or not params["user_profile"].get("gender")
                    or not params["user_profile"].get("height")
                    or not params["user_profile"].get("bmi")
            ):
                stats_records["user_profile"].append("用户画像必填项缺失")
                if not params["user_profile"].get("age"):
                    stats_records["user_profile"].append("age")
                if not params["user_profile"].get("gender"):
                    stats_records["user_profile"].append("gender")
                if not params["user_profile"].get("height"):
                    stats_records["user_profile"].append("height")
                if not params["user_profile"].get("bmi"):
                    stats_records["user_profile"].append("bmi")
            if not params.get("key_indicators"):
                stats_records["key_indicators"].append("缺少关键指标数据")
            else:
                key_list = [i["key"] for i in params["key_indicators"]]
                if "体重" not in key_list:
                    stats_records["key_indicators"].append("体重")
                if "bmi" not in key_list:
                    stats_records["key_indicators"].append("bmi")
                for item in params["key_indicators"]:
                    if item["key"] == "体重":
                        if not item.get("data"):
                            stats_records["key_indicators"].append("体重数据缺失")
                        elif not isinstance(item["data"], list):
                            stats_records["key_indicators"].append("体重数据格式不符")
                    elif item["key"] == "bmi":
                        if not item.get("data"):
                            stats_records["key_indicators"].append("BMI数据缺失")
                        elif not isinstance(item["data"], list):
                            stats_records["key_indicators"].append("BMI数据格式不符")

        elif intentCode == "aigc_functions_body_fat_weight_data_analysis":
            if (
                    not params.get("user_profile")
                    or not params["user_profile"].get("gender")
            ):
                stats_records["user_profile"].append("用户画像必填项缺失")
                if not params["user_profile"].get("gender"):
                    stats_records["user_profile"].append("gender")

            if not params.get("key_indicators"):
                stats_records["key_indicators"].append("缺少关键指标数据")
            else:
                key_list = [i["key"] for i in params["key_indicators"]]
                if "体脂率" not in key_list:
                    stats_records["key_indicators"].append("体脂率")
                for item in params["key_indicators"]:
                    if item["key"] == "体脂率":
                        if not item.get("data"):
                            stats_records["key_indicators"].append("体脂率数据缺失")
                        elif not isinstance(item["data"], list):
                            stats_records["key_indicators"].append("体脂率数据格式不符")

        for k, v in stats_records.items():
            if v:
                raise AssertionError(", ".join(v))

    @staticmethod
    def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
        """
        计算基础代谢率（BMR）

        参数:
            weight (float): 体重（千克）
            height (float): 身高（厘米）
            age (int): 年龄
            gender (str): 性别（"男" 或 "女"）

        返回:
            float: 计算出的基础代谢率
        """
        if gender == "男":
            return 10 * weight + 6.25 * height - 5 * age + 5
        elif gender == "女":
            return 10 * weight + 6.25 * height - 5 * age - 161
        else:
            raise ValueError("性别必须是 '男' 或 '女' (gender must be '男' or '女')")

    @classmethod
    async def check_and_calculate_bmr(cls, user_profile: dict) -> float:
        """
        检查计算基础代谢率（BMR）所需的数据是否存在，并计算BMR

        参数:
            user_profile (dict): 包含用户画像信息的字典
            field_names (dict): 字段中文释义字典

        返回:
            float: 计算出的基础代谢率

        抛出:
            ValueError: 如果缺少必要的数据，抛出错误
        """
        required_fields = ["weight", "height", "age", "gender"]
        missing_fields = [field for field in required_fields if field not in user_profile]
        if missing_fields:
            raise ValueError(
                f"缺少计算基础代谢率所需的数据 (missing data to calculate BMR): {', '.join(missing_fields)}")

        weight = cls.parse_measurement(user_profile["weight"], "weight")
        height = cls.parse_measurement(user_profile["height"], "height")
        age = int(user_profile["age"])
        gender = user_profile["gender"]

        return cls.calculate_bmr(weight, height, age, gender)

    @staticmethod
    def parse_measurement(value_str: str, measure_type: str) -> float:
        """
        解析测量值字符串，支持体重和身高

        参数:
            value_str (str): 测量值字符串
            measure_type (str): 测量类型（"weight" 或 "height"）

        返回:
            float: 解析出的测量值

        抛出:
            ValueError: 如果未知的测量类型或单位，抛出错误
        """
        if measure_type == "weight":
            if "kg" in value_str:
                return float(value_str.replace("kg", "").strip())
            elif "公斤" in value_str:
                return float(value_str.replace("公斤", "").strip())
            elif "斤" in value_str:
                return float(value_str.replace("斤", "").strip()) * 0.5
            else:
                raise ValueError("未知的体重单位")
        elif measure_type == "height":
            if "cm" in value_str:
                return float(value_str.replace("cm", "").strip())
            elif "米" in value_str:
                return float(value_str.replace("米", "").strip()) * 100
            else:
                raise ValueError("未知的身高单位")
        else:
            raise ValueError("未知的测量类型")


def get_highest_data_per_day(blood_pressure_data: List[Dict]) -> List[Dict]:
    """
    从给定的血压数据列表中获取每一天血压最高的一条数据。

    优先考虑收缩压最高的记录，如果收缩压相同，则选择舒张压最高的记录。

    Args:
        blood_pressure_data: 包含日期和血压数据的字典列表。

    Returns:
        List[Dict]: 每天血压最高的一条数据列表。
    """
    highest_data_per_day = {}

    for entry in blood_pressure_data:
        date = entry['date'].split(' ')[0]  # 仅获取日期部分（字符串）

        if date not in highest_data_per_day:
            highest_data_per_day[date] = entry  # 初始赋值
        else:
            current_record = highest_data_per_day[date]
            # 比较当前记录与已有记录的收缩压和舒张压值，优先选择收缩压高的记录
            if (entry['sbp'] > current_record['sbp']) or (
                entry['sbp'] == current_record['sbp'] and entry['dbp'] > current_record['dbp']):
                highest_data_per_day[date] = entry

    # 返回按日期排序后的列表
    return sorted(highest_data_per_day.values(), key=lambda x: x['date'])


def check_consecutive_days(blood_pressure_data: List[Dict]) -> bool:
    """
    判断给定的血压数据是否是连续5天的数据。

    Args:
        blood_pressure_data: 包含日期和血压数据的字典列表，每天可能有多条数据。

    Returns:
        bool: 如果数据是连续5天的，返回True；否则返回False。
    """
    # 获取每一天最近的一条数据
    blood_pressure_data = get_highest_data_per_day(blood_pressure_data)

    # 提取日期并排序
    dates = sorted(set([entry['date'].split(' ')[0] for entry in blood_pressure_data]))
    if len(dates) < 5:
        return False

    for i in range(4):
        if (datetime.strptime(dates[i + 1], '%Y-%m-%d') - datetime.strptime(dates[i], '%Y-%m-%d')).days != 1:
            return False

    return True


# 工具函数：判断血压等级
def determine_blood_pressure_level(sbp: float, dbp: float) -> str:
    """
    根据收缩压和舒张压判断血压等级。

    Args:
        sbp: 收缩压值
        dbp: 舒张压值

    Returns:
        str: 血压等级（如"正常血压", "1级高血压"等）
    """
    if sbp < 90 or dbp < 60:
        return "低血压"
    elif 90 <= sbp < 120 and 60 <= dbp < 80:
        return "正常血压"
    elif 120 <= sbp < 140 or 80 <= dbp < 90:
        return "正常高值"
    elif 140 <= sbp < 160 or 90 <= dbp < 100:
        return "1级高血压"
    elif 160 <= sbp < 180 or 100 <= dbp < 110:
        return "2级高血压"
    elif sbp >= 180 or dbp >= 110:
        return "3级高血压"
    return ""  # 默认返回空字符串
