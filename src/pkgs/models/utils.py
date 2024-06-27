# -*- encoding: utf-8 -*-
"""
@Time    :   2024-06-27 11:24:57
@desc    :   Tools for Agents models
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


from typing import List


class ParamTools:
    @classmethod
    def check_aigc_functions_sanji_plan_exercise_regimen(cls, params: dict) -> List:
        """检查参数是否满足三济方案 - 运动 - 运动调理原则

        必填项：
        1用户画像（其中必填项：年龄、性别、身高、体重、BMI、体力劳动强度、现患疾病或管理目标）
        2病历
        3体检报告, 暂无数据
        4检验/检查结果 未明确定义，暂不接入
        5关键指标数据
        上面5个，必须有一项
        """
        stats_records = {"user_profile": [], "medical_records": []}
        # 用户画像
        if (
            not params.get("user_profile")
            or not params["user_profile"].get("age")
            or not params["user_profile"].get("gender")
            or not params["user_profile"].get("height")
            or not params["user_profile"].get("weight")
            or not params["user_profile"].get("bmi")
            or not params["user_profile"].get("daily_physical_labor_intensity")
        ):
            stats_records["user_profile"].append("用户画像必填项缺失")
            if not params["user_profile"].get("age"):
                stats_records["user_profile"].append("age")
            if not params["user_profile"].get("gender"):
                stats_records["user_profile"].append("gender")
            if not params["user_profile"].get("height"):
                stats_records["user_profile"].append("height")
            if not params["user_profile"].get("weight"):
                stats_records["user_profile"].append("weight")
            if not params["user_profile"].get("bmi"):
                stats_records["user_profile"].append("bmi")
            if not params["user_profile"].get("daily_physical_labor_intensity"):
                stats_records["user_profile"].append("daily_physical_labor_intensity")
        if not params["user_profile"].get(
            "history_of_present_illness"
        ) and not params["user_profile"].get("health_goal"):
            stats_records["user_profile"].append(
                "user_profile 缺失现病史(history_of_present_illness) or 缺失健康管理目标(health_goal)"
            )
        if not params.get("medical_records"):
            stats_records["medical_records"].append("病历数据缺失")

        # 如果stats_records的values均非空, 返回首个非空的报错
        for k, v in stats_records.items():
            if not v:
                return
        for k, v in stats_records.items():
            if v:
                raise AssertionError(", ".join(v))
    @classmethod
    def check_aigc_functions_sanji_plan_exercise_plan(cls, params: dict) -> List:
        """检查参数是否满足三济方案 - 运动 - 运动调理原则

        必填项：
        1用户画像（其中必填项：年龄、性别、身高、体重、BMI、体力劳动强度、现患疾病或管理目标）
        2病历
        3体检报告 2024年06月27日下午郭姐告知暂无体检报告
        4检验/检查结果
        5关键指标数据
        上面5个，必须有一项
        """
        stats_records = {"user_profile": [], "medical_records": []}
        # 用户画像
        if (
            not params.get("user_profile")
            or not params["user_profile"].get("age")
            or not params["user_profile"].get("gender")
            or not params["user_profile"].get("height")
            or not params["user_profile"].get("weight")
            or not params["user_profile"].get("bmi")
            or not params["user_profile"].get("daily_physical_labor_intensity")
        ):
            stats_records["user_profile"].append("用户画像必填项缺失")
            if not params["user_profile"].get("age"):
                stats_records["user_profile"].append("age")
            if not params["user_profile"].get("gender"):
                stats_records["user_profile"].append("gender")
            if not params["user_profile"].get("height"):
                stats_records["user_profile"].append("height")
            if not params["user_profile"].get("weight"):
                stats_records["user_profile"].append("weight")
            if not params["user_profile"].get("bmi"):
                stats_records["user_profile"].append("bmi")
            if not params["user_profile"].get("daily_physical_labor_intensity"):
                stats_records["user_profile"].append("daily_physical_labor_intensity")
        if not params["user_profile"].get(
            "history_of_present_illness"
        ) and not params["user_profile"].get("health_goal"):
            stats_records["user_profile"].append(
                "user_profile 缺失现病史(history_of_present_illness) or 缺失健康管理目标(health_goal)"
            )
        if not params.get("medical_records"):
            stats_records["medical_records"].append("病历数据缺失")

        # 如果stats_records的values均非空, 返回首个非空的报错
        for k, v in stats_records.items():
            if not v:
                return
        for k, v in stats_records.items():
            if v:
                raise AssertionError(", ".join(v))
