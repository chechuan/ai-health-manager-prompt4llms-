# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-15 11:37:35
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

from typing import Any, Dict, List, Literal, Tuple, Union, Optional
from pydantic import BaseModel, Field
from fastapi import Body

BaseModel.model_config["protected_namespaces"] = ("model_config",)


class ihmHealthData(BaseModel):
    date: str
    value: float


class healthBloodPressureTrendAnalysis(BaseModel):
    ihm_health_sbp: List[ihmHealthData]
    ihm_health_dbp: List[ihmHealthData]
    ihm_health_hr: List[ihmHealthData]


class RolePlayChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None
    schedule: Optional[List] = None


class RolePlayRequest(BaseModel):
    model: str
    messages: List[RolePlayChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    items: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {"example": {"code": 200, "msg": "success", "items": ""}}


class TestRequest(BaseModel):
    body: Any = Body("", description="对话框ID")
    items: str = Field("test string", description="测试字符串输入")


USER_PROFILE_KEY_MAP = {
    "age": "年龄",
    "gender": "性别",
    "height": "身高",
    "weight": "体重",
    "disease_history": "疾病史",
    "allergic_history": "过敏史",
    "surgery_history": "手术史",
    "drug_name": "药品名称",
    "dosage": "剂量",
    "frequency": "频次",
    "usage": "用法",
    "precautions": "注意事项",
    "contraindication": "禁忌",
    "main_diagnosis_of_western_medicine": "西医主要诊断",
    "secondary_diagnosis_of_western_medicine": "西医次要诊断",
    "traditional_chinese_medicine_diagnosis": "中医诊断",
    "traditional_chinese_medicine_syndrome_types": "中医证型",
    "body_temperature": "体温(摄氏度)",
    "respiratory_rate": "呼吸频率(次/分)",
    "pulse_rate": "脉搏(次/分)",
    "diastolic_blood_pressure": "舒张压(mmHg)",
    "systolic_blood_pressure": "收缩压(mmHg)",
    "chief_complaint": "主诉",
    "history_of_present_illness": "现病史",
    "past_history_of_present_illness": "既往史",
    "specialist_check": "专科检查",
    "disposal_plan": "处置方案",
}


class DrugPlanItem(BaseModel):
    drug_name: str  # 药品名称
    dosage: str  # 剂量
    frequency: str  # 频次
    usage: str  # 用法
    precautions: str  # 注意事项
    contraindication: str  # 禁忌


class UserProfile(BaseModel):
    age: int = Field(description="年龄", ge=1, le=100)  # 年龄
    gender: Literal["男", "女"] = Field(description="性别", examples=["男", "女"])
    height: str = Body(None, description="身高", examples=["175cm", "1.8米"])  # 身高
    weight: str = Field(None, description="体重", examples=["65kg", "90斤"])  # 体重
    disease_history: Union[None, List[str]] = []  # 疾病史
    allergic_history: Union[None, List[str]] = []  # 过敏史
    surgery_history: Union[None, List[str]] = []  # 手术史
    main_diagnosis_of_western_medicine: Optional[str] = None  # 西医主要诊断
    secondary_diagnosis_of_western_medicine: Optional[str] = None  # 西医次要诊断
    traditional_chinese_medicine_diagnosis: Optional[str] = None  # 中医诊断
    traditional_chinese_medicine_syndrome_types: Optional[str] = None  # 中医证型
    body_temperature: Optional[str] = None  # 体温(摄氏度)
    respiratory_rate: Optional[str] = None  # 呼吸频率(次/分)
    pulse_rate: Optional[str] = None  # 脉搏(次/分)
    diastolic_blood_pressure: Optional[Union[float, int]] = None  # 舒张压(mmHg)
    systolic_blood_pressure: Optional[Union[float, int]] = None  # 收缩压(mmHg)
    chief_complaint: Optional[str] = None  # 主诉
    history_of_present_illness: Optional[str] = None  # 现病史
    past_history_of_present_illness: Optional[str] = None  # 既往史
    specialist_check: Optional[str] = None  # 专科检查
    disposal_plan: Optional[str] = None  # 处置方案


class AigcFunctionsRequest(BaseModel):
    intentCode: str = Field(
        "aigc_functions_consultation_summary", description="意图编码/事件编码"
    )
    url: str = Field(None, description="请求地址")
    prompt: Optional[str] = None
    options: Optional[List[str]] = None
    user_profile: UserProfile = Field(
        {},
        description="用户基本信息",
        type="object",
        example=UserProfile(age=18, gender="男", weight="65kg"),
    )
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="对话历史",
        example=[{"role": "user", "content": "最近早上经常咳嗽,怎么办"}],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        example={"stream": False},
    )
    durg_plan: Optional[List[DrugPlanItem]] = None  # 药方
    diagnosis: Optional[str] = None  # 诊断
    food_principle: Optional[str] = None  # 饮食原则
    sport_principle: Optional[str] = None  # 运动原则
    mental_principle: Optional[str] = None  # 心理原则
    chinese_therapy: Optional[str] = None  # 中医疗法
    plan_ai: Optional[str] = None  # AI方案
    plan_human: Optional[str] = None  # 人工方案


class AigcFunctionsResponse(BaseModel):
    code: int = 200
    message: str = ""
    end: bool = False


class AigcFunctionsCompletionResponse(BaseModel):
    head: int = 200
    items: Union[str, object] = ""
    msg: str = ""
