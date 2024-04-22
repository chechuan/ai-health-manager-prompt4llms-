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
    height: str = Field(None, description="身高", examples=["175cm", "1.8米"])  # 身高
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
    intentCode: Literal[
        "aigc_functions_single_choice",
        "aigc_functions_consultation_summary",
        "aigc_functions_diagnosis",
        "aigc_functions_diagnosis_result",
        "aigc_functions_drug_recommendation",
        "aigc_functions_food_principle",
        "aigc_functions_sport_principle",
        "aigc_functions_mental_principle",
        "aigc_functions_chinese_therapy",
        "aigc_functions_reason_for_care_plan",
    ] = Field(
        description="意图编码/事件编码",
        examples=[
            "aigc_functions_consultation_summary",
        ],
    )
    url: str = Field(None, description="请求地址")
    prompt: Optional[str] = None
    options: Optional[List[str]] = None
    user_profile: UserProfile = Field(
        {},
        description="用户基本信息",
        examples=[{"age": 18, "gender": "男", "weight": "65kg"}],
    )
    messages: Optional[List[ChatMessage]] = Field(
        None,
        description="对话历史",
        examples=[
            [
                {"role": "user", "content": "我肚子疼"},
                {
                    "role": "assistant",
                    "content": "你的腹痛是突然出现的还是慢慢发生的？",
                },
                {"role": "user", "content": "突然出现的"},
                {
                    "role": "assistant",
                    "content": "你能描述一下腹痛的部位吗，是在上腹部、下腹部还是两侧？",
                },
                {"role": "user", "content": "下腹痛"},
                {
                    "role": "assistant",
                    "content": "这种疼痛是一阵一阵的还是持续不断的？疼痛的程度如何，能否忍受？",
                },
                {"role": "user", "content": "一阵一阵的，还可以忍受"},
                {
                    "role": "assistant",
                    "content": "你的腹痛是刺痛、钝痛还是绞痛？同时有没有伴随其他症状，比如发热、恶心、呕吐、腹泻或者便秘？",
                },
                {"role": "user", "content": "绞痛吧，有点恶心"},
                {
                    "role": "assistant",
                    "content": "你的恶心感是与腹痛同时出现的吗？你有拉肚子或者便秘的情况吗？饮食和排便习惯近期有没有改变？",
                },
                {"role": "user", "content": "同时出现，没有别的症状"},
                {
                    "role": "assistant",
                    "content": "你近期有没有做过剧烈运动或者饮食上吃了不易消化的食物？这种腹痛是在月经期间更容易发生吗？",
                },
                {"role": "user", "content": "同时出现，没有别的症状"},
            ]
        ],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[[{"stream": False}]],
    )
    durg_plan: List[DrugPlanItem] = Field(
        None,
        description="aigc_functions_drug_recommendation输出的药方列表",
        examples=[
            [
                {
                    "drug_name": "蒙脱石散",
                    "dosage": "1袋",
                    "frequency": "3次/天",
                    "usage": "饭前半小时空腹用温水冲服",
                    "precautions": "服药期间避免食用生冷、油腻食物",
                    "contraindication": "对蒙脱石过敏者禁用",
                },
                {
                    "drug_name": "肠胃宁胶囊",
                    "dosage": "2粒",
                    "frequency": "3次/天",
                    "usage": "饭后服用",
                    "precautions": "孕妇慎用，服药期间避免饮酒",
                    "contraindication": "对本品过敏者禁用",
                },
                {
                    "drug_name": "复方消化酶胶囊",
                    "dosage": "2粒",
                    "frequency": "3次/天",
                    "usage": "饭时或饭后服用",
                    "precautions": "儿童、孕妇、哺乳期妇女应在医生指导下使用",
                    "contraindication": "对本品过敏者禁用",
                },
            ]
        ],
    )
    diagnosis: str = Field(
        None,
        description="诊断结果",
        examples=["急性肠胃炎"],
    )
    food_principle: str = Field(
        None,
        description="饮食原则",
        examples=[
            '饮食调理原则：目标是缓解肠胃炎症状，促进肠胃功能恢复。推荐饮食方案为"低脂易消化膳食"。该方案低脂易消化，减轻肠胃负担，同时确保营养供应。避免油腻和刺激性食物，多吃蒸煮食品，如瘦肉、鱼、蔬菜泥、水果泥等。注意饮食卫生，分餐多次，少量多餐。'
        ],
    )
    sport_principle: str = Field(
        None,
        description="运动原则",
        examples=[
            "由于你被诊断为急性肠胃炎，建议暂时避免剧烈运动，等待病情恢复。在症状缓解后，可以逐步开始轻度运动，如散步、瑜伽。运动时间可从每次15分钟开始，逐渐增加到30分钟，每天1-2次。注意运动时不要吃得过饱，避免饭后立即运动。最佳运动心率保持在最大心率的50%-70%之间，即约112-156次/分钟。最大心率=220-年龄。恢复期间，保持良好的饮食习惯和充足的休息，有助于身体康复。如果运动过程中感到不适，应立即停止并就医。"
        ],
    )
    mental_principle: str = Field(
        None,
        description="情志原则",
        examples=[
            "情志调理原则：保持心情愉悦，减轻焦虑。进行深呼吸练习，每日冥想10-15分钟。选择轻松的音乐助眠，保证7-9小时高质量睡眠。避免剧烈运动，做如瑜伽等轻柔运动促进身体舒缓。定期与亲朋交流，分享心情。如疼痛持续或加重，请及时就医。"
        ],
    )
    chinese_therapy: str = Field(
        None,
        description="中医疗法",
        examples=[
            "针灸推拿：针对急性肠胃炎，可选取中脘、气海、天枢、足三里等穴位进行温和的针灸治疗，以调理脾胃，缓解腹痛和恶心。配合轻柔的腹部推拿，促进气血流通，加速炎症消退。\n\n药膳调理：建议采用健脾和胃、清热解毒的食材。如山药、薏米、白术、黄连、金银花等，可煮粥或炖汤食用。同时，减少油腻、辛辣食物，以减轻肠胃负担。\n\n茶饮调养：推荐饮用陈皮茶，以理气消胀；搭配薄荷叶，可缓解恶心感；再加点菊花，清热解毒。每日适量饮用，有助于肠胃功能恢复。避免冷饮，以防加重肠胃负担。同时，多饮温开水，保持水分平衡。\n\n此外，生活调理也至关重要，保持规律作息，避免过度劳累，保持心情舒畅，有助于身体的康复。如有必要，可配合中药汤剂，但需在专业中医师指导下使用。"
        ],
    )
    plan_ai: str = Field(
        None,
        description="AI给出的方案",
        examples=["AI方案示例"],
    )
    plan_human: str = Field(
        None,
        description="专家修改后的方案",
        examples=["专家方案示例"],
    )


class AigcFunctionsResponse(BaseModel):
    code: int = Field(200, description="API status code")
    message: str = Field(None, description="返回内容")
    end: bool = Field(False, description="流式结束标志")


class AigcFunctionsCompletionResponse(BaseModel):
    head: int = Field(200, description="API status code")
    items: Union[str, object] = Field(None, description="返回内容")
    msg: str = Field("", description="报错信息")
