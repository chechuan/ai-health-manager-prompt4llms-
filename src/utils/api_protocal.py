# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-15 11:37:35
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

from datetime import date
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from click import File
from fastapi import Body
from pydantic import BaseModel, Field

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
    role: Literal["user", "assistant", "system", "function", "other"] = "other"
    content: str = Field(...)


class ChatMessages(BaseModel):
    messages: List[ChatMessage] = Field(...)


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
    "diagnosis_list": "诊断",
    "bmi": "BMI",
    "daily_physical_labor_intensity": "体力劳动强度",
    "current_diseases": "现患疾病",
    "management_goals": "管理目标",
    "food_allergies": "食物过敏",
    "special_diet": "特殊饮食习惯",
    "taste_preferences": "口味偏好",
    "special_physiological_period": "是否特殊生理期",
    "traditional_chinese_medicine_constitution": "中医体质",
    "region": "所处地域",
    "exercise_habits": "运动习惯",
    "exercise_level": "运动水平",
    "exercise_risk": "运动风险",
    "emotional_issues": "情志问题",
    "diagnosis": "诊断",
    "standard_weight": "标准体重",
    "target_weight": "用户目标体重",
    "standard_body_fat_rate": "标准体脂率"
}


DIETARY_GUIDELINES_KEY_MAP = {
    "focus_issues": "重点关注问题",
    "suitable_foods": "适宜吃",
    "unsuitable_foods": "不宜吃",
    "basic_nutritional_needs": "基础营养需求",
}


class DrugPlanItem(BaseModel):
    drug_name: str  # 药品名称
    dosage: str  # 剂量
    frequency: str  # 频次
    usage: str  # 用法
    precautions: str  # 注意事项
    contraindication: str  # 禁忌
    dosage_time: str  # 用药时间


class JiaheUserProfile(BaseModel):
    age: str = Field("未知", description="年龄")
    gender: str = Field("未知", description="性别")
    height: str = Field("未知", description="身高")
    weight: str = Field("未知", description="体重")
    manage_object: str = Field("未知", description="管理目标")
    disease: str = Field("未知", description="现患疾病")
    special_diet: str = Field("未知", description="特殊饮食习惯")
    allergy_food: str = Field("未知", description="过敏食物")
    taste_preference: str = Field("未知", description="口味偏好")
    tase_taboo: str = Field("未知", description="口味禁忌")
    is_specific_menstrual_period: str = Field("未知", description="是否特殊生理期")
    constitution: str = Field("未知", description="中医体质")


class UserProfile(BaseModel):
    # age: int = Field(None, description="年龄", ge=0, le=200)
    # gender: Literal["男", "女"] = Field(None, description="性别", examples=["男", "女"])
    age: Optional[int] = None
    gender: Optional[str] = None
    height: str = Field(None, description="身高", examples=["175cm", "1.8米"])
    weight: str = Field(None, description="体重", examples=["65kg", "90斤"])
    target_weight: str = Field(None, description="目标体重", examples=["65kg", "90斤"])
    weight_evaluation: Optional[str] = Field(
        None, description="体重评价", examples=["正常"]
    )
    bmi: Union[None, float] = None
    disease_history: Union[None, List[str]] = []  # 疾病史
    allergic_history: Union[None, List[str]] = []  # 过敏史
    surgery_history: Union[None, List[str]] = []  # 手术史
    main_diagnosis_of_western_medicine: Optional[str] = Field(
        None, description="西医主要诊断", examples=["高血压"]
    )
    secondary_diagnosis_of_western_medicine: Optional[str] = Field(
        None, description="西医次要诊断"
    )
    traditional_chinese_medicine_diagnosis: Optional[str] = None  # 中医诊断
    traditional_chinese_medicine_syndrome_types: Optional[str] = None  # 中医证型
    traditional_chinese_medicine_constitution: Optional[str] = None  # 中医体质
    dietary_habits: Optional[str] = Field(
        None, description="饮食习惯", examples=["少食"]
    )  # 饮食习惯
    body_temperature: Optional[str] = None  # 体温(摄氏度)
    respiratory_rate: Optional[str] = None  # 呼吸频率(次/分)
    pulse_rate: Optional[str] = None  # 脉搏(次/分)
    diastolic_blood_pressure: Optional[Union[float, int]] = None  # 舒张压(mmHg)
    systolic_blood_pressure: Optional[Union[float, int]] = None  # 收缩压(mmHg)
    chief_complaint: Optional[str] = None  # 主诉
    history_of_present_illness: Optional[str] = None  # 现病史
    family_history_of_disease: Optional[str] = None  # 家族疾病史
    past_history_of_present_illness: Optional[str] = None  # 既往史
    specialist_check: Optional[str] = None  # 专科检查
    disposal_plan: Optional[str] = None  # 处置方案
    nation: str = Field(None, description="民族", example=["汉族"])
    daily_physical_labor_intensity: Optional[str] = Field(
        None, description="日常体力劳动水平", examples=["中"]
    )
    mood_swings: Optional[str] = Field(
        None, description="情绪波动", examples=["正常波动"]
    )
    # 运动风险等级
    motion_risk_level: Optional[str] = Field(
        None, description="运动风险等级", examples=["正常"]
    )
    exercise_intensity: Optional[str] = Field(
        None, description="运动强度", examples=["正常强度"]
    )
    current_diseases: Optional[str] = Field(None, description="现患疾病", example=["高血压, 糖尿病"])
    management_goals: Optional[str] = Field(None, description="管理目标", example=["减重, 降血压"])
    food_allergies: Optional[str] = Field(None, description="食物过敏", example=["花生"])
    special_diet: Optional[str] = Field(None, description="特殊饮食习惯", example=["素食"])
    taste_preferences: Optional[str] = Field(None, description="口味偏好", example=["清淡"])
    special_physiological_period: Optional[str] = Field(None,
                                                        description="是否特殊生理期，如备孕期、孕早期、孕中期、孕晚期等",
                                                        example=["备孕期"])
    region: Optional[str] = Field(None, description="所处地域", example=["北京"])
    exercise_habits: Optional[str] = Field(None, description="运动习惯", example=["每天锻炼"])
    exercise_level: Optional[str] = Field(None, description="运动水平", example=["中等"])
    exercise_risk: Optional[str] = Field(None, description="运动风险", example=["低"])
    emotional_issues: Optional[str] = Field(None, description="情志问题", example=["焦虑"])


class AigcFunctionsRequest(BaseModel):
    intentCode: Literal[
        "switch_exercise",
        "report_summary",
        "report_interpretation",
        "aigc_functions_single_choice",
        "aigc_functions_consultation_summary",
        "aigc_functions_consultation_summary_chief_disease",
        "aigc_functions_diagnosis",
        "aigc_functions_diagnosis_result",
        "aigc_functions_drug_recommendation",
        "aigc_functions_food_principle",
        "aigc_functions_sport_principle",
        "aigc_functions_mental_principle",
        "aigc_functions_chinese_therapy",
        "aigc_functions_food_principle_new",
        "aigc_functions_sport_principle_new",
        "aigc_functions_mental_principle_new",
        "aigc_functions_chinese_therapy_new",
        "aigc_functions_reason_for_care_plan",
        "aigc_functions_doctor_recommend",
        "aigc_functions_consultation_summary_to_group",
        "aigc_functions_auxiliary_history_talking",
        "aigc_functions_auxiliary_diagnosis",
        "aigc_functions_relevant_inspection",
    ] = Field(
        description="意图编码/事件编码",
        examples=[
            "aigc_functions_consultation_summary",
        ],
    )
    url: str = Field(None, description="请求地址")
    prompt: Optional[str] = None
    chunk_size: Optional[int] = Field(
        None, description="报告总结分块", examples=[1000], ge=0, le=2048
    )
    report_content: Optional[str] = Field(
        None,
        description="报告总结报告内容",
        examples=[
            (
                "首页部分生命熵值60.85熵值评价中度失衡中度失衡解读熵值中度偏高"
                "提示近期所患疾病出现不稳定情况，或身体较长期处于慢病状态"
            )
        ],
    )
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
    drug_plan: Union[None, str, List[DrugPlanItem]] = Field(
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
    diagnosis: Union[str, List, None] = Field(
        None,
        description="诊断结果",
        examples=["急性肠胃炎"],
    )
    food_principle: Union[str, None] = Field(
        None,
        description="饮食原则",
        examples=[
            '饮食调理原则：目标是缓解肠胃炎症状，促进肠胃功能恢复。推荐饮食方案为"低脂易消化膳食"。该方案低脂易消化，减轻肠胃负担，同时确保营养供应。避免油腻和刺激性食物，多吃蒸煮食品，如瘦肉、鱼、蔬菜泥、水果泥等。注意饮食卫生，分餐多次，少量多餐。'
        ],
    )
    sport_principle: Union[str, None] = Field(
        None,
        description="运动原则",
        examples=[
            "由于你被诊断为急性肠胃炎，建议暂时避免剧烈运动，等待病情恢复。在症状缓解后，可以逐步开始轻度运动，如散步、瑜伽。运动时间可从每次15分钟开始，逐渐增加到30分钟，每天1-2次。注意运动时不要吃得过饱，避免饭后立即运动。最佳运动心率保持在最大心率的50%-70%之间，即约112-156次/分钟。最大心率=220-年龄。恢复期间，保持良好的饮食习惯和充足的休息，有助于身体康复。如果运动过程中感到不适，应立即停止并就医。"
        ],
    )
    mental_principle: Union[str, None] = Field(
        None,
        description="情志原则",
        examples=[
            "情志调理原则：保持心情愉悦，减轻焦虑。进行深呼吸练习，每日冥想10-15分钟。选择轻松的音乐助眠，保证7-9小时高质量睡眠。避免剧烈运动，做如瑜伽等轻柔运动促进身体舒缓。定期与亲朋交流，分享心情。如疼痛持续或加重，请及时就医。"
        ],
    )
    chinese_therapy: Union[str, None] = Field(
        None,
        description="中医疗法",
        examples=[
            "针灸推拿：针对急性肠胃炎，可选取中脘、气海、天枢、足三里等穴位进行温和的针灸治疗，以调理脾胃，缓解腹痛和恶心。配合轻柔的腹部推拿，促进气血流通，加速炎症消退。\n\n药膳调理：建议采用健脾和胃、清热解毒的食材。如山药、薏米、白术、黄连、金银花等，可煮粥或炖汤食用。同时，减少油腻、辛辣食物，以减轻肠胃负担。\n\n茶饮调养：推荐饮用陈皮茶，以理气消胀；搭配薄荷叶，可缓解恶心感；再加点菊花，清热解毒。每日适量饮用，有助于肠胃功能恢复。避免冷饮，以防加重肠胃负担。同时，多饮温开水，保持水分平衡。\n\n此外，生活调理也至关重要，保持规律作息，避免过度劳累，保持心情舒畅，有助于身体的康复。如有必要，可配合中药汤剂，但需在专业中医师指导下使用。"
        ],
    )
    plan_ai: Union[str, None] = Field(
        None,
        description="AI给出的方案",
        examples=["AI方案示例"],
    )
    plan_human: Union[str, None] = Field(
        None,
        description="专家修改后的方案",
        examples=["专家方案示例"],
    )


class AigcSanjiRequest(BaseModel):
    intentCode: Literal[
        "sanji_assess_3d_classification",
        "sanji_assess_keyword_classification",
        "sanji_assess_3health_classification",
        "sanji_assess_literature_classification",
        "sanji_intervene_goal_classification",
        # "sanji_intervene_literature_classification",
    ] = Field(
        description="意图编码/事件编码",
        examples=[
            "sanji_intervene_goal_classification",
        ],
    )

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
            ]
        ],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[[{"stream": False}]],
    )

    diagnosis: Union[str, None] = Field(
        None,
        description="诊断结果",
        examples=["急性肠胃炎"],
    )

    health_goal: Union[str, None] = Field(
        None,
        description="健康管理目标",
        examples=["恢复肠道健康"],
    )

    # currentDate: Union[str, None] = Field(
    #     None,
    #     description="当前日期",
    #     examples=["2024年6月20日"],
    # )

    # currentLoc: Union[str, None] = Field(
    #     None,
    #     description="当前地点",
    #     examples=["廊坊"],
    # )


class AigcFunctionsDoctorRecommendRequest(BaseModel):
    intentCode: Literal["aigc_functions_doctor_recommend",] = Field(
        description="意图编码/事件编码",
        examples=[
            "aigc_functions_doctor_recommend",
        ],
    )
    prompt: Optional[str] = Field(
        None,
        description="辅助诊断 & 报告解读chat 事件结束时的输出",
        examples=[
            (
                "李明，你的口腔检查结果显示有两颗蛀牙和一颗继发龋齿，可能与饮食习惯和口腔清洁有关。"
                "虽然你少吃糖，但主食吃得多可能也会增加蛀牙风险。牙刷软毛是好的，但未使用巴氏刷牙法可能清洁效果不足。"
                "建议每日至少刷牙两次，使用牙线清理牙缝，学习并实践巴氏刷牙法。纠正咬手指的习惯对预防牙齿不正也至关重要。"
                "此外，定期全口涂氟和口腔检查能有效预防蛀牙。记住，良好的口腔卫生是长期维护牙齿健康的关键。"
            )
        ],
    )
    messages: Optional[List[ChatMessage]] = Field(
        ...,
        description="对话历史",
        examples=[
            [
                {
                    "role": "assistant",
                    "content": "请问是否需要我帮您找一位医生？",
                },
                {
                    "role": "user",
                    "content": "我想找个西医比较厉害的医生",
                },
            ]
        ],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[{"stream": False}],
    )


class AigcFunctionsResponse(BaseModel):
    code: int = Field(200, description="API status code")
    message: str = Field(None, description="返回内容")
    end: bool = Field(False, description="流式结束标志")


class AigcFunctionsCompletionResponse(BaseModel):
    head: int = Field(200, description="API status code")
    items: Union[str, object] = Field(None, description="返回内容")
    msg: str = Field("", description="报错信息")


class DoctorInfo(BaseModel):
    doctor_name: str = Field(..., description="医生姓名")
    doctor_introduction: str = Field(None, description="医生信息")
    doctor_specialty: str = Field(None, description="医生擅长")
    organization_name: str = Field(None, description="机构名称")
    consultation_department: str = Field(None, description="出诊科室")
    gender: str = Field(None, description="性别")
    doctor_title: str = Field(None, description="医生职称")

    def __str__(self) -> str:
        return (
            f"姓名: {self.doctor_name}\n"
            f"医生信息:{self.doctor_introduction}\n"
            f"医生擅长:{self.doctor_specialty}\n"
            f"机构名称: {self.organization_name}\n"
            f"出诊科室: {self.consultation_department}\n"
            f"性别: {self.gender}\n"
            f"医生职称: {self.doctor_title}\n"
        )


class KeyIndicators(BaseModel): ...


class bloodPressureLevelResponse(BaseModel):
    level: int = Field(..., description="血压等级")
    contents: List[str] = Field([], description="要返回的话术")
    idx: int = Field(0, description="未知")
    thought: str = Field("", description="生成思考的内容")
    scheme_gen: int = Field(
        0, description="跳转子页面的图标显示在contents中的第几条会话 作为contents的索引"
    )
    visit_verbal_idx: int = Field(-1, description="上门话术索引")
    contact_doctor: int = Field(-1, description="联系医生索引")
    scene_ending: bool = Field(False, description="场景结束标志")
    blood_trend_gen: bool = Field(False, description="前端是否显示血压趋势图")
    notifi_daughter_doctor: bool = Field(False, description="通知女儿和医生")
    call_120: bool = Field(False, description="是否呼叫120")
    is_visit: bool = Field(False, description="是否上门")
    exercise_video: bool = Field(False, description="是否显示锻炼视频")
    events: List[Dict] = Field([], description="后续跟随事件")


class MedicalRecords(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="主诉")
    present_illness_history: Optional[str] = Field(None, description="现病史")
    past_history_of_present_illness: Optional[str] = Field(None, description="既往史")
    allergic_history: Optional[List[str]] = Field(None, description="过敏史")
    diagnosis_list: Optional[List[str]] = Field(None, description="诊断")


# 西医决策支持
class OutpatientSupportRequest(BaseModel):
    intentCode: Literal[
        "aigc_functions_diagnosis_generation",
        "aigc_functions_chief_complaint_generation",
        "aigc_functions_generate_present_illness",
        "aigc_functions_generate_past_medical_history",
        "aigc_functions_generate_allergic_history",
        "aigc_functions_generate_medication_plan",
        "aigc_functions_generate_examination_plan",
    ] = Field(description="意图编码/事件编码")
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[[{"stream": False}]],
    )
    user_profile: Optional[UserProfile] = None
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
    medical_records: Optional[MedicalRecords] = Field(
        None,
        description="病历",
        examples=[
            {
                "history_of_present_illness": "高血压",
                "chief_complaint": "肚子疼",
                "past_history_of_present_illness": "糖尿病",
                "allergic_history": ["无"],
                "diagnosis_list": ["无"],
            }
        ],
    )


# 三济康养方案
class DietaryGuidelinesDetails(BaseModel):
    focus_issues: Optional[str] = Field(
        None,
        description="重点关注问题",
        example="控制糖分摄入，减少甜食，注意监测血糖，适当增加运动。",
    )
    suitable_foods: Optional[str] = Field(
        None,
        description="适宜吃",
        example="高纤维食物，如全谷类、蔬菜和水果；低糖、低脂的乳制品；瘦肉和鱼。",
    )
    unsuitable_foods: Optional[str] = Field(
        None,
        description="不宜吃",
        example="高糖食物，如甜饮料、糖果；高脂食物，如炸食、肥肉；精细加工食品。",
    )
    basic_nutritional_needs: Optional[str] = Field(
        None,
        description="基础营养需求",
        example="每日热量摄取约1800-2000千卡，以55%的碳水化合物、25%的蛋白质和20%的健康脂肪为主。多吃绿叶蔬菜，保证每日至少1500ml水分摄入，避免饮酒。",
    )


class HistoricalDiet(BaseModel):
    date: str = Field(None, description="当前日期", example="2024年6月20日")
    meals: Optional[Dict[str, List[str]]] = Field(
        None,
        description="餐次和食物名称",
        example={
            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
            "晚餐": ["玉米", "鸡腿", "牛奶"],
        },
    )


class MealPlan(BaseModel):
    meal: str = Field(description="餐次", example="早餐")
    foods: List[str] = Field(
        description="食物名称", example=["燕麦粥", "鸡蛋", "拌菠菜", "小米粥"]
    )


class SanJiKangYangRequest(BaseModel):
    intentCode: Literal[
        "aigc_functions_sjkyn_guideline_generation",
        "aigc_functions_dietary_guidelines_generation",
        "aigc_functions_dietary_details_generation",
        "aigc_functions_meal_plan_generation",
        "aigc_functions_generate_food_quality_guidance",
        "aigc_functions_sanji_plan_exercise_regimen",
        "aigc_functions_sanji_plan_exercise_plan",
    ] = Field(
        description="意图编码/事件编码",
        examples=[
            "aigc_functions_sanji_plan_exercise_plan",
            "aigc_functions_sanji_plan_exercise_regimen",
        ],
    )
    user_profile: Optional[UserProfile] = Field(
        None,
        description="三济康养方案用户画像，包含详细的用户信息",
        examples=[{"age": 18, "gender": "男", "weight": "65kg"}],
    )
    medical_records: Optional[MedicalRecords] = Field(
        None,
        description="病历",
        examples=[
            {
                "history_of_present_illness": "高血压",
                "chief_complaint": "肚子疼",
                "past_history_of_present_illness": "糖尿病",
                "allergic_history": ["无"],
                "diagnosis_list": ["无"],
            }
        ],
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
                }
            ]
        ],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[[{"stream": False}]],
    )

    food_principle: Union[str, None] = Field(
        None,
        description="饮食原则",
        examples=[
            '饮食调理原则：目标是缓解肠胃炎症状，促进肠胃功能恢复。推荐饮食方案为"低脂易消化膳食"。该方案低脂易消化，减轻肠胃负担，同时确保营养供应。避免油腻和刺激性食物，多吃蒸煮食品，如瘦肉、鱼、蔬菜泥、水果泥等。注意饮食卫生，分餐多次，少量多餐。'
        ],
    )
    ietary_guidelines: Optional[DietaryGuidelinesDetails] = Field(
        None,
        description="饮食调理细则",
        examples=[
            {
                "重点关注问题": "控制糖分摄入，减少甜食，注意监测血糖，适当增加运动。",
                "适宜吃": "高纤维食物，如全谷类、蔬菜和水果；低糖、低脂的乳制品；瘦肉和鱼。",
                "不宜吃": "高糖食物，如甜饮料、糖果；高脂食物，如炸食、肥肉；精细加工食品。",
                "基础营养需求": "每日热量摄取约1800-2000千卡，以55%的碳水化合物、25%的蛋白质和20%的健康脂肪为主。多吃绿叶蔬菜，保证每日至少1500ml水分摄入，避免饮酒。",
            }
        ],
    )
    historical_diets: Optional[List[HistoricalDiet]] = Field(
        None,
        description="历史食谱",
        examples=[
            {
                "historical_diets": [
                    {
                        "date": "2024-05-01",
                        "meals": {
                            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
                            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
                            "晚餐": ["玉米", "鸡腿", "牛奶"],
                        },
                    },
                    {
                        "date": "2024-05-02",
                        "meals": {
                            "早餐": ["豆腐脑", "鸡蛋", "凉拌芹菜"],
                            "午餐": ["大米饭", "清炒油麦菜", "红烧鸡翅", "芹菜汁"],
                            "晚餐": ["玉米", "鸡腿", "牛奶"],
                        },
                    },
                ]
            }
        ],
    )
    meal_plan: List[MealPlan] = Field(
        None,
        description="待输出质量的食谱列表",
        examples=[
            {"meal": "早餐", "foods": ["燕麦粥", "鸡蛋", "拌菠菜", "小米粥"]},
            {"meal": "上午加餐", "foods": ["小番茄"]},
        ],
    )


class BodyFatWeightManagementRequest(BaseModel):
    intent_code: Literal[
        "aigc_functions_body_fat_weight_management_consultation",
        "aigc_functions_weight_data_analysis_1day",
        "aigc_functions_weight_data_analysis_2day",
        "aigc_functions_weight_data_analysis_multiday",
        "aigc_functions_body_fat_weight_data_analysis_1day",
        "aigc_functions_body_fat_weight_data_analysis_2day",
        "aigc_functions_body_fat_weight_data_analysis_multiday"
    ] = Field(
        description="意图编码/事件编码",
        examples=[
            "aigc_functions_sanji_plan_exercise_plan",
            "aigc_functions_sanji_plan_exercise_regimen"
        ]
    )
    user_profile: UserProfile = Field(
        ...,
        description="三济康养方案用户画像，包含详细的用户信息",
        examples=[{"age": 18, "gender": "男", "weight": "65kg"}],
    )
    messages: ChatMessages = Field(
        None,
        description="对话历史",
        examples=[
            [
                {"role": "user", "content": "我肚子疼"},
                {
                    "role": "assistant",
                    "content": "你的腹痛是什么时候出现的？",
                },
                {"role": "user", "content": "突然出现的"},
            ]
        ],
    )
    model_args: Union[Dict, None] = Field(
        None,
        description="模型参数",
        examples=[[{"stream": False}]],
    )
    key_indicators: Optional[KeyIndicators] = Field(
        None,
        description="关键指标",
        examples=[
            {
                "key_indicators": [
                    {
                        "key": "体重",
                        "data": [
                            {"time": "2024-06-28 10:25:32", "value": "80.8"},
                            {"time": "2024-06-27 10:25:32", "value": "80.3"},
                        ],
                    },
                    {
                        "key": "体脂率",
                        "data": [
                            {"time": "2024-06-28 10:25:32", "value": "30.10%"},
                            {"time": "2024-06-27 10:25:32", "value": "30.50%"},
                        ],
                    },
                    {
                        "key": "bmi",
                        "data": [
                            {"time": "2024-06-28 10:25:32", "value": "26.08"},
                            {"time": "2024-06-27 10:25:32", "value": "25.92"},
                        ],
                    },
                ]
            }
        ],
    )
