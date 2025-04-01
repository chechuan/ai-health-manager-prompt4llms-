# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

import sys
import time
import json
import re
import yaml
import asyncio
import aiohttp
import requests
import functools
import oss2
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    Any, AnyStr, AsyncGenerator, Dict, Generator, List, Tuple, Union, Optional
)
from contextlib import contextmanager

from lunar_python import Lunar, Solar

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

import openai

from src.utils.api_protocal import AigcFunctionsResponse
from src.utils.openai_api_protocal import (
    CompletionResponseStreamChoice, CompletionStreamResponse
)

try:
    from src.utils.Logger import logger
except ImportError:
    from Logger import logger


def clock(func):
    """info level 函数计时装饰器"""

    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        if time_cost < 1:
            logger.debug(func.__name__ + " -> {} ms".format(int(1000 * time_cost)))
        else:
            logger.info(func.__name__ + " -> {} s".format(int(time_cost)))
        return result

    return clocked


def param_check(check_params: List[AnyStr] = []):
    def dector(func):
        async def wrap(*args, **kwargs):
            for key in check_params:
                if key not in kwargs:
                    raise ValueError(f"No {key} passed.")
                elif kwargs[key] is None or kwargs[key] == []:
                    raise ValueError(f"{key} can't be empty")
            result = await func(*args, **kwargs)
            return result

        return wrap

    return dector


def update_mid_vars(
        mid_vars, input_text=Any, output_text=Any, key="节点名", model="调用模型", **kwargs
):
    """更新中间变量"""
    lth = len(mid_vars) + 1
    mid_vars.append(
        {
            "id": lth,
            "key": key,
            "input_text": input_text,
            "output_text": output_text,
            "model": model,
            **kwargs,
        }
    )
    return mid_vars


def make_meta_ret(
        end=False, msg="", code=None, type="Result", init_intent: bool = False, **kwargs
):
    ret = {
        "end": end,
        "message": msg,
        "intentCode": code,
        "type": type,
        "init_intent": init_intent,
        **kwargs,
    }
    if kwargs.get("gsr"):
        if not kwargs["gsr"].intent_desc_map.get(code):
            ret["intentDesc"] = "日常对话"
        else:
            ret["intentDesc"] = kwargs["gsr"].intent_desc_map[code]
        del ret["gsr"]
    else:
        ret["intentDesc"] = "闲聊"
    return ret


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical(exc_value.message, exc_info=(exc_type, exc_value, exc_traceback))


def load_yaml(path: Union[Path, str]):
    return yaml.load(open(path, "r", encoding="utf-8"), Loader=yaml.FullLoader)


def handle_error(func):
    def __inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            handle_exception(exc_type, exc_value, exc_tb)
        finally:
            print(e.message)

    return __inner


def loadJS(path):
    return json.load(open(path, "r", encoding="utf-8"))


class NpEncoder(json.JSONEncoder):
    """json npencoder, dumps时对np数据格式转为python原生格式"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def intent_init():
    # ==================== 固安来康郡相关 ====================
    LAKANG_INTENTS = {
        "温泉推荐": ("spa_rec", "温泉推荐"),
        "温泉知识": ("spa_knowledge", "温泉知识"),
        "行程推荐": ("route_rec", "行程推荐"),
        "固安来康郡": ("gu_an_laikangjun", "固安来康郡"),
    }

    # ==================== 医疗健康相关 ====================
    MEDICAL_INTENTS = {
        "医疗": ("med_health", "医疗健康"),
        "辅助诊断": ("auxiliary_diagnosis", "辅助诊断"),
        "问诊": ("auxiliary_diagnosis", "辅助诊断",),
        "用药": ("drug_rec", "用药咨询"),
        "血糖": ("blood_glucose_counseling", "血糖咨询"),
        "低血压": ("hypotensive_consultation", "低血压咨询"),
        "BMI": ("bmi_query", "BMI"),
        "健康知识咨询": ("health_qa", "健康知识科普"),
        "家康宝": ("jia_kang_bao", "家康宝服务咨询"),
    }

    # ==================== 饮食营养相关 ====================
    FOOD_INTENTS = {
        "饮食营养": ("food_nutri", "饮食营养"),
        "饮食处方": ("food_rec", "饮食处方推荐"),
        "饮食评价": ("food_eval", "饮食评价"),
        "营养其他": ("nutri_other", "营养其他"),
        "菜谱": ("cookbook", "菜谱"),
    }

    # ==================== 食材采购相关 ====================
    PURCHASE_INTENTS = {
        "食材采购": ("food_purchasing", "食材采购"),
    }

    # ==================== 专业人员咨询 ====================
    EXPERT_INTENTS = {
        "医师": ("call_doctor", "呼叫医师"),
        "医生": ("call_doctor", "呼叫医师"),
        # "运动师": ("call_sportMaster", "呼叫运动师"),
        # "心理": ("call_psychologist", "呼叫情志师"),
        # "情志": ("call_psychologist", "呼叫情志师"),
        # "营养师": ("call_dietista", "呼叫营养师"),
        # "健管师": ("call_health_manager", "呼叫健管师"),
        # "五师": ("wushi", "呼叫五师"),
        "呼叫其他": ("call_other", "呼叫其他"),
    }

    # ==================== 内容服务相关 ====================
    CONTENT_INTENTS = {
        "音乐": ("musicX", "音乐播放"),
        "音频": ("audio", "音频播放"),
        "新闻": ("news", "新闻"),
        "故事": ("story", "故事"),
        "圣经": ("AIUI.Bible", "圣经"),
        "戏曲": ("drama", "戏曲"),
        "评书": ("storyTelling", "评书"),
        "有声书": ("AIUI.audioBook", "有声书"),
        "笑话": ("joke", "笑话"),
    }

    # ==================== 实用工具相关 ====================
    UTILITY_INTENTS = {
        "天气": ("weather", "天气查询"),
        "网络": ("websearch", "网络搜索"),
        "彩票": ("lottery", "彩票"),
        "解梦": ("dream", "周公解梦"),
        "计算器": ("AIUI.calc", "计算器"),
        "翻译": ("translation", "翻译"),
        "垃圾": ("garbageClassifyPro", "垃圾分类"),
        "尾号限行": ("carNumber", "尾号限行"),
        "单位换算": ("AIUI.unitConversion", "单位换算"),
        "汇率": ("AIUI.forexPro", "汇率"),
        "眼保健操": ("AIUI.ocularGym", "眼保健操"),
        "时间日期": ("datetimePro", "时间日期"),
        "万年历": ("calendar", "万年历"),
    }
    # ==================== 日程管理 ====================
    DAYLY_INTENTS = {
        "日程管理": ("schedule_manager", "日程管理"), }

    # ==================== 其他功能 ====================
    OTHER_INTENTS = {
        "拉群共策": ("shared_decision", "拉群共策"),
        "新奥百科": ("enn_wiki", "新奥百科知识"),
        "猜你想问": ("aigc_functions_generate_related_questions", "猜你想问"),
    }

    # 复合条件意图
    COMPOUND_INTENTS = [
        (lambda t: "血压测量" in t or "测量血压" in t,
         ("remind_take_blood_pressure", "提醒他人测量血压")),
        (lambda t: "运动切换" in t or "切换运动" in t,
         ("switch_exercise", "运动切换")),
        (lambda t: "数字人" in t and "换回" in t,
         ("digital_image_back", "换回数字人皮肤")),
        (lambda t: "数字人" in t and "切换" in t,
         ("digital_image_switch", "切换数字人皮肤")),
        (lambda t: "功能页面" in t and "打开" in t,
         ("open_Function", "打开功能页面")),
        (lambda t: "设置页面" in t and "打开" in t,
         ("open_page", "打开页面")),
        (lambda t: "非会议" in t and "日程管理" in t,
         ("other_schedule", "非会议日程管理")),
        (lambda t: "会议" in t and "日程管理" in t and "非会议" not in t,
         ("meeting_schedule", "会议日程管理")),
        (lambda t: "食材采购清单" in t and "管理" in t,
         ("food_purchasing_list_management", "食材采购清单管理")),
        (lambda t: "食材采购清单" in t and "确认" in t,
         ("food_purchasing_list_verify", "食材采购清单确认")),
        (lambda t: "食材采购清单" in t and "关闭" in t,
         ("food_purchasing_list_close", "食材采购清单关闭")),
        (lambda t: "食材采购清单" in t and "生成" in t,
         ("create_food_purchasing_list", "生成食材采购清单")),
    ]

    # 合并所有简单意图
    ALL_INTENTS = {}
    for intent_dict in [
        LAKANG_INTENTS,
        MEDICAL_INTENTS,
        FOOD_INTENTS,
        PURCHASE_INTENTS,
        EXPERT_INTENTS,
        CONTENT_INTENTS,
        UTILITY_INTENTS,
        DAYLY_INTENTS,
        OTHER_INTENTS,
    ]:
        ALL_INTENTS.update(intent_dict)
    return ALL_INTENTS, COMPOUND_INTENTS


def get_intent(text, all_intent, com_intent):
    """通过关键词解析意图->code"""
    # 1. 检查复合条件
    for condition, intent in com_intent:
        if condition(text):
            code, desc = intent
            logger.debug(f"识别出的意图:{text} code:{code}")
            return code, desc

    # 2. 检查简单映射
    for keyword in all_intent.keys():
        if keyword in text:
            (code, desc) = all_intent[keyword]
            logger.debug(f"识别出的意图:{text} code:{code}")
            return code, desc

    # 3. 默认返回
    logger.debug(f"识别出的意图:{text} code:other")
    return "other", "日常对话"


def get_doc_role(code):
    if code == "call_dietista":
        return "ROLE_NUTRITIONIST"
    elif code == "call_sportMaster":
        return "ROLE_EXERCISE_SPECIALIST"
    elif code == "call_psychologist":
        return "ROLE_EMOTIONAL_COUNSELOR"
    elif code == "call_doctor":
        return "ROLE_DOCTOR"
    elif code == "call_health_manager":
        return "ROLE_HEALTH_SPECIALIST"
    else:
        return "ROLE_HEALTH_SPECIALIST"


def _parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    h = text.find("Thought:")
    i = text.find("\nAction:")
    j = text.find("\nAction Input:")
    k = (
        text.find("\nObservation:")
        if text.find("\nObservation:") > 0
        else j + len("\nAction Input:") + text[j + len("\nAction Input:"):].find("\n")
    )
    l = text.find("\nFinal Answer:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
            k = text.rfind("\nObservation:")
    if 0 <= i < j < k:
        plugin_thought = text[h + len("Thought:"): i].strip()
        plugin_name = text[i + len("\nAction:"): j].strip()
        plugin_args = text[j + len("\nAction Input:"): k].strip()
        return plugin_thought, plugin_name, plugin_args
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len("Thought:"): l].strip()
            plugin_args = text[l + len("\nFinal Answer:"):].strip()
            plugin_args.split("\n")[0]
            return plugin_thought, "直接回复用户问题", plugin_args
        else:
            plugin_args = text[l + len("\nFinal Answer:"):].strip()
            return "I know the final answer.", "直接回复用户问题", plugin_args
    return "", ""


def parse_latest_plugin_call(text: str, plugin_name: str = "AskHuman"):
    # TODO 优化解析逻辑
    h = text.find("\nThought:")
    i = text.find("\nAction:")
    j = text.find("\nAction Input:")
    k1 = text.find("\nObservation:")
    k2 = len(text[:j]) + text[j:].find("\nThought:")

    k = k1 if k1 and k1 > 0 else k2
    l = text.find("\nFinal Answer:")

    # plugin_name = "AskHuman"
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nThought:"  # Add it back.
            k = text.rfind("\nThought:")
    if 0 <= i < j < k:
        plugin_thought = text[h + len("\nThought:"): i].strip()
        plugin_name = text[i + len("\nAction:"): j].strip()
        plugin_args = text[j + len("\nAction Input:"): k].strip()
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len("Thought:"): l].strip()
            plugin_args = text[l + len("\nFinal Answer:"):].strip()
            plugin_args.split("\n")[0]
        else:
            plugin_args = text[l + len("\nFinal Answer:"):].strip()
            plugin_thought = "I know the final answer."
    else:
        m = text.find("\nAnswer: ")
        next_thought_index = text[m + len("\nAnswer: "):].find("\nThought:")
        if next_thought_index == -1:
            n = len(text)
        else:
            n = m + len("\nAnswer: ") + next_thought_index
        plugin_thought = text[len("\nThought: "): m].strip()
        plugin_args = text[m + len("\nAnswer: "): n].strip()
    return [plugin_thought, plugin_name, plugin_args]


def curr_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def date_after_days(days: int):
    now = datetime.now()
    date_after = (now + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return date_after


def date_after(**kwargs):
    now = datetime.now()
    date_after = (now + timedelta(**kwargs)).strftime("%Y-%m-%d %H:%M:%S")
    return date_after


def this_sunday():
    """返回下周一0点0分0秒"""
    today = datetime.strptime(datetime.now().strftime("%Y%m%d"), "%Y%m%d")
    return datetime.strftime(
        today + timedelta(7 - today.weekday()), "%Y-%m-%d %H:%M:%S"
    )


def curr_weekday():
    today = date.today().strftime("%A")
    return today


def dumpJS(obj, ensure_ascii=False, **kwargs):
    return json.dumps(obj, ensure_ascii=ensure_ascii, **kwargs)


def format_sse_chat_complete(data: str, event=None) -> str:
    msg = "data: {}\n\n".format(data)
    if event is not None:
        msg = "event: {}\n{}".format(event, msg)
    return msg


def accept_stream_response(response, verbose=True) -> str:
    """接受openai.response的stream响应"""
    content = ""
    tst = time.time()
    for chunk in response:
        # if chunk.object == "text_completion":
        if not chunk.object or "chat" not in chunk.object:
            if hasattr(chunk.choices[0], "text"):
                chunk_text = chunk.choices[0].text
                if chunk_text:
                    content += chunk_text
                    if verbose:
                        print(chunk_text, end="", flush=True)
        else:
            if hasattr(chunk.choices[0].delta, "content"):
                chunk_text = chunk.choices[0].delta.content
                if chunk_text:
                    content += chunk_text
                    if verbose:
                        print(chunk_text, end="", flush=True)
    if verbose:
        print()
    t_cost = round(time.time() - tst, 2)
    logger.debug(f"Model {chunk.model}, Generate {len(content)} words, Cost {t_cost}s")
    return content


def compute_blood_pressure_level(x: int, flag: str = "l" or "h") -> int:
    """计算血压等级 flag区分低血压or高血压 不同level"""
    if flag == "l":
        if x < 60:
            return -1
        if x <= 90:
            return 0
        elif x <= 90:
            return 1
        elif x <= 109:
            return 2
        else:
            return 3
    elif flag == "h":
        if x < 90:
            return -1
        if x <= 140:
            return 0
        elif x <= 159:
            return 1
        elif x <= 179:
            return 2
        else:
            return 3


def apply_chat_template(prompt: str, template: str = "chatml"):
    """应用chatml模板
    <|im_start|>user
    你是什么模型<|im_end|>
    <|im_start|>assistant
    """
    if template == "chatml":
        prompt = (
                "<|im_start|>system\n" + "You are a helpful assistant.<|im_end|>\n"
                                         "<|im_start|>user\n" + f"{prompt}<|im_end|>\n" + "<|im_start|>assistant\n"
        )
    return prompt


async def extract_clean_output(response: str) -> str:
    # 提取 <|im_start|>assistant 到 <|im_end|> 之间的内容
    matches = re.findall(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", response, re.DOTALL)
    if matches:
        # 如果有多段输出，合并成单段
        return "\n".join(match.strip() for match in matches)
    else:
        # 如果没有匹配，移除所有 <|im_start|> 和 <|im_end|>
        return response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


def replace_you(text) -> str:
    """统一替换‘您’为‘你’，并添加异常处理"""
    try:
        if isinstance(text, str):
            # 如果是字符串，直接替换“您”为“你”
            return text.replace('您', '你')
        elif isinstance(text, dict):
            # 如果是字典，递归替换所有字符串值中的“您”为“你”
            return {key: replace_you(value) for key, value in text.items()}
        elif isinstance(text, list):
            # 如果是列表，递归替换列表中的所有字符串元素
            return [replace_you(item) for item in text]
        elif hasattr(text, 'model_dump_json'):  # 针对自定义对象，如 AigcFunctionsResponse
            # 如果是 AigcFunctionsResponse 类型，替换其中的 message 字段
            text.message = replace_you(text.message)
            return text.model_dump_json(exclude_unset=False)  # 返回处理后的 JSON 字符串
        return text  # 如果是其他类型，则不做处理，直接返回
    except Exception as e:
        # 处理所有可能的异常，避免报错
        return f"替换过程中发生错误: {str(e)}"


async def response_generator(response, error: bool = False) -> AsyncGenerator:
    """异步生成器
    处理`openai.AsyncStream`
    """
    if not error:
        async for chunk in response:
            # if chunk.object == "text_completion":
            if not chunk.object or "chat" not in chunk.object:
                content = chunk.choices[0].text
            else:
                content = chunk.choices[0].delta.content
            if content:
                # 替换“您” -> “你”
                content = replace_you(content)  # 在这里调用统一替换方法
                chunk_resp = AigcFunctionsResponse(message=content, code=200, end=False)
                yield f"data: {chunk_resp.model_dump_json(exclude_unset=False)}\n\n"
        chunk_resp = AigcFunctionsResponse(message="", code=200, end=True)
    else:
        chunk_resp = AigcFunctionsResponse(message=response, code=601, end=True)
        chunk_resp.message = replace_you(response)  # 错误消息替换
    yield f"data: {chunk_resp.model_dump_json(exclude_unset=False)}\n\n"


async def construct_naive_response_generator(response: str) -> AsyncGenerator:
    """异步生成器
    处理`openai.AsyncStream`
    """
    for index, content in enumerate(response):
        choice_chunk = CompletionResponseStreamChoice(index=index, text=content)
        repsonse = CompletionStreamResponse(model="", choices=[choice_chunk])
        yield repsonse


def build_aigc_functions_response(ret):
    if isinstance(ret, str):
        return Response(ret, media_type="application/json")
    elif isinstance(ret, Union[Generator, AsyncGenerator]):
        return StreamingResponse(ret, media_type="text/event-stream")


async def async_accept_param_purge(request: Request):
    p = await request.json()
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parents[1] / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        """
        remove original route from app
        """
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


def download_from_oss(filepath: str = "oss path", save_path: str = "local save path"):
    oss_cfg = load_yaml(Path("config", "aliyun_lk_oss.yaml"))
    auth = oss2.Auth(oss_cfg["OSS_ACCESS_KEY_ID"], oss_cfg["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, oss_cfg["OSS_REGION"], oss_cfg["OSS_BUCKET_NAME"])
    logger.info(f"download {filepath} starting")
    bucket.get_object_to_file(filepath, save_path)
    logger.info(f"download {filepath} finished")


async def parse_examination_plan(content):
    """将字符串解析为 JSON 对象，如果解析失败则返回一个空列表"""
    try:
        # 检查 content 是否为字符串
        if isinstance(content, str):
            # 尝试将单引号替换为双引号
            content = content.replace("'", '"')
            # 解析JSON字符串
            examination_plan = json.loads(content)
        else:
            # 如果 content 不是字符串，直接返回它
            examination_plan = content
        return examination_plan
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []


def calculate_bmi(weight: float, height: float) -> float:
    # bmi计算
    return round(weight / ((height / 100) ** 2), 1)


def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
    # 基础代谢率计算
    if gender == "男":
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == "女":
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender")


def parse_measurement(value_str: str, measure_type: str) -> float:
    """解析测量值字符串，支持体重和身高"""
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


async def format_historical_meal_plans_v2(historical_meal_plans: list) -> str:
    """
    将历史食谱转换为指定格式的字符串

    参数:
        historical_meal_plans (list): 包含历史食谱的列表

    返回:
        str: 格式化的历史食谱字符串
    """
    if not historical_meal_plans:
        raise ValueError("历史食谱数据为空")

    formatted_output = ""

    for day in historical_meal_plans:
        date = day.get("date")
        if not date:
            continue
        formatted_output += f"{date}：\n"

        meals = day.get("meals", {})
        for meal_time, foods in meals.items():
            if not foods:
                continue
            formatted_output += f"{meal_time}：\n"
            for food in foods:
                formatted_output += f"{food}\n"

    return formatted_output.strip()


def async_clock(func):
    @wraps(func)
    async def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print(f'[{elapsed:.8f}s] {name}({arg_str}) -> {result}')
        return result

    return clocked


async def convert_meal_plan_to_text(meal_plan_data: List[Dict[str, List[str]]]) -> str:
    """将餐次、食物名称的字典结构转换为文本形式"""
    formatted_text = ""
    for meal in meal_plan_data:
        meal_name = meal["meal"]
        foods = meal["foods"]
        formatted_text += f"{meal_name}:\n"
        for food in foods:
            formatted_text += f"  - {food}\n"
    return formatted_text.strip()


def parse_height(height: str) -> float:
    """解析身高，将其标准化为米为单位"""
    if isinstance(height, (int, float)):
        return height / 100.0 if height > 10 else height
    elif isinstance(height, str):
        height = height.strip().lower()
        if 'cm' in height:
            return float(height.replace('cm', '').strip()) / 100.0
        elif 'm' in height:
            return float(height.replace('m', '').strip())
        else:
            value = float(height)
            return value / 100.0 if value > 10 else value
    raise ValueError("无法解析的身高格式")


async def calculate_standard_weight(height: str, gender: str) -> float:
    """计算标准体重"""
    height_value = parse_height(height) * 100.0  # 转换为厘米
    if gender == "男":
        return (height_value - 100) * 0.9
    elif gender == "女":
        return (height_value - 100) * 0.9 - 2.5
    return None


def calculate_standard_body_fat_rate(gender: str) -> str:
    """计算标准体脂率"""
    if gender == "男":
        return "10%-20%"
    elif gender == "女":
        return "15%-25%"
    return None


async def calculate_and_format_diet_plan(diet_plan_standards: dict) -> str:
    """
    根据饮食方案标准计算推荐的三大产能营养素克数和推荐餐次及每餐的能量，并格式化输出结果。

    参数:
        diet_plan_standards (dict): 包含饮食方案标准的字典。

    返回:
        str: 格式化的输出结果。
    """
    # 获取每日推荐摄入热量值
    calories = diet_plan_standards.get("recommended_daily_caloric_intake", {}).get("calories")
    if not calories:
        raise ValueError("缺少推荐每日饮食摄入热量值")

    macronutrients = diet_plan_standards.get("recommended_macronutrient_grams", [])
    meals = diet_plan_standards.get("recommended_meal_energy", [])
    if not macronutrients or not meals:
        raise ValueError("缺少推荐三大产能营养素或推荐餐次及每餐能量的数据")

    # 计算三大产能营养素的摄入量范围
    def calculate_macronutrient_range(min_ratio, max_ratio, divisor):
        min_value = (calories * min_ratio) / divisor
        max_value = (calories * max_ratio) / divisor
        return round(min_value), round(max_value)

    macronutrient_values = {}
    for item in macronutrients:
        nutrient = item["nutrient"]
        min_ratio = item.get("min_energy_ratio", 0)
        max_ratio = item.get("max_energy_ratio", 0)
        divisor = 4 if nutrient != "脂肪" else 9
        macronutrient_values[nutrient] = calculate_macronutrient_range(min_ratio, max_ratio, divisor)

    # 计算每餐的能量值
    meal_energy_values = {}
    meal_ratios = {}

    breakfast_ratio = next(meal["min_energy_ratio"] for meal in meals if meal["meal_name"] == "早餐")
    lunch_ratio = next(meal["max_energy_ratio"] for meal in meals if meal["meal_name"] == "午餐")

    # 计算所有“加餐”相关的餐次比例
    snack_ratios = [meal["max_energy_ratio"] for meal in meals if "加餐" in meal["meal_name"]]
    total_snack_ratio = sum(snack_ratios)

    remaining_ratio = 1.0 - (breakfast_ratio + lunch_ratio + total_snack_ratio)
    dinner_ratio = max(0, remaining_ratio)  # 确保不会为负值

    meal_ratios["早餐"] = breakfast_ratio
    meal_ratios["午餐"] = lunch_ratio
    meal_ratios["晚餐"] = dinner_ratio

    # 处理所有“加餐”相关的餐次
    for meal in meals:
        if "加餐" in meal["meal_name"]:
            meal_ratios[meal["meal_name"]] = meal["max_energy_ratio"]

    for meal_name, ratio in meal_ratios.items():
        meal_energy_values[meal_name] = round(calories * ratio)

    # 组装格式化输出
    output = "# 推荐餐次及热量值\n"
    for meal_name, kcal in meal_energy_values.items():
        output += f"{meal_name}：{kcal}kcal\n"

    output += "# 推荐一日总的三大产能营养素摄入量\n"
    output += f"碳水化合物：{macronutrient_values['碳水化合物'][0]}g-{macronutrient_values['碳水化合物'][1]}g\n"
    output += f"蛋白质：{macronutrient_values['蛋白质'][0]}g-{macronutrient_values['蛋白质'][1]}g\n"
    output += f"脂肪：{macronutrient_values['脂肪'][0]}g-{macronutrient_values['脂肪'][1]}g\n"

    return output


async def format_historical_meal_plans(historical_meal_plans: list) -> str:
    """
    将历史食谱转换为指定格式的字符串，仅保留最近三天的数据。
    如果数据少于三天，则返回所有可用数据。

    参数:
        historical_meal_plans (list): 包含历史食谱的列表

    返回:
        str: 格式化的历史食谱字符串
    """
    if not historical_meal_plans:
        return "历史食谱数据为空"

    # 按日期降序排序，最多取最近三天，如果少于三天则全部返回
    historical_meal_plans = sorted(historical_meal_plans, key=lambda x: x.get("date", ""), reverse=True)[
                            :min(3, len(historical_meal_plans))]

    formatted_output = ""

    for day in historical_meal_plans:
        date = day.get("date")
        if not date:
            continue
        formatted_output += f"{date}：\n"

        meals = day.get("meals", {})
        for meal_time, foods in meals.items():
            if not foods:
                continue
            formatted_output += f"{meal_time}：{'、'.join(foods)}\n"

    return formatted_output.strip()


class WeatherServiceError(Exception):
    """自定义天气服务异常"""
    pass


def get_weather_info(config: Dict[str, str], city: str) -> Optional[str]:
    """获取天气信息，增加错误处理、超时控制和日志"""
    try:
        required_keys = ['key', 'weather_base_url', 'geo_base_url', 'air_quality_now_url']
        if not all(key in config for key in required_keys):
            logger.error("Missing required configuration keys")
            raise WeatherServiceError("Invalid configuration")

        api_key = config['key']
        weather_base_url = config['weather_base_url']
        geoapi_url = config['geo_base_url']
        air_quality_url = config['air_quality_now_url']

        logger.info(f"Getting weather info for city: {city}")
        city_id = get_city_id(city, geoapi_url, api_key)
        if not city_id:
            logger.error(f"Could not find city ID for {city}")
            return None

        with timing_logger("weather_api_request"):
            url = f"{weather_base_url}?key={api_key}&location={city_id}"
            data = make_http_request(url)

        if not data.get('daily'):
            logger.error("Weather data not found in response")
            return None

        today_weather = data['daily'][0]

        required_weather_fields = [
            'textDay', 'tempMax', 'tempMin', 'windScaleDay',
            'uvIndex'
        ]

        if not all(field in today_weather for field in required_weather_fields):
            logger.error("Missing required weather fields in response")
            return None

        def get_uv_level(uvIndex):
            uvIndex = uvIndex.strip()  # 去除两端的空格或换行符
            try:
                uvIndexInt = int(uvIndex)  # 尝试转换为整数
            except ValueError:  # 如果转换失败，说明uvIndex不是有效的数字
                return None

            if uvIndexInt in [0, 1, 2]:
                return '最弱'
            elif uvIndexInt in [3, 4]:
                return '较弱'
            elif uvIndexInt in [5, 6]:
                return '中等'
            elif uvIndexInt in [7, 8, 9]:
                return '较强'
            elif uvIndexInt >= 10:
                return '最强'
            else:
                return None

        # 获取空气质量信息
        air_quality_category = get_air_quality_info(air_quality_url, city_id, api_key)

        formatted_weather = (
            f"今日{city}天气{today_weather['textDay']}，"
            f"最高温度{today_weather['tempMax']}度，"
            f"最低{today_weather['tempMin']}度，"
            f"风力{today_weather['windScaleDay']}级，"
            f"紫外线强度{get_uv_level(today_weather['uvIndex'])}，"
            f"空气质量{air_quality_category}。"
        )

        logger.info(f"Successfully got weather info for {city}")
        return formatted_weather

    except WeatherServiceError as e:
        logger.error(f"Weather service error: {str(e)}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in get_weather_info: {str(e)}")
        return None


def get_city_id(city: str, geoapi_url: str, api_key: str) -> Optional[str]:
    """获取城市ID，添加错误处理和日志"""
    try:
        with timing_logger("get_city_id"):
            url = f"{geoapi_url}?key={api_key}&location={city}"
            data = make_http_request(url)

            # 检查返回的数据中是否存在 'location' 且它是一个非空列表
            if not data.get('location') or len(data['location']) == 0:
                logger.warning(f"City '{city}' not found or 'location' is empty in response.")
                return None

            city_id = data['location'][0].get('id')
            logger.info(f"Successfully got city ID for {city}: {city_id}")
            return city_id

    except WeatherServiceError as e:
        logger.error(f"Failed to get city ID: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while getting city ID: {str(e)}")
        return None


def get_air_quality_info(air_quality_url: str, city_id: str, api_key: str) -> Optional[str]:
    """获取空气质量信息，添加错误处理和日志"""
    try:
        with timing_logger("get_air_quality_info"):
            url = f"{air_quality_url}?key={api_key}&location={city_id}"
            data = make_http_request(url)

            # 检查返回的数据中是否存在 'now' 且包含 'category'
            if not data.get('now') or 'category' not in data['now']:
                logger.warning("Air quality data not found or 'category' is missing in response.")
                return None

            air_quality_category = data['now']['category']
            logger.info(f"Successfully got air quality category: {air_quality_category}")
            return air_quality_category

    except WeatherServiceError as e:
        logger.error(f"Failed to get air quality info: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while getting air quality info: {str(e)}")
        return None


@contextmanager
def timing_logger(operation: str):
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation} took {duration:.2f} seconds")


def make_http_request(url: str, timeout: int = 5) -> Dict[str, Any]:
    try:
        logger.info(f"Making HTTP request to {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # 检查请求是否成功

        # 如果状态码不是200，直接调用 handle_api_error 进行处理
        if response.status_code != 200:
            data = response.json()  # 获取错误响应的数据
            logger.error(f"HTTP request to {url} failed with status code {response.status_code}")
            handle_api_error(response.status_code, data, logger)  # 调用 handle_api_error 处理错误
            return {}

        return response.json()

    except requests.exceptions.Timeout:
        logger.error(f"Request timeout: {url}")
        raise WeatherServiceError(f"Request timeout: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise WeatherServiceError(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise WeatherServiceError(f"JSON decode error: {str(e)}")


def handle_api_error(status_code, data, logger):
    """
    根据 API 返回的状态码和错误信息进行处理
    """
    if status_code == 400:
        # Bad request - INVALID PARAMETERS or MISSING PARAMETERS
        if 'error' in data:
            error_title = data['error'].get('title', 'Unknown error')
            error_detail = data['error'].get('detail', 'No details available')
            logger.error(f"Error 400 - {error_title}: {error_detail}")
        else:
            logger.error("Bad request: The request may contain invalid parameters or missing required parameters.")

    elif status_code == 401:
        # Unauthorized - Authentication failure
        logger.error("Error 401 - Unauthorized: Authentication failed, possibly due to incorrect API key or token.")

    elif status_code == 403:
        # Forbidden - Access denied
        logger.error(
            "Error 403 - Forbidden: Access denied, possibly due to insufficient balance, access restrictions, or invalid credentials.")

    elif status_code == 404:
        # Not Found - The requested data/resource was not found
        logger.error("Error 404 - Not Found: The requested data or resource could not be found.")

    elif status_code == 429:
        # Too Many Requests - Rate limit exceeded
        logger.error("Error 429 - Too Many Requests: Rate limit exceeded, please try again later.")

    elif status_code == 500:
        # Internal Server Error - Unknown server error
        logger.error(
            "Error 500 - Internal Server Error: An unknown error occurred on the server. Please contact the API provider.")

    else:
        # Unknown error
        logger.error(f"Unknown error, HTTP Status Code: {status_code}")


async def determine_recent_solar_terms(date_str: str = None):
    """
    确定传入日期的当前节气或下一个节气。

    Args:
        date_str (str, optional): 日期字符串，格式为 'YYYY-MM-DD'，默认为当前日期。

    Returns:
        str: 节气的名称和日期，例如 "2024-11-07 立冬"，如果失败则返回 "无"。
    """
    try:
        # 如果未传入日期，使用当前时间
        if date_str is None:
            date = datetime.now()
        else:
            date = datetime.strptime(date_str, "%Y-%m-%d")

        # 转换为农历对象
        lunar = Lunar.fromDate(date)

        # 获取所有节气
        jieqi_list = lunar.getJieQiTable()

        # 遍历节气，判断当前日期是否在某个节气范围内
        for jieqi_name, solar_date in jieqi_list.items():
            start_date = datetime.strptime(solar_date.toYmd(), "%Y-%m-%d")
            next_index = list(jieqi_list.keys()).index(jieqi_name) + 1
            next_jieqi_name = list(jieqi_list.keys())[next_index % len(jieqi_list)]
            next_solar_date = datetime.strptime(jieqi_list[next_jieqi_name].toYmd(), "%Y-%m-%d")

            # 检查是否在当前节气范围内
            if start_date <= date < next_solar_date:
                return f"{start_date.strftime('%Y-%m-%d')} {jieqi_name}"

        # 如果未找到当前节气，返回下一个节气
        next_jieqi = lunar.getNextJieQi(True)
        if next_jieqi:
            next_solar_date = next_jieqi.getSolar()
            return f"{next_solar_date.toYmd()} {next_jieqi.getName()}"

        return "无"

    except Exception as e:
        # 捕获异常并返回 "无"
        print(f"Error: {e}")
        return "无"


def determine_recent_solar_terms_sanji():
    date = datetime.now()
    lunar = Lunar.fromDate(date)

    # 获取当天的节气
    current_jieqi = lunar.getCurrentJieQi()
    if current_jieqi:
        return f"{current_jieqi.getSolar().toYmd()} {current_jieqi.getName()}"

    # 获取下一个节气
    next_jieqi = lunar.getNextJieQi(True)
    pre_jieqi = lunar.getPrevJieQi(True)
    if next_jieqi:
        next_jieqi_solar = next_jieqi.getSolar()

        next_jieqi_date = datetime(next_jieqi_solar.getYear(), next_jieqi_solar.getMonth(), next_jieqi_solar.getDay())
        delta_days = (next_jieqi_date - date).days

        if pre_jieqi:
            pre_jieqi_solar = pre_jieqi.getSolar()
            pre_jieqi_date = datetime(pre_jieqi_solar.getYear(), pre_jieqi_solar.getMonth(), pre_jieqi_solar.getDay())
            pre_delta_days = (date - pre_jieqi_date).days
            if pre_delta_days < delta_days:
                return f"{pre_jieqi_date.strftime('%Y-%m-%d')} {pre_jieqi.getName()}"
        return f"{next_jieqi_date.strftime('%Y-%m-%d')} {next_jieqi.getName()}"

    pre_jieqi_solar = pre_jieqi.getSolar()
    pre_jieqi_date = datetime(pre_jieqi_solar.getYear(), pre_jieqi_solar.getMonth(), pre_jieqi_solar.getDay())
    return pre_jieqi_date


async def get_festivals_and_other_festivals():
    # 获取当天的节日和纪念日
    date = datetime.now()
    year, month, day = date.year, date.month, date.day
    solar = Solar.fromYmd(year, month, day)
    festivals = solar.getFestivals()
    return ','.join(festivals) if festivals else None


async def generate_daily_schedule(schedule):
    """
    生成当日剩余日程的格式化字符串
    schedule: list of dicts, 每个字典包含时间和事件，例如：[{'time': '13:00', 'event': '吃火锅'}, {'time': '16:00', 'event': '复诊'}, {'time': '20:00', 'event': '服药'}]
    """
    schedule_str = ""
    for item in schedule:
        schedule_str += f"{item['time']} {item['event']}\n"
    return schedule_str


async def generate_key_indicators(data):
    """
    生成关键指标的表格字符串
    data: list of dicts, 每个字典包含标准格式的日期时间、收缩压、舒张压和单位，例如：[{'datetime': '2024-07-20 09:08:25', 'sbp': 116, 'dbp': 82, 'unit': 'mmHg'}]
    """
    table_str = ""
    if data:
        table_str += "| date       | time       | 收缩压 | 舒张压 | 单位      |\n"
        table_str += "|------------|------------|-------|-------|-----------|\n"
        for item in data:
            datetime_str = item['datetime']
            date_str, time_str = datetime_str.split()
            date_str = date_str.replace('-', '/')  # 将日期格式转换为 yyyy/mm/dd
            table_str += f"| {date_str}  | {time_str}   | {item['sbp']}   | {item['dbp']}   | {item['unit']} |\n"
    return table_str


async def parse_generic_content(content):
    """异步解析内容为 JSON 对象，如果解析失败则返回一个空列表"""

    if isinstance(content, openai.AsyncStream):
        return content

    errors = []  # 用于存储所有捕获的异常信息

    try:
        # 直接尝试解析为 JSON
        return json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Direct JSON parsing failed: {e}")

    try:
        # 将单引号替换为双引号
        corrected_content = content.replace("'", '"')

        # 解析 JSON
        parsed_data = json.loads(corrected_content)
        return parsed_data
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed after replacing single quotes: {e}")

    try:
        # 移除换行符
        content = content.replace('\n', '')
        # 在减号后面添加空格
        content = re.sub(r'(\d)-(\d)', r'\1 - \2', content)
        # 再次尝试解析
        return json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed after removing newlines and adjusting hyphens: {e}")

    try:
        # 处理JSON代码块
        content_json = re.findall(r"```json(.*?)```", content, re.DOTALL)
        if content_json:
            return json.loads(content_json[0])
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed for JSON code block: {e}")

    try:
        # 处理Python代码块
        content_python = re.findall(r"```python(.*?)```", content, re.DOTALL)
        if content_python:
            # 假设Python代码块中包含的是JSON字符串
            # 获取第一个匹配项并去掉首尾的空白字符
            python_code = content_python[0].strip()
            # 将单引号转换为双引号
            corrected_content = python_code.replace("'", '"')
            return json.loads(corrected_content)
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed for Python code block: {e}")

    try:
        # 处理自由文本中嵌入的JSON数据
        json_match = re.search(r'{.*}', content, re.DOTALL)
        if json_match:
            json_data = json_match.group(0)
            return json.loads(json_data)
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed for embedded JSON in free text: {e}")

    try:
        # 处理自由文本中嵌入的JSON数据
        json_match = re.search(r'```\s*(.*)```', content, re.DOTALL)
        if json_match:
            json_data = json_match.group(1).strip()
            return json.loads(json_data)
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed for embedded code block: {e}")

    try:
        # 移除三个反引号
        content = re.sub(r'^```|```$', '', content, flags=re.MULTILINE)
        # 尝试解析去除了三个反引号的内容
        return json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed after removing triple backticks: {e}")

    try:
        # 尝试将 content 解析为一个 JSON 对象
        content_obj = json.loads(content)
        return content_obj
    except json.JSONDecodeError as e:
        errors.append(f"Final direct JSON parsing failed: {e}")

    try:
        # 使用正则表达式精确移除最外层的反引号
        content = re.sub(r'^```(?P<content>.*)```$', r'\g<content>', content, flags=re.DOTALL)

        # 移除最后一个逗号（如果有）
        content = re.sub(r',\s*\]', r']', content)

        # 尝试解析去除了反引号和多余逗号的内容
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        # 记录错误日志
        errors.append(f"Parsing failed after removing extra commas and backticks: {e}")

    try:
        # 使用正则表达式来查找并提取JSON数据
        json_match = re.search(r'\[(.*?)\]', content, re.DOTALL)

        if json_match:
            # 提取JSON字符串
            json_data_str = json_match.group(0)

            # 清理JSON数据，移除可能的换行符和多余空格
            cleaned_json_data = json_data_str.replace('\n', '').replace('\t', '').strip()

            # 尝试解析JSON数据
            parsed_data = json.loads(cleaned_json_data)

            return parsed_data
        # 如果没有找到匹配的JSON数据，抛出异常
        pass
    except json.JSONDecodeError as e:
        errors.append(f"Parsing failed for JSON array in content: {e}")

    try:
        json_string = content
        # 使用正则表达式找到所有的数学表达式
        expressions = re.findall(r'\d+\s*[+\-*/]\s*\d+', json_string)

        # 如果没有找到任何表达式，则直接解析JSON
        if not expressions:
            return json.loads(json_string)

        # 计算表达式并将结果放回原字符串
        for expr in expressions:
            # 使用eval()计算表达式的值
            # 注意：eval()可能有安全风险，这里假设表达式是安全的
            result = str(eval(expr))
            # 替换字符串中的表达式为计算后的结果
            json_string = json_string.replace(expr, result)

        # 解析修正后的JSON字符串
        parsed_json = json.loads(json_string)
        return parsed_json
    except Exception as e:
        errors.append(f"Parsing failed after evaluating expressions: {e}")

    # 如果所有尝试都失败，返回空列表
    # 如果所有尝试都失败，并且返回的内容为空，则记录错误日志
    logger.error(f"Failed to parse content. Errors encountered: {errors}")

    return []


async def handle_calories(content: dict, **kwargs) -> dict:
    """
    如果 'calories' 不是数字类型，将其设置为 0。
    如果 content 不是字典，返回一个包含错误信息的字典。
    处理字典中的 'calories' 字段，并更新 'food_name' 和 'quantity' 字段。
    """
    if not isinstance(content, dict):
        return {"error": "Invalid content format"}
    if not isinstance(content.get("calories"), (int, float)):
        content["calories"] = 0
    # 更新 'food_name' 字段
    if "food_name" in kwargs and kwargs["food_name"]:
        content["food_name"] = kwargs["food_name"]

    # 更新 'quantity' 字段
    if "food_quantity" in kwargs and kwargs["food_quantity"]:
        content["quantity"] = kwargs["food_quantity"]
    return content


async def check_aigc_functions_body_fat_weight_management_consultation(params: dict
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


async def check_and_calculate_bmr(user_profile: dict) -> float:
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

    weight = parse_measurement(user_profile["weight"], "weight")
    height = parse_measurement(user_profile["height"], "height")
    age = int(user_profile["age"])
    gender = user_profile["gender"]

    return calculate_bmr(weight, height, age, gender)


async def get_highest_data_per_day(blood_pressure_data: List[Dict]) -> List[Dict]:
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


async def check_consecutive_days(blood_pressure_data: List[Dict]) -> bool:
    """
    判断给定的血压数据是否是连续5天的数据。

    Args:
        blood_pressure_data: 包含日期和血压数据的字典列表，每天可能有多条数据。

    Returns:
        bool: 如果数据是连续5天的，返回True；否则返回False。
    """
    # 获取每一天最近的一条数据
    blood_pressure_data = await get_highest_data_per_day(blood_pressure_data)

    # 提取日期并排序
    dates = sorted(set([entry['date'].split(' ')[0] for entry in blood_pressure_data]))
    if len(dates) < 5:
        return False

    for i in range(4):
        if (datetime.strptime(dates[i + 1], '%Y-%m-%d') - datetime.strptime(dates[i], '%Y-%m-%d')).days != 1:
            return False

    return True


async def run_in_executor(func, *args, **kwargs):
    """
    在线程池中执行同步函数。

    Args:
        func (callable): 要执行的同步函数。
        *args: 传递给函数的非关键字参数。
        **kwargs: 传递给函数的关键字参数。

    Returns:
        执行结果。
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def wrap_content_for_frontend(content_text, content_type="MARKDOWN", item_title=""):
    """
    将内容封装为前端需要的结构，支持不同类型的内容
    Args:
        content_text (str | list): 需要封装的内容，可以是字符串或字符串列表
        content_type (str): 内容类型，例如 MARKDOWN、HTML、TEXT
        item_title (str): 给前端的唯一标识符
    Returns:
        dict: 包装后的前端结构
    """
    # 检查参数合法性
    if not isinstance(content_text, (str, list)):
        raise ValueError("content_text must be a string or a list of strings.")
    if not isinstance(item_title, str):
        raise ValueError("item_title must be a string.")
    if content_type not in ["MARKDOWN", "HTML", "TEXT"]:
        raise ValueError("content_type must be one of ['MARKDOWN', 'HTML', 'TEXT'].")

    # 如果是字符串列表，将其转化为多个内容
    if isinstance(content_text, list):
        item_contents = [{"text": text} for text in content_text]
    else:
        item_contents = [{"text": content_text}]

    return {
        "text": "",
        "payload": [
            {
                "itemTitle": item_title,
                "itemType": content_type,
                "itemContents": item_contents
            }
        ]
    }


async def assemble_frontend_format_with_fixed_items(overview: dict) -> dict:
    """
    将 overview 数据组装为前端所需的格式，并固定第一天和最后一天的特殊活动。
    Args:
        overview (dict): 包含行程数据的 overview 字段
    Returns:
        dict: 返回组装后的前端数据结构
    """

    # 使用 JSON 深拷贝
    copied_overview = json.loads(json.dumps(overview))

    def format_date(date_str):
        """
        将日期转换为“10月27日”的格式
        :param date_str: 原始日期字符串，格式为 YYYY-MM-DD
        :return: 转换后的日期字符串
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%m月%d日")
        except ValueError:
            return date_str  # 如果日期格式不对，直接返回原始值

    markdown_intro = {
        "itemTitle": "",
        "itemType": "MARKDOWN",
        "itemContents": [
            {
                "text": copied_overview.get("intro", "")
            }
        ]
    }

    # 准备表格内容
    table_data = []
    itinerary = copied_overview.get("itinerary", [])

    # 如果行程存在，调整第一天和最后一天的固定活动
    if itinerary:
        # 固定第一天上午的活动为“到达和入住”，保留原地点
        first_day = itinerary[0]
        if "time_slots" in first_day and first_day["time_slots"]:
            first_time_slot = first_day["time_slots"][0]
            if "activities" in first_time_slot and first_time_slot["activities"]:
                first_time_slot["activities"][0]["name"] = "到达和入住"

        # 固定最后一天的最后一个活动为“返程”
        last_day = itinerary[-1]
        if "time_slots" in last_day and last_day["time_slots"]:
            last_time_slot = last_day["time_slots"][-1]
            if "activities" in last_time_slot and last_time_slot["activities"]:
                last_time_slot["activities"].append({
                    "name": "返程",
                    "location": "",
                    "description": "结束此次愉快的旅程，踏上归途。",
                    "external_id": None
                })

    # 遍历行程数据并构建表格数据
    for day_data in itinerary:
        formatted_date = format_date(day_data.get("date", ""))
        date_display = f"{formatted_date}\n（第{day_data.get('day')}天）"
        for time_slot in day_data.get("time_slots", []):
            period = time_slot.get("period", "")
            for activity in time_slot.get("activities", []):
                table_data.append({
                    "value1": date_display,
                    "value2": period,
                    "value3": activity.get("name", ""),
                    "value4": activity.get("location", ""),
                })

    table_structure = {
        "itemTitle": "",
        "itemType": "TABLE",
        "itemContents": [
            {
                "tableOptions": [
                    {
                        "label": "日期",
                        "prop": "value1",
                        "min-width": 80,
                        "mergeRows": ["value1"],
                        "fixed": False
                    },
                    {
                        "label": "时间",
                        "prop": "value2",
                        "min-width": 50,
                        "mergeRows": ["value1", "value2"]
                    },
                    {
                        "label": "活动",
                        "prop": "value3",
                        "min-width": 140
                    },
                    {
                        "label": "地点",
                        "prop": "value4",
                        "min-width": 140
                    }
                ],
                "tableData": table_data
            }
        ]
    }

    # 准备温馨提示和 closing 部分
    tips_and_closing = {
        "itemTitle": "",
        "itemType": "MARKDOWN",
        "itemContents": [
            {
                "text": f"## 温馨小贴士：\n- " + "\n- ".join(copied_overview.get("tips", [])) +
                        f"\n\n{copied_overview.get('closing', '')}"
            }
        ]
    }

    # 返回完整的结构
    return {
        "text": "",
        "payload": [
            markdown_intro,
            table_structure,
            tips_and_closing
        ]
    }


def log_with_source(func):
    """异步装饰器，用于根据 source 动态绑定日志"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        source = kwargs.get("source", "user")
        # logger.info(f"Source for logging: {source}")  # 验证 source 是否正确
        if source == "monitor":
            logger_with_source = logger.bind(source=source)  # 动态绑定日志上下文
            logger_with_source.info("Logger is bound with source=monitor")
        else:
            logger_with_source = logger

        kwargs["logger"] = logger_with_source  # 确保绑定后的 logger 传递
        return await func(*args, **kwargs)

    return wrapper


async def determine_weight_status(user_profile, bmi_value):
    age = user_profile["age"]
    if 18 <= age < 65:
        if bmi_value < 18.5:
            return "身材偏瘦", "偏低", "增肌"
        elif 18.5 <= bmi_value < 24:
            return "属于标准体重", "正常", "保持体重"
        elif 24 <= bmi_value < 28:
            return "体重超重", "偏高", "减脂"
        else:
            return "属于肥胖状态", "偏高", "减脂"
    else:
        if bmi_value < 20:
            return "身材偏瘦", "偏低", "增肌"
        elif 20 <= bmi_value < 26.9:
            return "属于标准体重", "正常", "保持体重"
        elif 26.9 <= bmi_value < 28:
            return "体重超重", "偏高", "减脂"
        else:
            return "属于肥胖状态", "偏高", "减脂"


async def determine_body_fat_status(gender: str, body_fat_rate: float):
    """确定体脂率状态和目标"""
    if gender == "男":
        if body_fat_rate < 10:
            return "偏低状态", "增重"
        elif 10 <= body_fat_rate < 20:
            return "正常范围", "保持体重"
        elif 20 <= body_fat_rate < 25:
            return "偏高状态", "减脂"
        else:
            return "肥胖状态", "减脂"
    elif gender == "女":
        if body_fat_rate < 15:
            return "偏低状态", "增重"
        elif 15 <= body_fat_rate < 25:
            return "正常范围", "保持体重"
        elif 25 <= body_fat_rate < 30:
            return "偏高状态", "减脂"
        else:
            return "肥胖状态", "减脂"


async def truncate_to_limit(text: str, limit: int) -> str:
    """
    截取文本至指定字符限制，若字符数超出限制则保留最后一个标点符号，或在句尾加上句号。
    :param text: 原始文本
    :param limit: 字符限制，默认为20个字符
    :return: 处理后的文本
    """
    # 去掉换行符
    text = text.replace("\n", "")

    # 如果文本长度小于等于限制，直接返回
    if len(text) <= limit:
        return text if text.endswith("。") else text + "。"

    # 截取前limit个字符
    truncated = text[:limit]

    # 在截断文本中寻找最后一个标点符号
    last_punctuation = max(truncated.rfind(p) for p in "。，！？")

    # 如果找到标点符号，将该标点转换为句号
    if last_punctuation != -1:
        truncated = truncated[:last_punctuation + 1]
        # 如果标点不是句号，替换成句号
        if truncated[-1] not in "。":
            truncated = truncated[:-1] + "。"
    else:
        # 如果没有标点符号，直接在末尾加句号
        truncated = truncated.rstrip() + "。"

    return truncated


async def filter_user_profile(user_profile):
    # 只保留需要的字段
    required_fields = [
        "age", "gender", "height", "weight", "allergic_history", "surgery_history",
        "chinese_medicine_disease", "chinese_medicine_symptom", "current_diseases",
        "severity", "traditional_chinese_medicine_constitution", "constitution_symptom"
    ]

    # 创建一个新的字典，只保留所需字段
    filtered_profile = {key: user_profile.get(key) for key in required_fields if key in user_profile}

    return filtered_profile


def prepare_question_list(data):
    """
    准备问题列表，将 `question` 和 `similar_questions` 合并，形成一个换行格式的字符串列表。
    同时排除 `guess_you_want` 中的内容被包含在其他问题或相似问题中的条目，
    或者当 `guess_you_want` 为空时，检查主问题是否已存在。

    参数:
        data (list): 原始 JSON 数据

    返回:
        str: 格式化为换行字符串的所有问题列表
    """
    # 构建全局的问题集合
    all_questions = set()  # 使用集合来自动去重

    for item in data:
        # 获取主问题和相似问题的列表
        combined_questions = [item.get("question", "")]
        combined_questions.extend(item.get("similar_questions", []))

        # 将当前条目的问题加入到集合中
        all_questions.update(combined_questions)

    # 将集合转换为格式化的字符串
    formatted_list = "[\n"
    for question in sorted(all_questions):  # 排序以保持输出一致
        formatted_list += f'    "{question}",\n'
    formatted_list = formatted_list.rstrip(",\n")  # 去掉最后的逗号和换行符
    formatted_list += "\n]"

    return formatted_list

def glucose_type(time, glucose):
    # 血糖值分类
    glucose=float(glucose)
    if time == "1":
        t= '空腹血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='你血糖非常低，请立即补充含糖食物'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='你血糖较低，请尽快补充含糖食物'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏低，请及时给予处理建议。如果本周发生1-2次低血糖，就属于频繁低血糖，必要时与客户取得联系邀请复诊"""
        elif 3.9 <= glucose <=7:
            result = "血糖正常"
            content='血糖正常，请继续保持'
            agent_content=""
        elif 7.0 < glucose <= 13.9:
            result = "血糖控制高"
            content='今日空腹血糖值偏高，请遵医嘱，规律生活'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏高，请给予关注。"""
        else:
            result = "血糖控制高危"
            content = "今日空腹血糖非常高，请严格遵医嘱！"
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值非常高，请及时关注用户运动量、用药量、饮食量等变化，并进一步与患者沟通，给予改善建议。"""
    elif time == "2":
        t= '餐后2小时血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='餐后2小时血糖值非常低，请立即补充含糖食物。'
            agent_content=f"""你好，客户目前血糖为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='餐后2小时血糖值较低，请尽快补充含糖食物。'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏低，请及时与客户取得联系，给予处理建议。"""
        elif 3.9 <= glucose <=10:
            result = "血糖正常"
            content='血糖正常，请继续保持。'
            agent_content=""
        elif 10 < glucose <= 16.7:
            result = "血糖控制高"
            content='餐后2小时血糖值过高，请遵医嘱调整饮食与运动等生活方式。'
            agent_content=f"""你好，客户目前餐后2小时血糖为{glucose}mmol/L,血糖值偏高，请给予关注。"""
        else:
            result = "血糖控制高危"
            content = "今日餐后2小时血糖值非常高，请严格遵医嘱！"
            agent_content=f"""你好，客户目前餐后2小时血糖为{glucose}mmol/L,血糖值非常高，请及时关注用户运动量、用药量、饮食量等变化，并进一步与患者沟通，给予改善建议。"""
    else:
        t= '随机血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='随机血糖值非常低，请立即补充含糖食物。'
            agent_content=f"""你好，客户目前{t}为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='随机血糖值较低，请尽快补充含糖食物。'
            agent_content=f"""你好，客户目前{t}为{glucose}mmol/L,血糖值偏低，请及时与客户取得联系，给予处理建议。"""
        elif 3.9 <= glucose <=7:
            result = "血糖正常"
            content='血糖正常，请继续保持。'
            agent_content=""
        elif 7 < glucose <= 13.9:
            result = "血糖控制高"
            content='今日随机血糖值高，请减少饮食量，增加运动量。'
            agent_content=f"""你好，客户目前{t}为{glucose}mmol/L,血糖值偏高，请关注该用户近2日动态血糖变化。"""
        elif 13.9 < glucose < 16.7:
            result = "血糖控制中危"
            content='今日随机血糖值非常高，请严格遵医嘱！'
            agent_content=f"""你好，客户目前{t}为{glucose}mmol/L,血糖值较高，请关注该用户近2日动态血糖变化，必要时进一步与患者沟通，给予改善建议"""
        else:
            result = "血糖控制高危"
            content = "随机血糖值极高，请严格遵医嘱，积极控制血糖！"
            agent_content=f"""你好，客户目前{t}为{glucose}mmol/L,血糖值极高，请关注该用户近2日动态血糖变化，必要时进一步与患者沟通，给予改善建议"""
    return result,content,agent_content,t

def extract_glucose(recent_time,glucose_data):
    # 假设当前时间
    try:
        recent_time = datetime.strptime(recent_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        recent_time = datetime.now()

    # 将时间字符串转换为datetime对象
    for entry in glucose_data:
        entry["exam_time"] = datetime.strptime(entry["exam_time"], "%Y-%m-%d %H:%M:%S")

    # 按时间排序
    glucose_data.sort(key=lambda x: x["exam_time"])

    # 过滤出在rencent_time前3小时内的数据
    recent_3h_data = [entry for entry in glucose_data if entry["exam_time"] >= recent_time - timedelta(hours=3) and entry["exam_time"] <= recent_time]

    # 采样逻辑
    sampled_data = []
    time_threshold_2h = recent_time.replace(second=0, microsecond=0) - timedelta(hours=2)
    time_threshold_3h = recent_time.replace(second=0, microsecond=0) - timedelta(hours=3)

    # 近2h内，每5分钟取一个
    last_2h_samples = []
    current_time = time_threshold_2h
    while current_time < time_threshold_2h + timedelta(hours=2):
        for entry in recent_3h_data:
            if entry["exam_time"].replace(second=0, microsecond=0) == current_time:
                last_2h_samples.append(entry)
                break
        current_time += timedelta(minutes=5)

    # 近3h-2h内，每10分钟取一个
    last_3h_2h_samples = []
    current_time = time_threshold_3h
    while current_time < time_threshold_2h:
        for entry in recent_3h_data:
            if entry["exam_time"].replace(second=0, microsecond=0) == current_time:
                last_3h_2h_samples.append(entry)
                break
        current_time += timedelta(minutes=10)

    # 合并采样数据
    sampled_data.extend(last_2h_samples)
    sampled_data.extend(last_3h_2h_samples)

    # 找出波峰波谷
    def find_peaks_and_valleys(data):
        peaks = []
        valleys = []
        for i in range(1, len(data)-1):
            if data[i]["item_value"] > data[i-1]["item_value"] and data[i]["item_value"] > data[i+1]["item_value"]:
                peaks.append(("波峰数据",data[i]["exam_time"], data[i]["item_value"]))
            if data[i]["item_value"] < data[i-1]["item_value"] and data[i]["item_value"] < data[i+1]["item_value"]:
                valleys.append(("波谷数据",data[i]["exam_time"], data[i]["item_value"]))
        return peaks, valleys

    peaks, valleys = find_peaks_and_valleys(recent_3h_data)

    return sampled_data,peaks,valleys


def blood_pressure_type(time, glucose):
    # 血压值分类
    glucose=float(glucose)
    if time == "1":
        t= '空腹血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='你血糖非常低，请立即补充含糖食物'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='你血糖较低，请尽快补充含糖食物'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏低，请及时给予处理建议。如果本周发生1-2次低血糖，就属于频繁低血糖，必要时与客户取得联系邀请复诊"""
        elif 3.9 <= glucose <=7:
            result = "血糖正常"
            content='血糖正常，请继续保持'
            agent_content=""
        elif 7.0 < glucose <= 13.9:
            result = "血糖控制高"
            content='今日空腹血糖值偏高，请遵医嘱，规律生活'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏高，请给予关注。"""
        else:
            result = "血糖控制高危"
            content = "今日空腹血糖非常高，请严格遵医嘱！"
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值非常高，请及时关注用户运动量、用药量、饮食量等变化，并进一步与患者沟通，给予改善建议。"""
    elif time == "2" and time != "":
        t= '餐后2小时血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='餐后2小时血糖值非常低，请立即补充含糖食物。'
            agent_content=f"""你好，客户目前血糖为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='餐后2小时血糖值较低，请尽快补充含糖食物。'
            agent_content=f"""你好，客户目前空腹血糖为{glucose}mmol/L,血糖值偏低，请及时与客户取得联系，给予处理建议。"""
        elif 3.9 <= glucose <=10:
            result = "血糖正常"
            content='血糖正常，请继续保持。'
            agent_content=""
        elif 10 < glucose <= 16.7:
            result = "血糖控制高"
            content='餐后2小时血糖值过高，请遵医嘱调整饮食与运动等生活方式。'
            agent_content=f"""你好，客户目前餐后2小时血糖为{glucose}mmol/L,血糖值偏高，请给予关注。"""
        else:
            result = "血糖控制高危"
            content = "今日餐后2小时血糖值非常高，请严格遵医嘱！"
            agent_content=f"""你好，客户目前餐后2小时血糖为{glucose}mmol/L,血糖值非常高，请及时关注用户运动量、用药量、饮食量等变化，并进一步与患者沟通，给予改善建议。"""
    else:
        t= '随机血糖'
        if glucose < 3:
            result = "高危低血糖"
            content='随机血糖值非常低，请立即补充含糖食物。'
            agent_content=f"""你好，客户目前血糖为{glucose}mmol/L,血糖值非常低，请及时与客户取得联系，给予处理建议。"""
        elif 3 <= glucose < 3.9:
            result = "低血糖"
            content='随机血糖值较低，请尽快补充含糖食物。'
            agent_content=f"""你好，客户目前血糖为{glucose}mmol/L,血糖值偏低，请及时与客户取得联系，给予处理建议。"""
        elif 3.9 <= glucose <=7:
            result = "血糖正常"
            content='血糖正常，请继续保持。'
            agent_content=""
        elif 7 < glucose <= 13.9:
            result = "血糖控制高"
            content='今日随机血糖值高，请减少饮食量，增加运动量。'
            agent_content=f"""你好，客户目前随机血糖为{glucose}mmol/L,血糖值偏高，请关注该用户近2日动态血糖变化。"""
        elif 13.9 < glucose < 16.7:
            result = "血糖控制中危"
            content='今日随机血糖值非常高，请严格遵医嘱！'
            agent_content=f"""你好，客户目前随机血糖为{glucose}mmol/L,血糖值较高，请关注该用户近2日动态血糖变化，必要时进一步与患者沟通，给予改善建议"""
        else:
            result = "血糖控制高危"
            content = "随机血糖值极高，请严格遵医嘱，积极控制血糖！"
            agent_content=f"""你好，客户目前随机血糖为{glucose}mmol/L,血糖值极高，请关注该用户近2日动态血糖变化，必要时进一步与患者沟通，给予改善建议"""
    return result,content,agent_content,t

def count_tokens(text: str, tokenizer=None) -> int:
    return len(tokenizer.encode(text)) if text else 0

async def async_count_tokens(text: str, tokenizer=None) -> int:
    return len(tokenizer.encode(text)) if text else 0


# 把 list[dict] 格式的 message history 转成纯文本
def flatten_input_text(data):
    if isinstance(data, str):
        return data
    elif isinstance(data, list):
        return "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in data])
    else:
        return str(data)


async def monitor_interface(**kwargs):
    """
    通用接口监控函数，通过 **kwargs 传递参数。

    Args:
        **kwargs: 动态参数，包括以下必需字段：
            - interface_name (str): 接口名称。
            - user_id (str): 用户 ID。
            - session_id (str): 会话 ID。
            - request_input (Any): 请求输入。
            - response_output (Any): 响应输出。

    Returns:
        None: 仅用于记录和追踪。
    """
    # 校验必需参数
    required_keys = ["request_input", "response_output"]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required parameter: {key}")

    # logger.debug(kwargs)
    # 解构参数，使用 .get() 方法
    interface_name = kwargs.get("interface_name")
    start_time = kwargs.get("start_time")
    end_time = kwargs.get("end_time", time.time())
    tags = kwargs.get("tags")
    user_id = kwargs.get("user_id") or "default"
    session_id = kwargs.get("session_id") or "default"
    request_input = kwargs.get("request_input")
    response_output = kwargs.get("response_output")
    langfuse = kwargs.get("langfuse")
    release = kwargs.get("release", "v1.0.0")  # 可以设置默认值
    model = kwargs.get("model")  # 默认模型名称
    metadata = kwargs.get("metadata", {"description": f"Monitoring {interface_name}"})

    # 请求名称
    request_name = f"post_{interface_name}"

    # 响应名称
    response_name = f"resp_post_{interface_name}"

    # 初始化 Langfuse Trace
    if not langfuse:
        raise ValueError("Missing required parameter: langfuse")
    trace = langfuse.trace(
        name=request_name,
        user_id=user_id,
        session_id=session_id,
        release=release,
        tags=tags,
        metadata=metadata
    )

    # 创建追踪 Generation 对象
    generation = trace.generation(
        start_time=start_time,
        name=response_name,
        model=model
    )

    t_start = time.time()
    try:
        # 更新 trace：记录请求输入
        trace.update(
            input=request_input
        )
        # logger.info(f"Trace input recorded for {interface_name}: {request_input}")

        # 更新 generation：记录请求输入
        generation.update(
            input=request_input
        )
        # logger.info(f"Generation input recorded for {interface_name}: {request_input}")

        # 更新 trace：记录响应输出
        trace.update(
            output=response_output,
        )
        # logger.info(f"Trace output recorded for {interface_name}: {response_output}")

        # 更新 generation：记录响应输出
        generation.update(
            output=response_output
        )

        input_text = json.dumps(request_input, ensure_ascii=False) if request_input else ""
        output_text = str(response_output) if response_output else ""
        qwen_tokenizer = kwargs.get("tokenizer")
        input_tokens = await async_count_tokens(input_text, qwen_tokenizer)
        output_tokens = await async_count_tokens(output_text, qwen_tokenizer)

        usage_details = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }
        cost_details = {
            "input": input_tokens * 0.00001,
            "output": output_tokens * 0.00002,
        }
        # ================================================

        generation.end(
            end_time=end_time,
            usage=usage_details,
            total_cost=cost_details,
        )

        # logger.info(f"Generation output recorded for {interface_name}: {response_output}")

    except Exception as e:
        t_cost = round(time.time() - t_start, 2)

        # 更新 trace：记录失败状态
        trace.update(
            name=f"{interface_name} Trace Failure"
        )
        logger.error(f"Trace failed for {interface_name} after {t_cost}s with error: {repr(e)}")

        # 更新 generation：记录失败状态
        generation.update(
            name=f"{interface_name} Generation Failure"
        )
        logger.error(f"Generation failed for {interface_name} after {t_cost}s with error: {repr(e)}")
        raise e  # 继续抛出异常

    finally:
        # 提交追踪状态
        trace.update(
            state="completed",
            properties={"final_status": "success" if response_output else "failure"}
        )
        generation.update(
            state="completed",
            properties={"final_status": "success" if response_output else "failure"}
        )
        langfuse.flush()


def extract_time(date: str) -> str:
    """
    提取血压测量时间的 HH:MM 部分。
    :param date: 完整的时间字符串
    :return: HH:MM 格式的时间
    """
    if not date:
        return ""
    return date.split(" ")[1][:5]


def generate_pressure_advice(sbp: int, dbp: int, user_name) -> tuple:
    """
    根据血压值生成不同的规则话术。
    :param sbp: 收缩压
    :param dbp: 舒张压
    :return: 四个话术元组
    """
    blood_pressure = f"{sbp}/{dbp}mmHg"
    greeting = f"您好，客户{user_name}目前血压{blood_pressure}，"
    advice_suffix = "请及时与客户取得联系，给予处理建议。"

    # 规则生成话术（1、2、4）
    if sbp < 90 or dbp < 60:
        front = "当前血压低于正常，请及时就医！"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏低（低血压），{advice_suffix}"
    elif 90 <= sbp <= 119 and 60 <= dbp <= 79:
        front = "当前血压正常，请继续保持，适量运动！"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = "血压过低时，请立刻就医并密切监测血压变化，若情况未改善请联系医生。"
    elif ((120 <= sbp <= 139 and 60 <= dbp <= 89) or
            (90 <= sbp <= 119 and 80 <= dbp <= 89) or
            (120 <= sbp <= 139 and 80 <= dbp <= 89)):
        front = "当前血压处于正常高值，请遵医嘱，规律生活适量运动！"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = "血压正常，继续保持均衡饮食和适量运动，避免过度压力。"
    elif ((140 <= sbp <= 159 and dbp < 90) or
            (sbp < 140 and 90 <= dbp <= 99) or
            (140 <= sbp <= 159 and 90 <= dbp <= 99)):
        front = "您血压值偏高，请遵医嘱，规律生活适量运动。"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏高（1级高血压），{advice_suffix}"
    elif ((160 <= sbp <= 179 and dbp < 100) or
        (sbp < 160 and 100 <= dbp <= 109) or
        (160 <= sbp <= 179 and 100 <= dbp <= 109)):
        front = "您血压值较高，请遵医嘱并就医。"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏高（2级高血压），{advice_suffix}"
    elif ((sbp >= 180 and dbp < 110) or
        (sbp < 180 and dbp >= 110) or
        (sbp >= 180 and dbp >= 110)):
        front = "您血压值极高，请遵医嘱并及时就医。"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏高（3级高血压），{advice_suffix}"
    elif sbp >= 140 and dbp < 90:
        front = "您血压值偏高，请遵医嘱，规律生活适量运动。"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏高（单纯收缩期高血压），{advice_suffix}"
    elif sbp < 140 and dbp >= 90:
        front = "您血压值偏高，请遵医嘱，规律生活适量运动。"
        user_warning = f"{blood_pressure}，{front}"
        sug_agent = f"{greeting}血压偏高（单纯舒张期高血压），{advice_suffix}"

    return front, user_warning, sug_agent


def add_ending_punctuation(text: str) -> str:
    """
    If the text doesn't end with a punctuation mark (.,!?) add a Chinese period (。) at the end.
    If it ends with any other punctuation, keep it as it is.
    """
    # Check if the last character is a punctuation mark.
    if not text:
        return text  # If the text is empty, just return it

    # Define a regex for punctuation characters (.,!?) at the end of the string.
    if re.match(r'.*[。!！？?.]$', text):
        return text  # If the text ends with a punctuation mark, return it unchanged.

    # Otherwise, add a Chinese period (。) at the end.
    return text + "。"


def process_text(user_content: str, limit: int = 200) -> str:
    """
    从 user_content 中按行提取“完整句子”，
    在不超过 limit 个字符总长度的情况下依次拼接。
    拼接时会移除行首的 '- ' 符号，但行尾标点保留。
    返回拼接后的字符串。
    """
    # 1. 按行拆分
    lines = user_content.split('\n')

    # 2. 去掉空行，并移除每行开头的“- ”
    cleaned_sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('- '):
            line = line[2:].strip()
        cleaned_sentences.append(line)

    # 3. 在不超过 limit 的前提下，依次拼接完整的行
    result_sentences = []
    current_length = 0
    for sentence in cleaned_sentences:
        length_if_added = current_length + len(sentence)
        if length_if_added <= limit:
            result_sentences.append(sentence)
            current_length = length_if_added
        else:
            break

    # 4. 最终结果：直接拼接，不额外添加空格或换行
    return ''.join(result_sentences)


async def call_mem0_add_memory(mem0_url: str, user_id: str, messages: list):
    """异步调用 mem0.add_memory 进行会话记录"""
    try:
        # 转换 role: "1" → "user", "3" → "assistant"
        role_mapping = {"1": "user", "3": "assistant"}
        converted_messages = [
            {"role": role_mapping.get(msg["role"], "user"), "content": msg["content"]}
            for msg in messages
        ]

        mem0_payload = {
            "user_id": user_id,
            "messages": converted_messages  # 传递转换后的 messages
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{mem0_url}/add_memory", json=mem0_payload) as response:
                return await response.json()
    except Exception as e:
        logger.exception(f"mem0 调用失败: {e}")
        return None


async def call_mem0_search_memory(mem0_url: str, query: str, user_id: str) -> dict:
    """
    异步调用 mem0.search_memory 接口，查询用户的记忆数据。

    :param mem0_url: mem0 服务的基础 URL，例如 http://ner0/mull-adapter:5800
    :param query: 查询内容
    :param user_id: 用户 ID
    :return: 若调用成功，返回服务返回的完整 JSON；失败则返回 None。
    """
    try:
        payload = {
            "query": query,
            "user_id": user_id
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{mem0_url}/search_memory", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"mem0 search_memory 成功, 返回数据: {data}")
                    return data
                else:
                    logger.error(f"mem0 search_memory 调用失败, HTTP状态码: {response.status}")
                    return None
    except Exception as e:
        logger.exception(f"mem0 search_memory 调用异常: {e}")
        return None


async def call_mem0_get_all_memories(mem0_url: str, limit: Optional[int] = 1000) -> Optional[dict]:
    """
    异步调用 mem0 的 /get_all_memories 接口，获取所有用户的记忆数据。

    :param mem0_url: mem0 服务的基础 URL，例如 http://mem0-vllm-adapter:5000
    :param limit: 可选，返回的最大条数，默认 1000
    :return: 成功返回 dict（包含 all_memories 字段），失败返回 None
    """
    try:
        params = {"limit": limit} if limit is not None else {}

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{mem0_url}/get_all_memories", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"mem0 get_all_memories 成功, 返回数据: {data}")
                    return data
                else:
                    logger.error(f"mem0 get_all_memories 调用失败, HTTP状态码: {response.status}")
                    return None
    except Exception as e:
        logger.exception(f"mem0 get_all_memories 调用异常: {e}")
        return None


def format_mem0_search_result(search_data: dict) -> str:
    """
    将查询到的记忆数据整理成多行字符串。
    例如:
        感冒了
        骨折了
        123456今天刚学完英语
        123456开始学德语了

    :param search_data: mem0 返回的 JSON 数据
    :return: 按行拼接的字符串
    """
    if not search_data:
        logger.warning("search_data 为空或 None，无法格式化记忆。")
        return ""

    # 取出 search_result -> results
    search_result = search_data.get("search_result", {})
    results = search_result.get("results", [])

    if not isinstance(results, list):
        logger.warning("search_data 中的 results 字段不是列表格式。")
        return ""

    # 收集所有 memory 字段
    memory_lines = []
    for item in results:
        memory_text = item.get("memory")
        if memory_text:
            memory_lines.append(memory_text)
        else:
            logger.debug(f"结果中缺少 memory 字段或为空: {item}")

    # 以换行分割
    formatted = "\n".join(memory_lines)
    logger.debug(f"格式化后的记忆:\n{formatted}")
    return formatted


def match_health_label(health_labels_data: dict, label_name: str, tag_value: str) -> Optional[Dict]:
    """
    通过 `gs_resource.health_labels_data` 进行高效查询（O(1) 复杂度）

    参数:
        gs_resource: `InitAllResource` 实例，已预加载数据
        label_name (str): 例如 "现患疾病"
        tag_value (str): 例如 "感冒"

    返回:
        Dict | None: 匹配成功返回完整信息，否则返回 None。
    """
    return health_labels_data.get(label_name, {}).get(tag_value)


def jaccard_text_similarity(text1, text2):
    """ 计算 Jaccard 相似度 """
    set1, set2 = set(text1), set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def get_best_matching_dish(dish, ds, threshold=0.5):
    """ 使用 Jaccard 相似度匹配最相似的菜品 """
    target_dish = None

    for i, x in enumerate(ds):
        similarity = jaccard_text_similarity(x['name'], dish)
        if similarity < threshold:
            continue
        if not target_dish:
            target_dish = x
        elif similarity > jaccard_text_similarity(target_dish['name'], dish):
            target_dish = x

    if target_dish:
        logger.debug(f"匹配结果: {json.dumps(target_dish, ensure_ascii=False)}")
        return target_dish.get("image")
    else:
        logger.info(f"未找到匹配项: '{dish}'")
        return None


def enrich_meal_items_with_images(items, dishes_data, threshold, meal_time_mapping):
    """
    处理items，转换结构并添加对应的图片字段，同时新增推荐进食时间点（加餐时间点动态匹配）。
    """

    if not items or not isinstance(items, list):
        logger.warning("无效输入：items 应为非空列表。")
        return []

    processed_items = []
    previous_meal = None  # 记录上一个正餐

    for item in items:
        try:
            if not item or "meal" not in item or "foods" not in item:
                logger.warning(f"跳过无效的餐食项：{item}")
                continue

            meal_name, meal_calories = item["meal"]

            # 判断是否是加餐
            if "加餐" in meal_name:
                # 如果有上一个正餐，就根据上一个正餐来匹配加餐的时间
                if previous_meal == "早餐":
                    meal_time = meal_time_mapping.get("上午加餐", "")
                elif previous_meal == "午餐":
                    meal_time = meal_time_mapping.get("下午加餐", "")
                elif previous_meal == "晚餐":
                    meal_time = meal_time_mapping.get("晚餐加餐", "")
                else:
                    meal_time = ""  # 兜底处理
            else:
                meal_time = meal_time_mapping.get(meal_name, "")
                previous_meal = meal_name  # 记录当前正餐

            processed_meal = {
                "meal": meal_name,
                "time": meal_time,  # 赋值推荐时间
                "foods": []
            }

            for food in item["foods"]:
                try:
                    if not food:
                        logger.warning(f"跳过无效的食物项：{food}")
                        continue

                    recipe_name, amount, calories = food
                    image_url = get_best_matching_dish(recipe_name, dishes_data, threshold)

                    processed_meal["foods"].append({
                        "recipe": recipe_name,
                        "recommended_amount": amount,
                        "calories": {
                            "value": int(calories.replace("kcal", "")),
                            "unit": "kcal"
                        },
                        "image": image_url
                    })
                except Exception as e:
                    logger.error(f"处理食物项 {food} 时出错：{e}")

            processed_items.append(processed_meal)
        except Exception as e:
            logger.error(f"处理餐食项 {item} 时出错：{e}")

    return processed_items


def query_course(data_dict, course_name):
    """
    通过课程名称查询课程详细信息

    :param data_dict: 包含所有课程相关数据的字典
    :param course_name: 需要查询的课程名称
    :return: 课程详情，包括动作信息
    """
    sports_lessons = data_dict["sports_lessons"]
    sports_lesson_exercise_course = data_dict["sports_lesson_exercise_course"]
    exercise_course_action = data_dict["exercise_course_action"]
    actions = data_dict["actions"]

    # 1️⃣ 查找课程信息
    course_info = next((lesson for lesson in sports_lessons if lesson.get("NAME") == course_name), None)
    if not course_info:
        return {"head": 404, "items": {}, "msg": f"❌ 未找到课程: {course_name}"}

    course_code = course_info.get("Sports_lesson_code", "")
    course_image = course_info.get("Sports_lesson_picture", "")

    result = {
        "head": 200,
        "items": {
            "course_name": course_name,  # 课程名称
            "course_image": course_image,  # 课程封面图片
            "course_code": course_code,  # 课程编码
            "actions": []  # 课程包含的动作列表
        },
        "msg": ""
    }

    # 2️⃣ 查找课程的运动处方
    exercise_courses = sports_lesson_exercise_course.get(course_code, {}).get("relationDatas", [])
    if not exercise_courses:
        result["msg"] = "⚠️ 该课程未关联任何动作"
        return result

    # 3️⃣ 查找运动处方对应的所有动作
    for exercise in exercise_courses:
        exercise_course_code = exercise.get("exercise_course_code", "")
        exercise_actions = exercise_course_action.get(exercise_course_code, {}).get("relationDatas", [])

        for action in exercise_actions:
            action_code = action.get("Action_code", "")

            # 4️⃣ 在 `actions.json` 里查找具体动作信息
            action_info = next((act for act in actions if act.get("Action_code") == action_code), None)

            if action_info:
                result["items"]["actions"].append({
                    "action_name": action_info.get("Action_display_name", "未知动作"),  # 动作名称
                    "video_duration": action_info.get("Action_video_duration", 0),  # 视频时长
                    "action_image": action_info.get("Action_picture", ""),  # 动作图片
                    "video_url": action_info.get("Action_Video", "")  # 视频链接
                })
            else:
                result["items"]["actions"].append({"error": f"⚠️ 动作代码 {action_code} 在 action.json 里找不到！"})

    return result


async def convert_dict_to_key_value_section(
    data: Dict,
    image: str = "",
    section_key: str = "title",
    list_key: str = "data",
    key_field: str = "name",
    value_field: str = "value",
    image_field: str = "image"
) -> Dict[str, Union[str, List[Dict]]]:
    """
    将 { 一级标题: {子项key: 子项value} } 的嵌套结构转换为标准结构，带图片字段：
    {
        "section": 一级标题,
        "image": 图片地址,
        "data": [{key: ..., value: ...}]
    }

    参数:
        data (dict): 原始嵌套结构
        image (str): 可选的图片地址
        section_key (str): 一级标题字段名（默认 "section"）
        list_key (str): 子项列表字段名（默认 "data"）
        key_field (str): 子项标题字段名（默认 "key"）
        value_field (str): 子项内容字段名（默认 "value"）
        image_field (str): 图片字段名（默认 "image"）

    返回:
        dict: 标准结构
    """
    if not isinstance(data, dict) or len(data) == 0:
        return {section_key: "", image_field: image, list_key: []}

    first_key = next(iter(data))
    raw = data.get(first_key, {})

    if not isinstance(raw, dict):
        return {section_key: first_key, image_field: image, list_key: []}

    items = [{key_field: k, value_field: v} for k, v in raw.items()]

    return {
        section_key: first_key,
        image_field: image,
        list_key: items
    }


async def strip_think_block(text: str) -> str:
    """
    去除模型输出中的 <think>...</think> 思考标签段落（DeepSeek 等模型专用）

    参数:
        text (str): 模型原始输出

    返回:
        str: 去除 <think> 块后的纯净内容
    """
    if not isinstance(text, str):
        return text

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def convert_structured_kv_to_prompt_dict(section: dict) -> dict:
    """
    将结构化的 title + data 转换为 prompt 需要的 dict 格式
    """
    if not section or not isinstance(section, dict):
        return {}

    title = section.get("title")
    data = section.get("data", [])
    if not title or not isinstance(data, list):
        return {}

    mapped = {item["name"]: item["value"] for item in data if "name" in item and "value" in item}
    res = json.dumps({title: mapped}, ensure_ascii=False, indent=2)
    return res


async def convert_schedule_fields_to_english(schedule: dict) -> dict:
    """
    将日程中的中文字段转换为英文字段
    """
    mapping = {
        "日程名称": "schedule_name",
        "日程时间": "schedule_time",
        "推送话术": "push_text"
    }
    return {mapping.get(k, k): v for k, v in schedule.items()}


async def enrich_schedule_with_extras(item: dict) -> dict:
    """
    为每个日程打卡追加视频/图片/cate_code 字段，统一结构
    """
    name = item.get("schedule_name", "")
    item.setdefault("videos", None)
    item.setdefault("image", [])

    # 分类编码
    if "餐" in name:
        item["cate_code"] = "diet_schedule"
    elif "运动" in name:
        item["cate_code"] = "exercise_schedule"
    else:
        item["cate_code"] = "unknown_schedule"

    # 视频和图片处理
    if "运动" in name and "14" in item.get("schedule_time", ""):
        item["videos"] = [
            {"name": "摸膝卷腹", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/ai-laikang-com/recommend_new/sports/核心-力量/摸膝卷腹.mp4"},
            {"name": "垫上仰卧踢腿", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/ai-laikang-com/recommend_new/sports/核心结合心肺训练/垫上仰卧踢腿.mp4"},
            {"name": "垫上对侧抬手抬脚", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/ai-laikang-com/recommend_new/sports/力量-背/垫上对侧抬手抬脚.mp4"},
        ]
    elif "运动" in name and "19" in item.get("schedule_time", ""):
        item["image"] = [
            {"name": "散步", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/干预计划图片/散步.png"},
            {"name": "游泳", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/干预计划图片/游泳.png"},
            {"name": "跑步", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/干预计划图片/跑步.png"},
        ]
    elif "早餐" in name:
        item["image"] = [
            {"name": "早餐", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/一日3餐/早餐.png"}
        ]
    elif "午餐" in name:
        item["image"] = [
            {"name": "午餐", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/一日3餐/午餐.png"}
        ]
    elif "晚餐" in name:
        item["image"] = [
            {"name": "晚餐", "url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/姚树坤专家体系20250331/智能日程/一日3餐/晚餐.png"}
        ]

    return item


async def extract_daily_schedule(parsed) -> List[Dict]:
    """
    通用提取模型返回的日程结构，兼容字典/列表/字段名变动等
    """
    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        for key in parsed:
            if "日程" in key or "打卡" in key or "推送" in key:
                value = parsed[key]
                if isinstance(value, list):
                    return value
    return []


def init_langfuse_trace_with_input(
    *,
    extra_params: dict,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool,
    query: str = "",
    history: List[Dict] = [],
):
    """初始化 Langfuse trace 和 generation，并记录输入 input"""
    langfuse = extra_params.get("langfuse")
    user_id = extra_params.get("user_id")
    session_id = extra_params.get("session_id")

    if not (langfuse and user_id and session_id):
        return None, None, None, None, query or history

    trace = langfuse.trace(
        name=extra_params.get("name"),
        user_id=user_id,
        session_id=session_id,
        release=extra_params.get("release"),
        tags=extra_params.get("tags"),
        metadata=extra_params.get("metadata"),
    )

    generation = trace.generation(
        name="llm-generation",
        model=model,
        model_parameters={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        metadata={"streaming": stream},
    )

    input_data = query or history
    generation.update(input=input_data)
    trace.update(input=input_data)

    tokenizer = extra_params.get("tokenizer")

    return trace, generation, langfuse, tokenizer, input_data

async def async_init_langfuse_trace_with_input(
    *,
    extra_params: dict,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool,
    query: str = "",
    history: List[Dict] = [],
):
    """初始化 Langfuse trace 和 generation，并记录输入 input"""
    langfuse = extra_params.get("langfuse")
    user_id = extra_params.get("user_id")
    session_id = extra_params.get("session_id")

    if not (langfuse and user_id and session_id):
        return None, None, None, None, query or history

    trace = langfuse.trace(
        name=extra_params.get("name"),
        user_id=user_id,
        session_id=session_id,
        release=extra_params.get("release"),
        tags=extra_params.get("tags"),
        metadata=extra_params.get("metadata"),
    )

    generation = trace.generation(
        name="llm-generation",
        model=model,
        model_parameters={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        metadata={"streaming": stream},
    )

    input_data = query or history
    generation.update(input=input_data)
    trace.update(input=input_data)

    tokenizer = extra_params.get("tokenizer")

    return trace, generation, langfuse, tokenizer, input_data


ALL_INTENTS, _ = intent_init()

# 构建 intent_code -> intent_name 的映射
INTENT_NAME_MAP = {v[0]: (v[0], v[1]) for v in ALL_INTENTS.values()}
INTENT_NAME_MAP["other"] = ("other", "闲聊/其他")

def get_intent_name_and_tags(intent_code: str) -> Tuple[str, List[str]]:
    """
    根据 intent_code 映射 endpoint_name 与 tags
    """
    name, tag = INTENT_NAME_MAP.get(intent_code, ("other", "闲聊"))
    return f"{tag}", ["chat_gen", tag, intent_code]

def track_completion_with_langfuse(
    *,
    trace,
    generation,
    langfuse=None,
    input_data="",
    output_data="",
    tokenizer=None,
):
    """非流式调用的 Langfuse 监控记录"""
    try:
        input_text = flatten_input_text(input_data)
        input_tokens = count_tokens(input_text, tokenizer) if tokenizer else 0
        output_tokens = count_tokens(output_data, tokenizer) if tokenizer else 0

        usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }
        cost = {
            "input": input_tokens * 0.00001,
            "output": output_tokens * 0.00002,
        }

        generation.end(usage=usage, total_cost=cost)
        generation.update(output=output_data)
        trace.update(output=output_data)
        if langfuse:
            langfuse.flush()

        logger.info(f"[Langfuse] 非流式追踪完成 ✅ tokens: in={input_tokens}, out={output_tokens}")
    except Exception as e:
        logger.warning(f"[Langfuse] 非流式追踪失败 ❌: {repr(e)}")


async def async_track_completion_with_langfuse(
    *,
    trace,
    generation,
    langfuse=None,
    input_data="",
    output_data="",
    tokenizer=None,
):
    """非流式调用的 Langfuse 监控记录"""
    try:
        input_text = flatten_input_text(input_data)
        input_tokens = await async_count_tokens(input_text, tokenizer) if tokenizer else 0
        output_tokens = await async_count_tokens(output_data, tokenizer) if tokenizer else 0

        usage = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }
        cost = {
            "input": input_tokens * 0.00001,
            "output": output_tokens * 0.00002,
        }

        generation.end(usage=usage, total_cost=cost)
        generation.update(output=output_data)
        trace.update(output=output_data)
        if langfuse:
            langfuse.flush()

        logger.info(f"[Langfuse] 非流式追踪完成 ✅ tokens: in={input_tokens}, out={output_tokens}")
    except Exception as e:
        logger.warning(f"[Langfuse] 非流式追踪失败 ❌: {repr(e)}")


def wrap_stream_with_langfuse(
    stream,
    generation,
    trace,
    langfuse=None,
    tokenizer=None,
    input_data=""
):
    """包装流式生成器，在结束后补充 Langfuse 追踪信息"""

    all_chunks = []
    try:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            all_chunks.append(delta)
            yield chunk
    finally:
        full_output = "".join(all_chunks)
        input_text = flatten_input_text(input_data)

        if generation:
            try:
                input_tokens = count_tokens(input_text, tokenizer) if tokenizer else 0
                output_tokens = count_tokens(full_output, tokenizer) if tokenizer else 0

                usage = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                }
                cost = {
                    "input": input_tokens * 0.00001,
                    "output": output_tokens * 0.00002,
                }
                logger.debug("full_output", full_output)
                generation.end(usage=usage, total_cost=cost)
                generation.update(output=full_output)
                trace.update(output=full_output)
                langfuse.flush()
                logger.info(f"[Langfuse] 流式追踪完成 ✅ tokens: in={input_tokens}, out={output_tokens}")
            except Exception as e:
                logger.warning(f"[Langfuse] 流式追踪失败 ❌: {repr(e)}")


async def async_wrap_stream_with_langfuse(
    stream,
    generation,
    trace,
    langfuse=None,
    tokenizer=None,
    input_data=""
):
    """包装流式生成器，在结束后补充 Langfuse 追踪信息"""

    all_chunks = []
    try:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            all_chunks.append(delta)
            yield chunk
    finally:
        full_output = "".join(all_chunks)
        input_text = flatten_input_text(input_data)

        if generation:
            try:
                input_tokens = await async_count_tokens(input_text, tokenizer) if tokenizer else 0
                output_tokens = await async_count_tokens(full_output, tokenizer) if tokenizer else 0

                usage = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                }
                cost = {
                    "input": input_tokens * 0.00001,
                    "output": output_tokens * 0.00002,
                }
                generation.end(usage=usage, total_cost=cost)
                generation.update(output=full_output)
                trace.update(output=full_output)
                langfuse.flush()
                logger.info(f"[Langfuse] 流式追踪完成 ✅ tokens: in={input_tokens}, out={output_tokens}")
            except Exception as e:
                logger.warning(f"[Langfuse] 流式追踪失败 ❌: {repr(e)}")
