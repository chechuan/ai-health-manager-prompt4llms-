# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

import functools
import json
import sys
import time
import re
import yaml
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    Any, AnyStr, AsyncGenerator, Dict, Generator, List, Tuple, Union, Optional
)
import oss2
import numpy as np
import openai
import requests
from lunar_python import Lunar, Solar
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from contextlib import contextmanager
from src.utils.api_protocal import AigcFunctionsResponse
from src.utils.openai_api_protocal import (
    CompletionResponseStreamChoice, CompletionStreamResponse
)
import asyncio

try:
    from src.utils.Logger import logger
except Exception as err:
    from Logger import logger
from functools import wraps


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


def get_intent(text):
    """通过关键词解析意图->code"""
    if "饮食营养" in text:
        code = "food_nutri"
        desc = "饮食营养"
    elif "医疗" in text:
        code = "med_health"
        desc = "医疗健康"
    elif "血压测量" in text or "测量血压" in text:
        code = "remind_take_blood_pressure"
        desc = "提醒他人测量血压"
    elif "运动切换" in text or "切换运动" in text:
        code = "switch_exercise"
        desc = "运动切换"
    elif "数字人" in text and "换回" in text:
        code = "digital_image_back"
        desc = "换回数字人皮肤"
    elif "数字人" in text and "切换" in text:
        code = "digital_image_switch"
        desc = "切换数字人皮肤"
    elif "运动评价" in text:
        code = "sport_eval"
        desc = "运动评价"
    elif "运动咨询" in text:
        code = "sport_rec"
        desc = "运动咨询"
    elif "行程" in text:
        code = "route_rec"
        desc = "行程推荐"
    elif "温泉" in text:
        code = "spa_rec"
        desc = "温泉推荐"
    elif "页面" in text or "打开" in text:
        code = "open_Function"
        desc = "打开功能页面"
    elif "菜谱" in text:
        code = "cookbook"
        desc = "菜谱"
    elif "音乐" in text:
        code = "musicX"
        desc = "音乐播放"
    elif "天气" in text:
        code = "weather"
        desc = "天气查询"
    elif "辅助诊断" in text:
        code = "auxiliary_diagnosis"
        desc = "辅助诊断"
    elif "问诊" in text:
        code = "auxiliary_diagnosis"
        desc = ("辅助诊断",)
    elif "用药" in text:
        code = "drug_rec"
        desc = "用药咨询"
    elif "营养知识" in text:
        code = "nutri_knowledge_rec"
        desc = "营养知识咨询"
    elif "饮食处方" in text:
        code = "food_rec"
        desc = "饮食处方推荐"
    elif "饮食评价" in text:
        code = "food_eval"
        desc = "饮食评价"
    elif "医师" in text or "医生" in text:
        code = "call_doctor"
        desc = "呼叫医师"
    elif "运动师" in text:
        code = "call_sportMaster"
        desc = "呼叫运动师"
    elif "心理" in text or "情志" in text:
        code = "call_psychologist"
        desc = "呼叫情志师"
    elif "营养师" in text:
        code = "call_dietista"
        desc = "呼叫营养师"
    elif "健管师" in text:
        code = "call_health_manager"
        desc = "呼叫健管师"
    elif "网络" in text:
        code = "websearch"
        desc = "网络搜索"
    elif "首都" in text:
        code = "KLLI3.captialInfo"
        desc = "首都查询"
    elif "彩票" in text:
        code = "lottery"
        desc = "彩票"
    elif "解梦" in text:
        code = "dream"
        desc = "周公解梦"
    elif "计算器" in text:
        code = "AIUI.calc"
        desc = "计算器"
    elif "国内城市查询" in text:
        code = "LEIQIAO.cityOfPro"
        desc = "国内城市查询"
    elif "省会查询" in text:
        code = "ZUOMX.queryCapital"
        desc = "省会查询"
    elif "翻译" in text:
        code = "translation"
        desc = "翻译"
    elif "垃圾" in text:
        code = "garbageClassifyPro"
        desc = "垃圾分类"
    elif "尾号限行" in text:
        code = "carNumber"
        desc = "尾号限行"
    elif "单位换算" in text:
        code = "AIUI.unitConversion"
        desc = "单位换算"
    elif "汇率" in text:
        code = "AIUI.forexPro"
        desc = "汇率"
    elif "时间日期" in text:
        code = "datetimePro"
        desc = "时间日期"
    elif "眼保健操" in text:
        code = "AIUI.ocularGym"
        desc = "眼保健操"
    elif "故事" in text:
        code = "story"
        desc = "故事"
    elif "圣经" in text:
        code = "AIUI.Bible"
        desc = "圣经"
    elif "戏曲" in text:
        code = "drama"
        desc = "戏曲"
    elif "评书" in text:
        code = "storyTelling"
        desc = "评书"
    elif "有声书" in text:
        code = "AIUI.audioBook"
        desc = "有声书"
    elif "新闻" in text:
        code = "news"
        desc = "新闻"
    elif "BMI" in text:
        code = "BMI"
        desc = "BMI"
    elif "万年历" in text:
        code = "calendar"
        desc = "万年历"
    elif "音频" in text:
        code = "audio"
        desc = "音频播放"
    elif "笑话" in text:
        code = "joke"
        desc = "笑话"
    elif "五师" in text:
        code = "wushi"
        desc = "呼叫五师"
    elif "呼叫其他" in text:
        code = "call_other"
        desc = "呼叫其他"
    elif "营养其他" in text:
        code = "nutri_other"
        desc = "营养其他"
    elif "高血压" in text:
        code = "chronic_qa"
        desc = "高血压知识问答"
    elif "非会议日程管理" in text:
        code = "other_schedule"
        desc = "非会议日程管理"
    elif "会议日程管理" in text:
        code = "meeting_schedule"
        desc = "会议日程管理"
    elif "日程管理" in text:
        code = "schedule_manager"
        desc = "日程管理"
    elif "食材采购清单管理" in text:
        code = "food_purchasing_list_management"
        desc = "食材采购清单管理"
    elif "生成食材采购清单" in text:
        code = "create_food_purchasing_list"
        desc = "生成食材采购清单"
    elif "食材采购" in text:
        code = "food_purchasing"
        desc = "食材采购"
    elif "食材采购清单确认" in text:
        code = "food_purchasing_list_verify"
        desc = "食材采购清单确认"
    elif "食材采购清单关闭" in text:
        code = "food_purchasing_list_close"
        desc = "食材采购清单关闭"
    elif "拉群共策" in text:
        code = "shared_decision"
        desc = "拉群共策"
    elif "新奥百科" in text:
        code = "enn_wiki"
        desc = "新奥百科知识"
    elif "猜你想问" in text:
        code = "aigc_functions_generate_related_questions"
        desc = "猜你想问"
    elif "健康宝" in text:
        code = "hospital"
        desc = "健康宝"
    elif "医院预约" in text:
        code = "hospital_appointment"
        desc = "医院预约"
    elif "检查结果查询" in text:
        code = "exam_results"
        desc = "检查结果查询"
    elif "健康报告解读" in text:
        code = "report_analysis"
        desc = "健康报告解读"
    else:
        code = "other"
        desc = "日常对话"
    logger.debug(f"识别出的意图:{text} code:{code}")
    return code, desc


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
        else j + len("\nAction Input:") + text[j + len("\nAction Input:") :].find("\n")
    )
    l = text.find("\nFinal Answer:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
            k = text.rfind("\nObservation:")
    if 0 <= i < j < k:
        plugin_thought = text[h + len("Thought:") : i].strip()
        plugin_name = text[i + len("\nAction:") : j].strip()
        plugin_args = text[j + len("\nAction Input:") : k].strip()
        return plugin_thought, plugin_name, plugin_args
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len("Thought:") : l].strip()
            plugin_args = text[l + len("\nFinal Answer:") :].strip()
            plugin_args.split("\n")[0]
            return plugin_thought, "直接回复用户问题", plugin_args
        else:
            plugin_args = text[l + len("\nFinal Answer:") :].strip()
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
        plugin_thought = text[h + len("\nThought:") : i].strip()
        plugin_name = text[i + len("\nAction:") : j].strip()
        plugin_args = text[j + len("\nAction Input:") : k].strip()
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len("Thought:") : l].strip()
            plugin_args = text[l + len("\nFinal Answer:") :].strip()
            plugin_args.split("\n")[0]
        else:
            plugin_args = text[l + len("\nFinal Answer:") :].strip()
            plugin_thought = "I know the final answer."
    else:
        m = text.find("\nAnswer: ")
        next_thought_index = text[m + len("\nAnswer: ") :].find("\nThought:")
        if next_thought_index == -1:
            n = len(text)
        else:
            n = m + len("\nAnswer: ") + next_thought_index
        plugin_thought = text[len("\nThought: ") : m].strip()
        plugin_args = text[m + len("\nAnswer: ") : n].strip()
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
                chunk_resp = AigcFunctionsResponse(message=content, code=200, end=False)
                yield f"data: {chunk_resp.model_dump_json(exclude_unset=False)}\n\n"
        chunk_resp = AigcFunctionsResponse(message="", code=200, end=True)
    else:
        chunk_resp = AigcFunctionsResponse(message=response, code=601, end=True)
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
    根据饮食方案标准计算推荐的三大产能营养素克数和推荐餐次及每餐的能量，并格式化输出结果

    参数:
        diet_plan_standards (dict): 包含饮食方案标准的字典

    返回:
        str: 格式化的输出结果

    抛出:
        DietPlanCalculationError: 如果数据缺失，则抛出异常
    """
    # 检查推荐每日饮食摄入热量值是否存在且有效
    if not diet_plan_standards.get("recommended_daily_caloric_intake") or not diet_plan_standards[
        "recommended_daily_caloric_intake"].get("calories"):
        raise ValueError("缺少推荐每日饮食摄入热量值")

    calories = diet_plan_standards["recommended_daily_caloric_intake"]["calories"]

    # 获取推荐的三大产能营养素和推荐餐次及每餐的能量数据
    macronutrients = diet_plan_standards.get("recommended_macronutrient_grams", [])
    meals = diet_plan_standards.get("recommended_meal_energy", [])

    # 如果推荐的三大产能营养素或推荐餐次及每餐能量的数据缺失，抛出异常
    if not macronutrients or not meals:
        raise ValueError("缺少推荐三大产能营养素或推荐餐次及每餐能量的数据")

    def calculate_macronutrient_range(calories, min_ratio, max_ratio, divisor):
        """
        根据能量占比计算营养素的质量范围

        参数:
            calories (float): 推荐每日饮食摄入热量值
            min_ratio (float): 能量占比的最小值
            max_ratio (float): 能量占比的最大值
            divisor (int): 能量转换因子，碳水化合物和蛋白质为4，脂肪为9

        返回:
            tuple: 最小值和最大值（四舍五入后的整数）
        """
        min_value = (calories * min_ratio) / divisor
        max_value = (calories * max_ratio) / divisor
        return round(min_value), round(max_value)

    def get_macronutrient_range(nutrient):
        """
        获取指定营养素的质量范围

        参数:
            nutrient (str): 营养素名称

        返回:
            tuple: 最小值和最大值（四舍五入后的整数）
        """
        for item in macronutrients:
            if item["nutrient"] == nutrient:
                return calculate_macronutrient_range(calories, item.get("min_energy_ratio", 0),
                                                     item.get("max_energy_ratio", 0),
                                                     4 if nutrient != "脂肪" else 9)
        return 0, 0

    # 计算各营养素的质量范围
    carb_min, carb_max = get_macronutrient_range("碳水化合物")
    protein_min, protein_max = get_macronutrient_range("蛋白质")
    fat_min, fat_max = get_macronutrient_range("脂肪")

    def calculate_meal_energy_range(calories, min_ratio, max_ratio):
        """
        根据能量占比计算每餐的能量范围

        参数:
            calories (float): 推荐每日饮食摄入热量值
            min_ratio (float): 每餐能量占比的最小值
            max_ratio (float): 每餐能量占比的最大值

        返回:
            tuple: 最小值和最大值（四舍五入后的整数）
        """
        min_value = calories * min_ratio
        max_value = calories * max_ratio
        return round(min_value), round(max_value)

    # 计算每餐的能量范围，并格式化输出
    meal_energy_ranges = []
    for meal in meals:
        meal_name = meal["meal_name"]
        min_energy, max_energy = calculate_meal_energy_range(calories, meal.get("min_energy_ratio", 0),
                                                             meal.get("max_energy_ratio", 0))
        meal_energy_ranges.append(f"{meal_name}：{min_energy}kcal-{max_energy}kcal")

    # 格式化输出结果
    output = f"## 推荐每日饮食摄入热量值\n{calories}kcal\n"
    output += "## 推荐三大产能营养素推荐克数\n"
    output += f"碳水化合物：{carb_min}g-{carb_max}g\n"
    output += f"蛋白质：{protein_min}g-{protein_max}g\n"
    output += f"脂肪：{fat_min}g-{fat_max}g\n"
    output += "## 推荐餐次及每餐能量\n"
    output += "\n".join(meal_energy_ranges)

    return output


async def format_historical_meal_plans(historical_meal_plans: list) -> str:
    """
    将历史食谱转换为指定格式的字符串

    参数:
        historical_meal_plans (list): 包含历史食谱的列表

    返回:
        str: 格式化的历史食谱字符串
    """
    if not historical_meal_plans:
        return "历史食谱数据为空"

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
        required_keys = ['key', 'weather_base_url', 'geo_base_url']
        if not all(key in config for key in required_keys):
            logger.error("Missing required configuration keys")
            raise WeatherServiceError("Invalid configuration")

        api_key = config['key']
        weather_base_url = config['weather_base_url']
        geoapi_url = config['geo_base_url']

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
            'uvIndex', 'humidity', 'precip', 'pressure', 'vis'
        ]

        if not all(field in today_weather for field in required_weather_fields):
            logger.error("Missing required weather fields in response")
            return None

        formatted_weather = (
            f"今日{city}天气{today_weather['textDay']}，"
            f"最高温度{today_weather['tempMax']}度，"
            f"最低温度{today_weather['tempMin']}度，"
            f"风力{today_weather['windScaleDay']}级，"
            f"紫外线强度指数{today_weather['uvIndex']}，"
            f"湿度{today_weather['humidity']}%，"
            f"降水量{today_weather['precip']}mm，"
            f"气压{today_weather['pressure']}hPa，"
            f"能见度{today_weather['vis']}km。"
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


async def determine_recent_solar_terms():
    date = datetime.now()
    lunar = Lunar.fromDate(date)

    # 获取当天的节气
    current_jieqi = lunar.getCurrentJieQi()
    if current_jieqi:
        return f"{current_jieqi.getSolar().toYmd()} {current_jieqi.getName()}"

    # 获取下一个节气
    next_jieqi = lunar.getNextJieQi(True)
    if next_jieqi:
        next_jieqi_solar = next_jieqi.getSolar()
        next_jieqi_date = datetime(next_jieqi_solar.getYear(), next_jieqi_solar.getMonth(), next_jieqi_solar.getDay())
        delta_days = (next_jieqi_date - date).days
        if delta_days <= 7:
            return f"{next_jieqi_date.strftime('%Y-%m-%d')} {next_jieqi.getName()}"

    return None

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
            pre_delta_days = (date-pre_jieqi_date).days
            if pre_delta_days<delta_days:
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


def remove_empty_dicts(data):
    # 移除列表中为空的字典
    if isinstance(data, list):
        return [remove_empty_dicts(item) for item in data if not (isinstance(item, dict) and len(item) == 0)]
    elif isinstance(data, dict):
        return {key: remove_empty_dicts(value) for key, value in data.items() if not (isinstance(value, dict) and len(value) == 0)}
    else:
        return data


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


async def check_required_fields(params: dict, required_fields: dict, at_least_one: list = []):
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


def handle_api_error(status_code, data, logger):
    """
    根据 API 返回的状态码和错误信息进行处理
    """
    if status_code == 400:
        # 错误的请求 - 无效的参数或缺少参数
        if 'error' in data:
            error_title = data['error'].get('title', '未知错误')
            error_detail = data['error'].get('detail', '没有可用的详细信息')
            logger.error(f"Error 400 - {error_title}: {error_detail}")
        else:
            logger.error("错误的请求：请求可能包含无效的参数或缺少必要的参数。")

    elif status_code == 401:
        # 未授权 - 身份验证失败
        logger.error("错误 401 - 未授权：身份验证失败，可能是由于不正确的 API 密钥或令牌。")

    elif status_code == 403:
        # 禁止访问 - 访问被拒绝
        logger.error("错误 403 - 禁止访问：访问被拒绝，可能是由于余额不足、访问限制或无效的凭据。")

    elif status_code == 404:
        # 未找到 - 请求的数据/资源未找到
        logger.error("错误 404 - 未找到：无法找到请求的数据或资源。")

    elif status_code == 429:
        # 请求过多 - 超过速率限制
        logger.error("错误 429 - 请求过多：超过速率限制，请稍后再试。")

    elif status_code == 500:
        # 内部服务器错误 - 未知的服务器错误
        logger.error("错误 500 - 内部服务器错误：服务器发生未知错误。请联系 API 提供商。")

    else:
        # 未知错误
        logger.error(f"未知错误，HTTP 状态码：{status_code}")
