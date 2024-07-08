# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import argparse
import functools
import json
import pickle
import os
import re
import sys
import time
import oss2
from base64 import encode
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    AnyStr,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Literal,
    Tuple,
    Union,
)
from urllib import parse
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import numpy as np
import openai
import pandas as pd
import requests
import yaml
from sqlalchemy import MetaData, Table, create_engine
from typing import Optional
from data.constrant import CACHE_DIR
from src.utils.api_protocal import AigcFunctionsRequest, AigcFunctionsResponse
from src.utils.openai_api_protocal import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)

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


class InitAllResource:
    def __init__(self) -> None:
        """初始化公共资源"""
        self.__parse_args__()
        self.session = requests.Session()
        self.cache_dir = Path(CACHE_DIR)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()

        self.__load_config__()
        self.__knowledge_connect_check__()

        self.prompt_meta_data = self.req_prompt_data_from_mysql()

        self.__init_model_supplier__()
        self.client = openai.OpenAI()
        self.aclient = openai.AsyncOpenAI()

    def __parse_args__(
        self,
    ) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            type=str,
            default=os.environ.get("ENV", "local"),
            help="env: local, dev, test, prod",
        )
        parser.add_argument("--ip", type=str, default="0.0.0.0", help="ip")
        parser.add_argument("--port", type=int, default=6500, help="port")
        parser.add_argument(
            "--use_proxy", action="store_true", help="whether use proxy"
        )
        parser.add_argument(
            "--use_cache", action="store_true", help="是否使用缓存, Default为False"
        )
        parser.add_argument(
            "--special_prompt_version",
            action="store_true",
            help="是否使用指定的prompt版本, Default为False,都使用lastest",
        )
        self.args = parser.parse_args()
        # os.environ["env"] = self.args.env
        logger.info(f"Initialize args: {self.args}")

    def __init_model_supplier__(self) -> None:
        """初始化模型供应商"""
        for supplier_name, supplier_config in self.api_config["model_supply"].items():
            if not isinstance(supplier_config, dict):
                continue
            client = openai.OpenAI(
                base_url=supplier_config["api_base"] + "/v1",
                api_key=supplier_config.get("api_key"),
            )
            models = ",".join([i.id for i in client.models.list().data])
            logger.info(f"Supplier [{supplier_name:^6}] support models: {models:<15}")
        default_supplier = self.api_config["model_supply"].get("default", "vllm")
        os.environ["OPENAI_BASE_URL"] = (
            self.api_config["model_supply"][default_supplier]["api_base"] + "/v1"
        )
        os.environ["OPENAI_API_KEY"] = self.api_config["model_supply"][
            default_supplier
        ]["api_key"]
        self.default_model = "Qwen1.5-32B-Chat"
        logger.info(
            f"Set default supplier [{default_supplier}], default model: {self.default_model}"
        )

    def __knowledge_connect_check__(self) -> None:
        """检查知识库服务连接"""
        try:
            response = self.session.get(
                self.api_config["langchain"] + "/knowledge_base/list_knowledge_bases",
                timeout=10,
            )
            if response.status_code == 200:
                support_knowledge_list = response.json()["data"]
                logger.success(f"Support knowledge list: {support_knowledge_list}")
            else:
                raise Exception(f"Knowledge connect error: {response.status_code}")
        except Exception as err:
            logger.error(f"Knowledge connect error: {err}")
            sys.exit(1)

    def __load_config__(self) -> None:
        """指定env加载配置"""
        self.api_config = load_yaml(Path("config", "api_config.yaml"))[self.args.env]
        self.mysql_config = load_yaml(Path("config", "mysql_config.yaml"))[
            self.args.env
        ]
        self.prompt_version = load_yaml(Path("config", "prompt_version.yaml"))[
            self.args.env
        ]
        model_config = load_yaml(Path("config", "model_config.yaml"))[self.args.env]
        self.model_config = {
            event: model
            for model, event_list in model_config.items()
            for event in event_list
        }
        intent_aigcfunc_map = load_yaml(Path("config", "intent_aigcfunc_map.yaml"))

        self.intent_aigcfunc_map = {}
        _tmp_dict = {}
        for aigcFuncCode, detail in intent_aigcfunc_map.items():
            _intent_code_list = [
                i.strip() for i in detail["intent_code_list"].split(",")
            ]
            for intentCode in _intent_code_list:
                if _tmp_dict.get(intentCode):
                    logger.warning(
                        f"intent_code {intentCode} has been used by {aigcFuncCode} and {_tmp_dict[intentCode]}"
                    )
                else:
                    _tmp_dict[intentCode] = aigcFuncCode
                self.intent_aigcfunc_map[intentCode] = aigcFuncCode
        self.__info_config__(model_config)

    def __info_config__(self, model_config):
        logger.debug(f"Initialize api config ...")
        for key, value in self.api_config.items():
            if key == "model_supply":
                value_list = [f"default: {value['default']}"]
                for k, v in value.items():
                    if isinstance(v, dict):
                        api_base = v["api_base"]
                        support_models = v["support_models"]
                        value_list.append(f"[{k}]: api_base: {api_base}, support_models: {support_models}")
                value = ", ".join(value_list)
            logger.debug(f"[{key:^16}]: {value}")
        logger.debug(
            f"Initialize mysql config: {self.mysql_config['user']}@{self.mysql_config['ip']}:{self.mysql_config['port']} {self.mysql_config['db_name']}"
        )
        for key, value in self.prompt_version.items():
            if not value:
                continue
            for ik, iv in value.items():
                logger.debug(f"Initialize prompt version {key} - {ik} - {iv}")
        for key, model_list in model_config.items():
            logger.debug(f"Model Usage: {key} - {model_list}")

    @clock
    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data"""

        def filter_format(obj, splited=False):
            obj_str = json.dumps(obj, ensure_ascii=False).replace("\\r\\n", "\\n")
            obj_rev = json.loads(obj_str)
            if splited:
                for obj_rev_item in obj_rev:
                    if obj_rev_item.get("event"):
                        obj_rev_item["event"] = obj_rev_item["event"].split("\n")
            return obj_rev

        def search_target_version_item(item_list, ikey, curr_item_id, version):
            """从列表中返回指定version的item, 默认未指定则为latest"""
            spec_version = version.get(curr_item_id, None)
            if spec_version:
                for item in item_list:
                    if item[ikey] == curr_item_id:
                        if item["version"] == spec_version:
                            logger.debug(
                                f"load spec version {ikey} - {curr_item_id} - {spec_version}"
                            )
                            return item
                        else:
                            latest_item = item
                    else:
                        continue
            else:
                latest_item = [
                    i
                    for i in item_list
                    if i[ikey] == curr_item_id and i["version"] == "latest"
                ][0]
            return latest_item

        data_cache_file = self.cache_dir.joinpath("prompt_meta_data.pkl")
        if not self.args.use_cache or not data_cache_file.exists():
            mysql_conn = MysqlConnector(**self.mysql_config)
            prompt_meta_data = defaultdict(dict)
            prompt_character = mysql_conn.query("select * from ai_prompt_character")
            prompt_event = mysql_conn.query("select * from ai_prompt_event")
            prompt_tool = mysql_conn.query("select * from ai_prompt_tool")
            prompt_intent = mysql_conn.query("select * from ai_prompt_intent")
            prompt_character = filter_format(prompt_character, splited=True)
            prompt_event = filter_format(prompt_event)
            prompt_tool = filter_format(prompt_tool)
            prompt_intent = filter_format(prompt_intent)

            # 优先使用指定的version 否则使用latest
            if self.args.special_prompt_version:
                for key, v in self.prompt_version.items():
                    if not v:
                        if key == "character":
                            prompt_meta_data["character"] = {
                                i["name"]: i for i in prompt_character
                            }
                        elif key == "event":
                            prompt_meta_data["event"] = {
                                i["intent_code"]: i for i in prompt_event
                            }
                        elif key == "tool":
                            prompt_meta_data["tool"] = {
                                i["name"]: i for i in prompt_tool
                            }
                        elif key == "intent":
                            prompt_meta_data["intent"] = {
                                i["name"]: i for i in prompt_intent
                            }
                    else:
                        if key == "character":
                            for i in prompt_character:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = (
                                    search_target_version_item(
                                        prompt_character, "name", i["name"], v
                                    )
                                )
                        elif key == "event":
                            for i in prompt_event:
                                if prompt_meta_data[key].get(i["intent_code"]):
                                    continue
                                prompt_meta_data[key][i["intent_code"]] = (
                                    search_target_version_item(
                                        prompt_event, "intent_code", i["intent_code"], v
                                    )
                                )
                        elif key == "tool":
                            for i in prompt_tool:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = (
                                    search_target_version_item(
                                        prompt_tool, "name", i["name"], v
                                    )
                                )
                        elif key == "intent":
                            for i in prompt_intent:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = (
                                    search_target_version_item(
                                        prompt_intent, "name", i["name"], v
                                    )
                                )
            else:
                prompt_meta_data["character"] = {
                    i["name"]: i for i in prompt_character if i["type"] == "event"
                }
                prompt_meta_data["role_play"] = {
                    i["name"]: i for i in prompt_character if i["type"] == "role_play"
                }
                prompt_meta_data["event"] = {i["intent_code"]: i for i in prompt_event}
                prompt_meta_data["tool"] = {
                    i["name"]: i for i in prompt_tool if i["in_used"] == 1
                }
                prompt_meta_data["intent"] = {i["name"]: i for i in prompt_intent}
            prompt_meta_data["init_intent"] = {
                i["code"]: True for i in prompt_tool if i["init_intent"] == 1
            }
            prompt_meta_data["rollout_tool"] = {
                i["code"]: 1 for i in prompt_tool if i["requirement"] == "rollout"
            }
            prompt_meta_data["rollout_tool_after_complete"] = {
                i["code"]: 1
                for i in prompt_tool
                if i["requirement"] == "complete_rollout"
            }
            prompt_meta_data["prompt_tool_code_map"] = {
                i["code"]: i["name"] for i in prompt_tool if i["code"]
            }
            pickle.dump(prompt_meta_data, open(data_cache_file, "wb"))
            logger.debug(f"dump prompt_meta_data to {data_cache_file}")
            del mysql_conn
        else:
            prompt_meta_data = pickle.load(open(data_cache_file, "rb"))
            logger.debug(f"load prompt_meta_data from {data_cache_file}")

        for name, func in prompt_meta_data["tool"].items():
            func["params"] = (
                json.loads(func["params"]) if func["params"] else func["params"]
            )
        intent_desc_map = {
            code: item["intent_desc"]
            for code, item in prompt_meta_data["event"].items()
        }
        default_desc_map = loadJS(Path("data", "intent_desc_map.json"))
        # 以intent_desc_map.json定义的intent_desc优先
        self.intent_desc_map = {**intent_desc_map, **default_desc_map}
        return prompt_meta_data

    def get_model(self, event: str) -> str:
        """根据事件获取模型"""
        try:
            assert (
                isinstance(event, str) and event in self.model_config
            ), f"event {event} not in model_config"
        except Exception as err:
            logger.critical(err)
        return self.model_config.get(event, self.default_model)

    def get_event_item(self, event: str) -> Dict:
        """根据事件获取对应item"""
        assert isinstance(event, str) and self.prompt_meta_data["event"].get(
            event
        ), f"event {event} not in prompt_meta_data"
        prompt_item = self.prompt_meta_data["event"][event]
        return prompt_item


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
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


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


class MysqlConnector:
    def __init__(
        self,
        user: str = None,
        passwd: str = None,
        ip: str = "0.0.0.0",
        port: int = 3306,
        db_name: str = "localhost",
    ) -> None:
        """
        user: 用户名
        passwd: 密码
        ip: 目标ip
        port: mysql端口
        db_name: 目标库名
        """
        passwd = parse.quote_plus(passwd)
        self.url = f"mysql+pymysql://{user}:{passwd}@{ip}:{port}/{db_name}"
        self.engine = create_engine(self.url)
        self.metadata = MetaData(self.engine)
        self.connect = self.engine.connect()

    def reconnect(self):
        """
        mysql重连
        """
        self.engine = create_engine(self.url)
        self.metadata = MetaData(self.engine)
        self.connect = self.engine.connect()

    def insert(self, table_name, datas):
        """
        表插入接口
        table_name: 指定表名
        datas: 要插入的数据 形如 [{col1:d1, col2:d2}, {col1:d1, col2:d2}] 字段名和表列名保持一致
        """
        table_obj = Table(table_name, self.metadata, autoload=True)
        try:
            self.connect.execute(table_obj.insert(), datas)
        except Exception as error:
            try:
                self.reconnect()
                self.connect.execute(table_obj.insert(), datas)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        print(table_name, "\tinsert ->\t", len(datas))

    def query(self, sql, orient="records"):
        """
        pd.read_sql_query
        sql: sql查询语句
        orient: 默认"orient" 返回的数据格式 [{col1:d1, col2:d2},{}]
        """
        try:
            # res = self.engine.connect().execute(sql)
            res = pd.read_sql_query(sql, self.connect).to_dict(orient=orient)
        except Exception as error:
            try:
                self.reconnect()
                res = pd.read_sql_query(sql, self.connect).to_dict(orient=orient)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        return res

    def execute(self, sql):
        """
        执行sql语句
        """
        try:
            res = self.connect.execute(sql)
        except Exception as error:
            try:
                self.reconnect()
                res = self.connect.execute(sql)
            except Exception as error:
                print(error)
        finally:
            self.engine.dispose()
        return res


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
        if chunk.object == "text_completion":
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
            if "chat" not in chunk.object:
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
        # 尝试将单引号替换为双引号
        content = content.replace("'", '"')
        # 解析JSON字符串
        examination_plan = json.loads(content)
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


def parse_historical_diets(historical_diets: List[Dict[str, Union[str, Dict[str, List[str]]]]]) -> str:
    """解析历史食谱的列表数据并转换成所需的格式"""
    formatted_diets = ""
    try:
        for diet in historical_diets:
            date = diet.get("date", "未知日期")
            meals = diet.get("meals", {})
            formatted_diets += f"{date}：\n"
            for meal, foods in meals.items():
                formatted_diets += f"{meal}：\n"
                for food in foods:
                    formatted_diets += f"  - {food}\n"
    except Exception as e:
        print(f"解析错误: {e}")

    return formatted_diets.strip()

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

def convert_meal_plan_to_text(meal_plan_data: List[Dict[str, List[str]]]) -> str:
    """将餐次、食物名称的字典结构转换为文本形式"""
    formatted_text = ""
    for meal in meal_plan_data:
        meal_name = meal["meal"]
        foods = meal["foods"]
        formatted_text += f"{meal_name}:\n"
        for food in foods:
            formatted_text += f"  - {food}\n"
    return formatted_text.strip()





if __name__ == "__main__":
    InitAllResource()
