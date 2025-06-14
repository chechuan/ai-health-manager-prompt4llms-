# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

import argparse
import json
import pickle
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Generator, Dict, Union, List, Literal, AsyncGenerator, Set
from transformers import AutoTokenizer

import openai
import requests
from langfuse import Langfuse
import pandas as pd

from data.constrant import CACHE_DIR
from src.utils.module import clock, load_yaml, loadJS, intent_init
from src.utils.database import MysqlConnector

try:
    from src.utils.Logger import logger
except Exception as err:
    from Logger import logger


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
        self.langfuse_client = self.__init_langfuse__()

        # API鉴权信息
        self.parameter_config = self.api_config.get("parameter_config", {})
        self.host = self.parameter_config.get("host")
        self.api_key = self.parameter_config.get("api_key")
        self.api_secret = self.parameter_config.get("api_secret")
        self.api_endpoints = self.parameter_config.get("api_endpoints", {})

        try:
            self.prompt_meta_data = self.req_prompt_data_from_mysql()
        except Exception as e:
            self.prompt_meta_data = {}

        self.__init_model_supplier__()
        self.client = openai.OpenAI()
        self.aclient = openai.AsyncOpenAI()

        self.weather_api_config = self.__load_weather_api_config__()
        self.all_intent, self.com_intent = intent_init()[:2]

        # **预加载家康宝数据**
        self.jia_kang_bao_data = self.__load_jia_kang_bao_data__()
        self.jia_kang_bao_data_id_item = {item["id"]: item for item in self.jia_kang_bao_data}

        # **预加载健康标签数据**
        self.health_labels_data = self.__load_health_labels__()

        # **预加载菜品数据**
        self.dishes_data = self.__load_dishes__()

        self.exercise_data = self.__load_exercise_data__()

        self.qwen_tokenizer = self.__init_qwen_tokenizer__()

        self.sensitive_words = self.__load_sensitive_words__()

        self.simulation_rule_base = self.__load_simulation_rule_base__()

    def __load_simulation_rule_base__(self) -> List[Dict]:
        """通用规则加载：用于诊断、预测等用途"""
        try:
            mysql_conn = MysqlConnector(**self.mysql_config)
            rules = mysql_conn.query("SELECT * FROM simulation_rules")
            logger.info(f"✅ 加载 simulation_rules 表成功，共 {len(rules)} 条启用规则")
            return rules
        except Exception as e:
            logger.error(f"❌ 加载 simulation_rules 表失败: {e}")
            return []

    def __load_sensitive_words__(self) -> Set[str]:
        """加载敏感词表（从 .txt 文件）"""
        txt_path = Path("data", "sensitive_data", "sensitive_words.txt")
        with txt_path.open("r", encoding="utf-8") as f:
            return set(f.read().splitlines())

    def __init_qwen_tokenizer__(self):
        """加载 Qwen Tokenizer（本地）"""
        tokenizer_path = Path("qwen_tokenizer").resolve()  # 统一成 Path 风格
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"✅ Qwen tokenizer loaded from {tokenizer_path}")
        return tokenizer

    def __load_dishes__(self) -> List[Dict]:
        """加载菜品数据"""
        try:
            dishes_data = loadJS("data/dishes.json")
            logger.info(f"成功加载 {len(dishes_data)} 道菜品")
            return dishes_data
        except Exception as e:
            logger.error(f"加载菜品数据失败: {e}")
            return []

    def __load_exercise_data__(self) -> Dict:
        """加载所有运动课程相关数据，返回一个大字典"""
        base_path = "data/exec_data/"
        try:
            data = {
                "sports_lessons": loadJS(base_path + "Sports_lesson.json"),
                "sports_lesson_exercise_course": loadJS(base_path + "Sports_lesson_exercise_course.json"),
                "exercise_course_action": loadJS(base_path + "exercise_course_Action.json"),
                "actions": loadJS(base_path + "action.json"),
            }
            logger.info(
                f"✅ 运动课程数据加载完成: 课程 {len(data['sports_lessons'])} 条, 动作 {len(data['actions'])} 条")
            return data
        except Exception as e:
            logger.error(f"❌ 运动课程数据加载失败: {e}")
            return {}

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
        os.environ["ZB_ENV"] = self.args.env
        # os.environ["env"] = self.args.env
        logger.info(f"Initialize args: {self.args}")

    def __init_langfuse__(self) -> Langfuse:
        """初始化 Langfuse 客户端"""
        langfuse_config = load_yaml(Path("config", "langfuse_config.yaml"))[self.args.env]
        lf_client = Langfuse(
            secret_key=langfuse_config["LANGFUSE_SECRET_KEY"],
            public_key=langfuse_config["LANGFUSE_PUBLIC_KEY"],
            host=langfuse_config["LANGFUSE_HOST"]
        )
        logger.info(langfuse_config)
        logger.info("Langfuse client initialized successfully.")
        return lf_client

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
        self.mem0_url = self.api_config.get("mem0")
        intent_aigcfunc_map = load_yaml(Path("config", "intent_aigcfunc_map.yaml"))
        self.multimodal_config = load_yaml(Path("config", "multimodal_config.yaml"))[self.args.env]

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

    def __load_weather_api_config__(self) -> dict:
        """加载天气 API 配置"""
        weather_api = load_yaml(Path("config", "weather_config.yaml"))["weather_api"]
        return weather_api

    def __load_jia_kang_bao_data__(self) -> List[Dict]:
        """
        加载家康宝数据

        返回:
            List[Dict]: 家康宝数据列表
        """
        jiakangbao_path = Path("data", "qa_data", "jia_kang_bao_qa.json")  # 文件路径
        with jiakangbao_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def __load_health_labels__(self) -> Dict:
        """
        预加载健康标签数据，并构建索引，优化查询性能

        返回:
            Dict: 健康标签索引
        """
        """
            读取 health_labels.csv 并构建索引，提高查询速度
            """
        df = pd.read_excel(Path("doc", "客户标签体系", "health_labels.xlsx"), engine="openpyxl")

        indexed_data = {}
        for _, row in df.iterrows():
            label_name = row["标签名称"]
            value_name = row["值域"]

            if label_name not in indexed_data:
                indexed_data[label_name] = {}

            indexed_data[label_name][value_name] = {
                "group_code": row["组套代码"],
                "group_name": row["组套名称"],
                "nav_code": row["标签导航码"],
                "label_code": row["标签代码"],
                "label_name": label_name,
                "value_code": row["值域编码"],
                "value_name": value_name
            }

        return indexed_data

    @clock
    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data"""

        def filter_format(obj, splited=False):
            """格式化对象数据，处理换行符并可选地分割事件"""
            obj_str = json.dumps(obj, ensure_ascii=False).replace("\\r\\n", "\\n")
            obj_rev = json.loads(obj_str)
            if splited:
                for obj_rev_item in obj_rev:
                    if obj_rev_item.get("event"):
                        obj_rev_item["event"] = obj_rev_item["event"].split("\n")
            return obj_rev

        def search_target_version_item(item_list, ikey, curr_item_id, version):
            """从列表中返回指定version的item, 默认未指定则为latest"""
            try:
                # 获取指定的版本，如果没有则为None
                spec_version = version.get(curr_item_id, None)
                latest_item = None

                if spec_version:
                    # 查找指定版本的item
                    for item in item_list:
                        if item.get(ikey) == curr_item_id:
                            if item.get("version") == spec_version:
                                logger.debug(
                                    f"load spec version {ikey} - {curr_item_id} - {spec_version}"
                                )
                                return item
                            else:
                                # 如果不是指定版本，则记录为最新版本
                                latest_item = item
                        else:
                            continue
                else:
                    # 如果没有指定版本，查找最新版本的item
                    latest_item = next(
                        (i for i in item_list if i.get(ikey) == curr_item_id and i.get("version") == "latest"),
                        None
                    )

                # 返回找到的最新版本item
                if latest_item:
                    return latest_item
                else:
                    return None
            except IndexError as e:
                logger.error(
                    f"IndexError: {e}. item_list: {item_list}, ikey: {ikey}, curr_item_id: {curr_item_id}, version: {version}")
            except KeyError as e:
                logger.error(
                    f"KeyError: {e}. item_list: {item_list}, ikey: {ikey}, curr_item_id: {curr_item_id}, version: {version}")
            except Exception as e:
                logger.error(
                    f"Unexpected error: {e}. item_list: {item_list}, ikey: {ikey}, curr_item_id: {curr_item_id}, version: {version}")

            return None

        data_cache_file = self.cache_dir.joinpath("prompt_meta_data.pkl")

        # 检查是否使用缓存数据
        if not self.args.use_cache or not data_cache_file.exists():
            # 从数据库中查询数据
            mysql_conn = MysqlConnector(**self.mysql_config)
            prompt_meta_data = defaultdict(dict)

            # 查询各类prompt数据
            prompt_character = mysql_conn.query("select * from ai_prompt_character")
            prompt_event = mysql_conn.query("select * from ai_prompt_event")
            prompt_tool = mysql_conn.query("select * from ai_prompt_tool")
            prompt_intent_detect = mysql_conn.query("select * from ai_intent_detect")
            prompt_intent = mysql_conn.query("select * from ai_prompt_intent")

            # 格式化查询结果
            prompt_character = filter_format(prompt_character, splited=True)
            prompt_event = filter_format(prompt_event)
            prompt_tool = filter_format(prompt_tool)
            prompt_intent = filter_format(prompt_intent)
            prompt_intent_detect = filter_format(prompt_intent_detect)

            # 根据指定的版本构建prompt meta data
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
                            prompt_meta_data["intent_detect"] = {
                                i["name"]: i for i in prompt_intent_detect
                            }
                    else:
                        if key == "character":
                            for i in prompt_character:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = search_target_version_item(
                                    prompt_character, "name", i["name"], v
                                )
                        elif key == "event":
                            for i in prompt_event:
                                if prompt_meta_data[key].get(i["intent_code"]):
                                    continue
                                prompt_meta_data[key][i["intent_code"]] = search_target_version_item(
                                    prompt_event, "intent_code", i["intent_code"], v
                                )
                        elif key == "tool":
                            for i in prompt_tool:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = search_target_version_item(
                                    prompt_tool, "name", i["name"], v
                                )
                        elif key == "intent":
                            for i in prompt_intent:
                                if prompt_meta_data[key].get(i["name"]):
                                    continue
                                prompt_meta_data[key][i["name"]] = search_target_version_item(
                                    prompt_intent, "name", i["name"], v
                                )
            else:
                # 默认情况下构建prompt meta data
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
                prompt_meta_data["intent_detect"] = {i["name"]: i for i in prompt_intent_detect}

            # 初始化intent和tool的映射关系
            prompt_meta_data["init_intent"] = {
                i["code"]: True for i in prompt_tool if i["init_intent"] == 1
            }
            prompt_meta_data["rollout_tool"] = {
                i["code"]: 1 for i in prompt_tool if i["requirement"] == "rollout"
            }
            prompt_meta_data["rollout_tool_after_complete"] = {
                i["code"]: 1 for i in prompt_tool if i["requirement"] == "complete_rollout"
            }
            prompt_meta_data["prompt_tool_code_map"] = {
                i["code"]: i["name"] for i in prompt_tool if i["code"]
            }

            # 将数据缓存到本地文件
            pickle.dump(prompt_meta_data, open(data_cache_file, "wb"))
            logger.debug(f"dump prompt_meta_data to {data_cache_file}")
            del mysql_conn
        else:
            # 从本地缓存文件中加载数据
            prompt_meta_data = pickle.load(open(data_cache_file, "rb"))
            logger.debug(f"load prompt_meta_data from {data_cache_file}")

        # 处理tool的参数
        for name, func in prompt_meta_data["tool"].items():
            func["params"] = json.loads(func["params"]) if func["params"] else func["params"]

        # 构建intent描述的映射关系
        intent_desc_map = {
            code: item.get("intent_desc", "") for code, item in prompt_meta_data["event"].items() if item is not None
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


if __name__ == "__main__":
    InitAllResource()