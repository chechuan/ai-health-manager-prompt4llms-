# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 09:58:18
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

# 标准库导入
import argparse
import json
import pickle
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

# 第三方库导入
import openai
import requests

# 本地模块导入
from data.constrant import CACHE_DIR
from src.utils.module import clock, load_yaml, loadJS
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

        self.prompt_meta_data = self.req_prompt_data_from_mysql()

        self.__init_model_supplier__()
        self.client = openai.OpenAI()
        self.aclient = openai.AsyncOpenAI()

        self.weather_api_config = self.__load_weather_api_config__()

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

    def __load_weather_api_config__(self) -> dict:
        """加载天气 API 配置"""
        weather_api = load_yaml(Path("config", "weather_config.yaml"))["weather_api"]
        return weather_api

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
            prompt_intent = mysql_conn.query("select * from ai_prompt_intent")

            # 格式化查询结果
            prompt_character = filter_format(prompt_character, splited=True)
            prompt_event = filter_format(prompt_event)
            prompt_tool = filter_format(prompt_tool)
            prompt_intent = filter_format(prompt_intent)

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