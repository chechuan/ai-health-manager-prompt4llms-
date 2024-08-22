# -*- encoding: utf-8 -*-
"""
@Time    :   2023-10-20 13:45:09
@desc    :   call function script
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

import asyncio
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, AnyStr, Dict

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.prompt.task_schedule_manager import scheduleManager

sys.path.append(str(Path.cwd()))

from data.constrant import DEFAULT_DATA_SOURCE, ParamServer
from data.constrant_for_task_schedule import (
    query_schedule_template,
    query_schedule_template_v2,
)
from src.pkgs.knowledge.utils import (
    SearchQAChain,
    check_task,
    get_template,
    search_engine_chat,
)
from src.prompt.model_init import ChatMessage, callLLM
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    accept_stream_response,
    clock,
    curr_time,
    curr_weekday,
    date_after_days,
    this_sunday,
)


class FuncCall:
    headers: Dict = {"content-type": "application/json"}
    proxies = {"http": "socks5://127.0.0.1:7891", "https": "socks5://127.0.0.1:7891"}
    session = Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=2, method_whitelist=frozenset(["GET", "POST"]))
        ),
    )
    param_server: object = ParamServer()

    def __init__(self, gsr: InitAllResource = None):
        if not gsr:
            gsr = InitAllResource()
        self.api_config = gsr.api_config
        self.model_config = gsr.model_config
        self.prompt_meta_data = gsr.prompt_meta_data

        if gsr.args.use_proxy:
            self.session.proxies = self.proxies
            logger.info("FuncCall.session use proxy: " + str(self.proxies))
        self.scheduleManager = scheduleManager(gsr)
        self.register_for_all()
        self.scheduleManager.__init_vars__(self.funcmap, self.session)
        self.ext_api_factory = extApiFactory(self)
        # self._prepare_for_search_qa_chain()

    def _prepare_for_search_qa_chain(self) -> None:
        """Initlization 搜索qachain"""
        from langchain.llms import openai

        default_model_supplier = self.api_config["model_supply"]["default"]
        api_config = self.api_config["model_supply"][default_model_supplier]
        llm = openai.OpenAI(
            model_name=self.model_config.get("call_llm_with_search_engine"),
            openai_api_base=api_config["api_base"] + "/v1",
            openai_api_key=api_config["api_key"],
        )
        self.search_qa_chain = SearchQAChain.from_llm(
            llm=llm, return_only_outputs=False, verbose=False
        )

    def register_for_all(self):
        self.funcmap = {}
        self.funcname_map = {
            i["name"]: i["code"] for i in self.prompt_meta_data["tool"].values()
        }
        self.registration_list = []
        self.register_func(
            "searchKB", self.call_search_knowledge, "/chat/knowledge_base_chat"
        )
        self.register_func("searchDB", self.call_search_database)
        self.register_func("searchEngine", self.call_llm_with_search_engine)
        self.register_func(
            "get_schedule", self.call_get_schedule, "/alg-api/schedule/query"
        )
        self.register_func(
            "modify_schedule", self.call_schedule_modify, "/alg-api/schedule/manage"
        )
        self.register_func("askAPI", self.call_external_api)
        # self.register_func("create_schedule",   self.call_schedule_create,          "/alg-api/schedule/manage")
        # self.register_func("query_schedule",    self.call_schedule_query)
        # self.register_func("cancel_schedule",   self.call_schedule_cancel,          "/alg-api/schedule/manage")

        self.register_func(
            "create_schedule", self.scheduleManager.create, "/alg-api/schedule/manage"
        )
        self.register_func("query_schedule", self.scheduleManager.query)
        self.register_func(
            "cancel_schedule", self.scheduleManager.cancel, "/alg-api/schedule/manage"
        )
        # self.register_func("searchEngine",      self.call_llm_with_search_engine_v1)
        logger.success(f"Register {self.registration_list} Finish.")

    def register_func(
        self, func_name: AnyStr, func_call: Any, method: AnyStr = ""
    ) -> None:
        """注册called func funcmap"""
        self.funcmap[func_name] = {"func": func_call, "method": method}
        self.registration_list.append(func_name + ": " + str(func_call).split(" ")[2])

    def update_mid_vars(
        self,
        mid_vars,
        input_text=Any,
        output_text=Any,
        key="节点名",
        model="调用模型",
        **kwargs,
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

    @clock
    def call_get_schedule(self, *args, **kwds):
        """查询用户实时待办日程，默认查询未来两周"""
        func_item = self.funcmap["get_schedule"]
        url = self.api_config["ai_backend"] + func_item["method"]
        assert kwds.get("orgCode"), KeyError("orgCode is required")
        assert kwds.get("customId"), KeyError("customId is required")

        cur_time = (
            kwds.get("startTime") if kwds.get("startTime") else curr_time()
        )  # 查询未来两周的日程
        end_time = kwds.get("endTime") if kwds.get("endTime") else date_after_days(14)
        payload = {
            "orgCode": kwds["orgCode"],
            "customId": kwds["customId"],
            "startTime": cur_time,
            "endTime": end_time,
        }
        resp_js = self.session.post(url, json=payload, headers=self.headers).json()
        data = resp_js["data"]
        schedule = [
            {"task": i["taskName"], "time": i["cronDate"]}
            for i in data
            if i["taskName"]
        ]
        set_str = set([json.dumps(i, ensure_ascii=False) for i in schedule])
        schedule = [json.loads(i) for i in set_str]
        schedule = list(
            sorted(schedule, key=lambda item: item["time"])
        )  # 增加对查到的日程按时间排序
        self.update_mid_vars(
            kwds["mid_vars"],
            key="调用查询日程接口",
            input_text=payload,
            output_text=data,
            model="算法后端",
        )
        return schedule

    def call_schedule_create(self, *args, **kwds):
        """调用创建日程接口
        orgCode     String	组织编码
        customId    String	客户id
        taskName	String	任务内容
        taskType	String	任务类型（reminder/clock）
        taskDesc	String	任务备注
        intentCode	String	意图编码 `CREATE`新建提醒 `CHANGE`更改提醒 `CANCEL`取消提醒
        repeatType	String	提醒的频率 `EVERYDAY`每天 `W3`每周三 `M3`每月三号
        cronDate	Date	执行时间
        fromTime	Date	变更原始时间
        toTime	    Date	变更目的时间
        """
        msg = ChatMessage(**kwds["out_history"][-1])
        # schedule = self.funcmap["get_schedule"]['func'](**kwds)
        arguments = eval(msg.function_call["arguments"])
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert cur_time, "time is None"

        func_item = self.funcmap["create_schedule"]
        url = self.api_config["ai_backend"] + func_item["method"]
        input_payload = {
            "customId": customId,
            "orgCode": orgCode,
            "taskName": task,
            "taskType": "reminder",
            "intentCode": "CREATE",
            "cronDate": cur_time,
        }
        payload = {**input_payload}
        headers = {"content-type": "application/json"}
        response = self.session.post(url, json=payload, headers=headers).text
        resp_js = json.loads(response)

        if resp_js["code"] == 200:
            msg = eval(msg.function_call["arguments"])
            if msg.get("ask"):
                content = msg["ask"]
            else:
                content = "日程创建成功"
            logger.info(
                f"Create schedule org: {orgCode} - uid: {customId} - {cur_time} {task}"
            )
        else:
            content = "抱歉，日程创建失败，请稍后再试"
            logger.exception(f"日程创建失败: \n{resp_js}")
        self.update_mid_vars(
            kwds["mid_vars"],
            key="调用创建日程接口",
            input_text=payload,
            output_text=resp_js,
            model="算法后端",
        )
        return content

    def call_schedule_cancel(self, *args, **kwds):
        """取消日程"""
        msg = ChatMessage(**kwds["out_history"][-1])
        schedule = self.funcmap["get_schedule"]["func"](**kwds)
        try:
            arguments = eval(msg.function_call["arguments"])
        except Exception as err:
            arguments = msg.function_call["arguments"]
        task = arguments.get("task")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert customId, "customId is None"
        assert orgCode, "orgCode is None"
        assert len(schedule) != 0, "no schedule can be canceled"

        try:
            cronDate = [i for i in schedule if i["task"] == task][0]["time"]
        except Exception as err:
            logger.exception(err)
            logger.error(
                f"日程取消失败,目标操作日程 `{task}` not in current schedule {schedule}"
            )
            return f"{task}日程取消失败"

        func_item = self.funcmap["cancel_schedule"]
        url = self.api_config["ai_backend"] + func_item["method"]
        payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CANCEL",
            "cronDate": cronDate,
        }
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        if resp_js["code"] == 200:
            logger.info(f"Cancle schedule org:{orgCode} - uid:{customId} - {task}")
            content = f"已为您取消{task}的提醒"
        else:
            content = "日程取消失败, 请重试"
        self.update_mid_vars(
            kwds["mid_vars"],
            key="调用取消日程接口",
            input_text=payload,
            output_text=resp_js,
            model="算法后端",
        )
        return content

    def call_schedule_modify(self, *args, **kwds):
        """修改日程时间， 当前算法后端逻辑应该是根据task和from time查询 都改为toTime"""
        msg = ChatMessage(**kwds["out_history"][-1])
        schedule = self.funcmap["get_schedule"]["func"](**kwds)
        if isinstance(msg.function_call["arguments"], str):
            arguments = eval(msg.function_call["arguments"])
        elif isinstance(msg.function_call["arguments"], dict):
            arguments = msg.function_call["arguments"]
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, ValueError("修改日程 task can't be none")
        assert cur_time, ValueError("修改日程 time can't be none")

        try:
            task_time_ori = [i for i in schedule if i["task"] == task][0]["time"]
        except Exception as err:
            logger.exception(err)
            logger.error(
                f"日程修改失败, 目标操作日程 `{task}` not in current schedule {schedule}"
            )
            # return f"{task}日程修改失败"
            return f"对不起, 我没有理解您的意思, 请完善信息并重新尝试"

        func_item = self.funcmap["modify_schedule"]
        url = self.api_config["ai_backend"] + func_item["method"]
        payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CHANGE",
            "fromTime": task_time_ori,
            "toTime": cur_time,
        }
        logger.debug(f"修改日程 call api Input:\n{payload}")
        response = self.session.post(url, json=payload, headers=self.headers).text
        logger.debug(f"修改日程 call api Output:\n{response}")
        resp_js = json.loads(response)
        if resp_js["code"] == 200:
            changed_time = cur_time[11:]
            content = f"{task}提醒时间修改为{changed_time}"
            logger.info(
                f"Change schedule org:{orgCode} - uid:{customId} - {task} from {task_time_ori} to {cur_time}."
            )
        else:
            content = "日程修改失败"
        self.update_mid_vars(
            kwds["mid_vars"],
            key="调用取消日程接口",
            input_text=payload,
            output_text=resp_js,
            model="算法后端",
        )
        return content

    def call_schedule_query(self, *args, **kwds):
        """查询用户日程处理逻辑

        Note:
            1. 仅查询当日未完成日程
        """

        def compose_today_schedule(schedule, **kwds):
            """组装用户当日日程
            "您还有5项日程需要完成\n",
            "事项：血压测量，时间：8:00、20:00\n",
            "事项：三餐，时间：7：00、11：00、17：00\n",
            "事项：会议14：00，提前15min 提醒时间：14：00\n",
            "事项：用药，时间：21：00\n",
            "事项：慢走20min，今日完成，时间：21：00\n\n",
            """
            prompt = f"用户日程为：\n您还有{len(schedule)}项日程需要完成\n"
            schedule = list(sorted(schedule, key=lambda item: item["time"]))
            task_dict = {i["task"]: "" for i in schedule}
            for sch in schedule:
                task_dict[sch["task"]] += sch["time"][11:-3]
            prompt += "\n".join(
                [
                    f"事项：{task_name}，时间：{time}"
                    for task_name, time in task_dict.items()
                ]
            )
            prompt += "\n\n日程提醒:\n"
            return prompt

        model = kwds.get("model", "Qwen-14B-Chat")
        schedule = self.funcmap["get_schedule"]["func"](**kwds)
        cur_time, end_time = curr_time(), date_after_days(14)
        prompt = query_schedule_template.replace("{{cur_time}}", cur_time)
        today_schedule = [
            i
            for i in schedule
            if i["time"][:10] == cur_time[:10] and i["time"] > cur_time
        ]
        history = [{"role": "system", "content": prompt}]
        user_input = compose_today_schedule(today_schedule)
        history.append({"role": "user", "content": user_input})
        logger.debug(f"查询用户日程 LLM Input:\n{history}")
        raw_content = callLLM(history=history, top_p=0.8, temperature=0.7, model=model)
        logger.debug(f"查询用户日程 LLM Output:\n{raw_content}")
        self.update_mid_vars(
            kwds["mid_vars"],
            key=f"查询用户日程",
            input_text=history,
            output_text=raw_content,
            model=model,
        )
        return raw_content

    def call_search_knowledge_decorate_query(self, query: str) -> str:
        """优化要查询的query"""
        model = self.model_config.get("decorate_search_prompt", "Qwen-14B-Chat")
        system_prompt = "我的问题是: {query}\n我需要查询关于此问题哪些方面的知识?请以列表的格式给出最多三条问题,并保持静默模式"
        prompt = system_prompt.format(query=query)
        his = [{"role": "user", "content": prompt}]
        logger.debug(f"装饰知识库查询query LLM Input:\n{his}")
        content = callLLM(history=his, temperature=0.7, top_p=0.8, model=model)
        logger.debug(f"装饰知识库查询query LLM Output:\n{content}")
        return content

    def call_search_knowledge(
        self,
        *args,
        local_doc_url=False,
        stream=False,
        score_threshold=1,
        temperature=0.7,
        top_k=3,
        top_p=0.8,
        knowledge_base_name="高血压",
        prompt_name="default",
        model_name="Qwen-14B-Chat",
        **kwargs,
    ) -> AnyStr:
        """使用默认参数调用知识库"""
        called_method = self.funcmap["searchKB"]["method"]
        try:
            params = json.loads(args[0])
            query = params["query"]
            knowledge_base_name = params.get("knowledge_base_name", knowledge_base_name)
        except:
            query = args[0]

        payload = {}
        # payload['query'] = self.call_search_knowledge_decorate_query(query) + "\n" + query
        payload["query"] = query
        payload["knowledge_base_name"] = knowledge_base_name  # TODO 让模型选择知识库
        payload["local_doc_url"] = local_doc_url
        payload["model_name"] = model_name
        payload["score_threshold"] = score_threshold
        payload["stream"] = stream
        payload["temperature"] = temperature
        payload["top_k"] = top_k
        payload["top_p"] = top_p
        payload["prompt_name"] = prompt_name

        dataSource = None
        url = self.api_config["langchain"] + called_method
        try:
            logger.debug(f"Call 知识库问答 with payload\n{payload}")
            response = self.session.post(url, json=payload, headers=self.headers)
            msg = eval(response.text)
            logger.debug(f"知识库问答 Response\n{msg}")
        except Exception as e:
            logger.error(e)
            content = "对不起,当前知识服务连接存在一些小问题,请稍后再试"
            ret = {"content": content, "dataSource": dataSource}
            return ret

        if (
            "未找到相关文档" in msg["answer"]
            or "未找到相关文档" in msg["docs"][0]
            or "无法回答" in msg["answer"]
            or not msg["docs"]
        ):
            content = msg["answer"]
            dataSource = "语言模型"
            self.update_mid_vars(
                kwargs["mid_vars"],
                key=f"查询知识库",
                input_text=query,
                output_text=msg,
                model=model_name,
            )

            # 知识库未查到,可能是阈值过高或者知识不匹配,使用搜索引擎做保底策略
            # try:
            #     content = self.funcmap["searchEngine"]["func"](query, **kwargs).strip()
            #     dataSource = "搜索引擎"
            # except:
            #     content = "对不起,没有检索到相关答案,请稍后再试"
            #     dataSource = "语言模型"
            # self.update_mid_vars(
            #     kwargs["mid_vars"],
            #     key=f"搜索引擎",
            #     input_text=query,
            #     output_text=content,
            #     model="baidu crawler",
            # )
        else:
            doc_name_list = [
                re.findall("\[.*?\]", msg["docs"][1][7:])[0][1:-1] for i in msg["docs"]
            ]
            doc_name_list = list(set([i.split(".")[0] for i in doc_name_list]))
            dataSource = "、".join(doc_name_list)
            content = msg["answer"].strip()
            try:
                self.update_mid_vars(
                    kwargs["mid_vars"],
                    key=f"知识库问答",
                    input_text=payload,
                    output_text=msg,
                    model="langchain",
                )
            except:
                pass

        ret = {"content": content, "dataSource": dataSource}
        return ret

    def call_search_engine(self, *args, **kwargs) -> AnyStr:
        """调用搜索引擎"""
        query = args[0] if args else kwargs.get("keywords")
        search_result = asyncio.run(
            search_engine_chat(
                query,
                top_k=kwargs.get("top_k", 3),
                max_length=kwargs.get("max_length", 500),
                session=self.session,
                return_list=True,
            )
        )
        return search_result

    def call_llm_with_search_engine(
        self, *args, model_name="Qwen-14B-Chat", **kwargs
    ) -> AnyStr:
        """llm + 搜索引擎

        使用src/pkgs/knowledge/config/prompt_config.py中定义的拼接模板 (from langchain-Chatchat)
        """
        try:
            params = json.loads(args[0])
            query = params["query"]
        except:
            query = args[0]
        logger.debug(f"搜索引擎 Input: {query}")
        # search_result = asyncio.run(search_engine_chat(query, top_k=kwargs.get("top_k", 3), max_length=500,session=self.session))
        search_result = search_engine_chat(
            query, top_k=kwargs.get("top_k", 3), max_length=500, session=self.session
        )
        logger.debug(f"搜索引擎检索 Output:\n{search_result}")
        template = get_template("search_engine_chat")
        if search_result:
            template = (
                template["search"]
                .strip()
                .replace("{{ context }}", "\n" + search_result + "\n")
            )
            self.update_mid_vars(
                kwargs["mid_vars"],
                key=f"查询搜索引擎",
                input_text=query,
                output_text=search_result,
                model="baidu crawler",
            )
        else:
            template = template["Empty"]
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": query},
        ]

        logger.debug(f"结合搜索内容 LLM Input: {messages}")
        content = callLLM(
            history=messages, model_name=model_name, temperature=0.7, top_p=0.8
        )
        logger.debug(f"结合搜索内容 LLM Output:\n{content}")
        self.update_mid_vars(
            kwargs["mid_vars"],
            key=f"搜索引擎 -> LLM",
            input_text=messages,
            output_text=content,
            model=model_name,
        )
        ret_obj = {"content": content, "dataSource": "搜索引擎"}
        return ret_obj

    def call_llm_with_search_engine_v1(self, *args, **kwargs) -> AnyStr:
        """自定义SearchQAChain实现搜索引擎问答"""
        qa_outputs = self.search_qa_chain.run(query=args[0], max_results=3)
        content = qa_outputs[self.search_qa_chain.output_key]
        self.update_mid_vars(
            kwargs["mid_vars"],
            key=f"搜索引擎",
            input_text=args[0],
            output_text=qa_outputs["search_results"],
            model="duckduckgo",
        )
        self.update_mid_vars(
            kwargs["mid_vars"],
            key=f"search_qa_chain",
            input_text=qa_outputs["qa_content"],
            output_text=content,
        )

        ret_obj = {"content": content, "dataSource": "搜索引擎"}
        return ret_obj

    def call_llm_with_graph(self, *args, **kwargs) -> AnyStr:
        """ """
        called_method = "/v1/subgraph_qa"
        payload = self.param_server.llm_with_graph
        payload["messages"][0]["content"] = args[0]["query"]
        response = self.session.post(
            self.api_config["graph"] + called_method, json=payload, headers=self.headers
        )
        content = eval(response.text)["result"]
        ret_obj = {"content": content, "dataSource": "知识图谱"}
        return ret_obj

    def call_search_database(self, *args, **kwargs) -> AnyStr:
        """调用查询数据库接口"""
        return "暂不支持查询db"

    def call_external_api(self, *args, **kwargs):
        """调用外部api"""
        task: str = check_task(json.loads(args[0]))
        param_desc = self.prompt_meta_data["tool"]["调用接口"]["params"]

        candidate_task = [
            j for i in param_desc for j in i["optional"] if i["name"] == "task"
        ]
        assert (
            task in candidate_task
        ), f"Generate task: {task} not in the candidate_task {candidate_task}"
        content = self.ext_api_factory._call(*args, **kwargs, task=task, func_cls=self)

        ret_obj = {"content": content, "dataSource": "外部api"}
        return ret_obj

    def _call(self, **kwargs):
        """"""
        history = copy.deepcopy(kwargs["out_history"])
        function_call = history[-1]["function_call"]
        func_name = function_call["name"]
        arguments = function_call["arguments"]
        assert (
            self.funcmap.get(func_name) is not None
        ), f"Unregistered function {{{func_name}}}, 请重试"
        try:
            ret_obj = self.funcmap[func_name]["func"](arguments, **kwargs)
        except Exception as e:
            logger.exception(e)
            ret_obj = f"工具调用失败, 请重试"
        if isinstance(ret_obj, str):
            content = ret_obj
            dataSource = DEFAULT_DATA_SOURCE
        elif isinstance(ret_obj, dict):
            content = ret_obj["content"]
            dataSource = ret_obj["dataSource"]
        history.append(
            {
                "role": "assistant",
                "content": content,
                "intentCode": kwargs["intentCode"],
            }
        )
        # logger.debug(f"Observation: {content}")
        return history, dataSource


class extApiFactory:
    def __init__(self, *args, **kwargs) -> None:
        self.api_config = args[0].api_config
        self.session = args[0].session

    @staticmethod
    def __extract_user_message__(*args, **kwargs) -> dict:
        """提取格式化用户信息"""
        return {}

    @staticmethod
    def __diet_compose_nutr__(body: Dict) -> AnyStr:
        """拼接饮食 营养元素部分内容"""
        nutr_target = body["nutr_target"]
        nutr_extra = body["nutr_target_extra"]

        nutr_sum = copy.deepcopy(nutr_target)
        nutr_sum = {k: round(v + nutr_extra.get(k, 0), 1) for k, v in nutr_sum.items()}

        content = "推荐您今日四大营养素摄入量:"
        content += (
            f"碳水{nutr_sum['heat']}千焦,"
            f"蛋白{nutr_sum['protein']}g,"
            f"脂肪{nutr_sum['fat']}g,"
            f"碳水{nutr_sum['carbonWater']}g. "
        )
        heat_rec = [round(i * nutr_target["heat"], 1) for i in [0.3, 0.4, 0.3]]
        content += (
            f"建议您早餐热量摄入{heat_rec[0]}千焦,"
            f"午餐热量摄入{heat_rec[1]}千焦,"
            f"晚餐热量摄入{heat_rec[2]}千焦. "
        )
        content += "各类膳食建议摄入量:"
        content += ",".join(
            [
                f"{item['name']}: {round(item['value'], 1)}克"
                for item in body["food_component_weight"]
                if item["name"] != "烹调油"
            ]
        )
        return content

    @staticmethod
    def __diet_compose_recipes__(body: Dict) -> AnyStr:
        """拼接饮食 食谱部分内容"""
        component_map = {i["tag"]: i["component"] for i in body["food_component_list"]}
        content = ""
        meal_order_map = {
            "food_time_morning": "早餐",
            "food_time_noon": "午餐",
            "food_time_night": "晚餐",
            "food_time_extra": "加餐",
        }
        tag_map = {
            "主食": "recipe_type_zs",
            "豆鱼蛋肉": "recipe_type_zc",
            "蔬菜类": "recipe_type_sc",
            "奶类": "food_type_nai",
            "水果类": "food_type_shuiguo",
            "坚果": "food_type_jianguozhongzi",
        }
        for code, tag in meal_order_map.items():
            component = component_map[code]
            content += f"{tag}:\n"
            for i in component:
                if i["name"] == "烹调油":
                    continue
                tag_code = tag_map[i["name"]]
                detail_recipes = ",".join(
                    [
                        i["rname"]
                        for i in random.sample(
                            body["recipe_rec_detail"][code][tag_code], 3
                        )
                    ]
                )
                content += f"{i['name']}{i['unit']},例: {detail_recipes}\n"
        logger.info(content)
        return content

    def __call_diet_v5_compoent_list__(self, *args, **kwargs):
        """调用三剂处方api"""
        payload = {
            "baseInfoExtra": {"height": 173, "weight": 65},
            "baseInfoList": {
                "age": "20",
                "physicalActivity": "轻",
                "physiologicalStage": "",
                "religiousBelief": "汉族",
                "sex": "女",
            },
            "limitRecipesNums": 6,
            "is_ommon": "1",
        }
        url = self.api_config["algo_rec"] + "/component_list"
        headers = {"content-type": "application/json"}
        ret = self.session.post(url, json=payload, headers=headers).json()
        if ret["head"] == 200:
            content = ""
            content += extApiFactory.__diet_compose_nutr__(ret["body"])
            content += extApiFactory.__diet_compose_recipes__(ret["body"])

        else:
            logger.error(
                f"调用三济处方接口失败, 返回信息：{json.dumps(ret, ensure_ascii=False)}"
            )
            content = "调用三济处方接口失败"
        return content

    def _call(self, *args, **kwargs) -> dict:
        """处理调用外部api逻辑"""
        # TODO 接入综合饮食推荐接口
        task = kwargs["task"]
        if task == "三济饮食处方":
            user_msg = extApiFactory.__extract_user_message__(*args, **kwargs)
            content = self.__call_diet_v5_compoent_list__(
                *args, **kwargs, user_msg=user_msg
            )
        else:
            content = ""
        return content


if __name__ == "__main__":
    funcall = FuncCall()
    # function_call = {'name': 'searchKB', 'arguments': '后脑勺持续一个月的头疼'}
    function_call = {"name": "askAPI", "arguments": '{"task":"三济饮食处方"}'}
    funcall._call(out_history=[{"function_call": function_call}], verbose=True)
