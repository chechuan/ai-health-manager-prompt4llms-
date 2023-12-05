# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-20 13:45:09
@desc    :   call function script
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AnyStr, Dict

import yaml
from requests import Session

sys.path.append(str(Path.cwd()))

from config.constrant import ParamServer
from config.constrant_for_task_schedule import query_schedule_template
from src.pkgs.knowledge.utils import get_template, search_engine_chat
from src.prompt.model_init import ChatMessage, chat_qwen
from src.utils.Logger import logger
from src.utils.module import req_prompt_data_from_mysql


class funcCall:
    headers: Dict = {"content-type": "application/json"}
    session = Session()
    api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
    param_server: object = ParamServer()

    def __init__(self, prompt_meta_data=None, env="local"):
        self.prompt_meta_data = prompt_meta_data if prompt_meta_data else req_prompt_data_from_mysql(env)
        self.funcmap = {}
        self.funcname_map = {i['name']: i['code'] for i in self.prompt_meta_data['tool'].values()}
        self.register_func("searchKB",   self.call_search_knowledge,         "/chat/knowledge_base_chat")
        self.register_func("searchEngine",      self.call_llm_with_search_engine)
        self.register_func("get_schedule",      self.call_get_schedule,             "/alg-api/schedule/query")
        self.register_func("create_schedule",   self.call_schedule_create,          "/alg-api/schedule/manage")
        self.register_func("query_schedule",    self.call_schedule_query)
        self.register_func("cancel_schedule",   self.call_schedule_cancel,          "/alg-api/schedule/manage")
        self.register_func("modify_schedule",   self.call_schedule_modify,          "/alg-api/schedule/manage")
        logger.success(f"register finish.")

    def register_func(self, func_name: AnyStr, func_call: Any, method: AnyStr="") -> None:
        """注册called func funcmap
        """
        self.funcmap[func_name] = {"func": func_call, "method": method}
        logger.success(f"register tool {func_name}.")
    
    def update_mid_vars(self, mid_vars, input_text=Any, output_text=Any, key="节点名", model="调用模型", **kwargs):
        """更新中间变量
        """
        lth = len(mid_vars) + 1
        mid_vars.append({"id": lth, "key":key, "input_text": input_text, "output_text":output_text, "model":model, **kwargs})
        return mid_vars

    def call_get_schedule(self, *args, **kwds):
        """查询用户实时日程
        """
        func_item = self.funcmap['get_schedule']
        url = self.api_config["ai_backend"] + func_item['method']
        assert kwds.get("orgCode"), KeyError("orgCode is required")
        assert kwds.get("customId"), KeyError("customId is required")

        payload = {"orgCode": kwds.get("orgCode"),"customId": kwds.get("customId")}
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        data = resp_js['data']
        ret = [{"task": i['taskName'], "time": i['cronDate']} for i in data]
        set_str = set([json.dumps(i, ensure_ascii=False) for i in ret])
        ret = [json.loads(i) for i in set_str]
        self.update_mid_vars(kwds['mid_vars'], key="查询用户日程", input_text=payload, output_text=data, model="算法后端")
        return ret

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
        msg = ChatMessage(**kwds['out_history'][-1])
        # schedule = self.funcmap["get_schedule"]['func'](**kwds)
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert cur_time, "time is None"

        func_item = self.funcmap['create_schedule']
        url = self.api_config["ai_backend"] + func_item['method']
        input_payload = {
            "customId": customId,
            "orgCode": orgCode,
            "taskName": task,
            "taskType": "reminder",
            "intentCode": "CREATE",
            "cronDate": cur_time
        }
        payload = {**input_payload}
        headers = {"content-type": "application/json"}
        response = self.session.post(url, json=payload, headers=headers).text
        resp_js = json.loads(response)

        if resp_js["code"] == 200:
            content = eval(msg.function_call['arguments'])['ask']
            logger.info(f"Create schedule org: {orgCode} - uid: {customId} - {cur_time} {task}")
        else:
            content = "抱歉，日程创建失败，请稍后再试"
            logger.exception(f"日程创建失败: \n{resp_js}")
        self.update_mid_vars(kwds['mid_vars'], key="调用创建日程接口", input_text=payload, output_text=resp_js, model="算法后端")
        return content

    def call_schedule_cancel(self, *args, **kwds):
        """取消日程
        """
        msg = ChatMessage(**kwds['out_history'][-1])
        schedule = self.funcmap["get_schedule"]['func'](**kwds)
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert customId, "customId is None"
        assert orgCode, "orgCode is None"
        assert len(schedule) != 0, "no schedule can be canceled"
        
        cronDate = [i for i in schedule if i['task'] == task][0]['time']

        func_item = self.funcmap['cancel_schedule']
        url = self.api_config["ai_backend"] + func_item['method']
        payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CANCEL",
            "cronDate": cronDate
        }
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        if resp_js["code"] == 200:
            logger.info(f"Cancle schedule org:{orgCode} - uid:{customId} - {task}")
            content = f"已为您取消{task}的提醒"
        else:
            content = "日程取消失败, 请重试"
        self.update_mid_vars(kwds['mid_vars'], key="调用取消日程接口", input_text=payload, output_text=resp_js, model="算法后端")
        return content

    def call_schedule_modify(self, *args, **kwds):
        """修改日程时间， 当前算法后端逻辑应该是根据task和from time查询 都改为toTime
        """
        msg = ChatMessage(**kwds['out_history'][-1])
        schedule = self.funcmap["get_schedule"]['func'](**kwds)
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert cur_time, "time is None"

        task_time_ori = [i for i in schedule if i['task']==arguments['task']][0]['time']

        func_item = self.funcmap['modify_schedule']
        url = self.api_config["ai_backend"] + func_item['method']
        payload = {
                "customId": kwds.get("customId"),
                "orgCode": kwds.get("orgCode"),
                "taskName": task,
                "intentCode": "CHANGE",
                "fromTime": task_time_ori,
                "toTime": cur_time
            }
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        if resp_js["code"] == 200:
            changed_time = cur_time[11:]
            content = f"{task}提醒时间修改为{changed_time}"
            logger.info(f"Change schedule org:{orgCode} - uid:{customId} - {task} from {task_time_ori} to {cur_time}.")
        else:
            content = "日程修改失败"
        self.update_mid_vars(kwds['mid_vars'], key="调用取消日程接口", input_text=payload, output_text=resp_js, model="算法后端")
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
            schedule = list(sorted(schedule, key=lambda item: item['time']))
            task_dict = {i['task']: "" for i in schedule}
            for sch in schedule:
                task_dict[sch['task']] += sch['time'][11:-3]
            prompt += "\n".join([f"事项：{task_name}，时间：{time}" for task_name, time in task_dict.items()])
            prompt += "\n\n日程提醒:\n"
            return prompt
        
        model = kwds.get("model", "Qwen-14B-Chat")
        schedule = self.funcmap["get_schedule"]['func'](**kwds)

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = query_schedule_template.replace("{{cur_time}}", cur_time)
        today_schedule = [i for i in schedule if i['time'][:10]==cur_time[:10] and i['time'] > cur_time]
        history = [{"role": "system", "content": prompt}]
        user_input = compose_today_schedule(today_schedule)
        history.append({"role": "user", "content": user_input})
        raw_content = chat_qwen(history=history, top_p=0.8, temperature=0.7, model=model)
        self.update_mid_vars(kwds['mid_vars'], key=f"查询用户日程", input_text=history, output_text=raw_content, model=model)
        return raw_content

    def call_search_knowledge(self, *args, local_doc_url=False, stream=False, 
                              score_threshold=0.5, temperature=0.7, top_k=3, top_p=0.8, 
                              knowledge_base_name="内科学", prompt_name="default", 
                              model_name="Qwen-14B-Chat", 
                              **kwargs) -> AnyStr:
        """使用默认参数调用知识库
        """
        def decorate_search_prompt(query: str) -> str:
            """优化要查询的query"""
            prompt = (
                "You are a powerful assistant, capable of understanding requirements and responding accordingly."
                "# 要求\n"
                "1. 提取其中关键信息并作出解释\n"
                "2. 针对问题给出相关的有用的信息\n"
                "3. 组装成一段话,要求语义连贯,适当简洁\n"
            )
            his = [{"role": "system", "content": prompt}, {"role":"user", "content": query}]
            content = chat_qwen(history=his, temperature=0.7, top_p=0.8, model="Qwen-1_8B-Chat")
            return content

        called_method = self.funcmap['searchKB']['method']
        query = args[0]
        payload = {}
        payload['query'] = decorate_search_prompt(query)
        payload["knowledge_base_name"] = knowledge_base_name    # 让模型选择知识库
        payload["local_doc_url"] = local_doc_url
        payload["model_name"] = model_name
        payload["score_threshold"] = score_threshold
        payload["stream"] = stream
        payload["temperature"] = temperature
        payload["top_k"] = top_k
        payload["top_p"] = top_p
        payload["prompt_name"] = prompt_name
        
        url = self.api_config['langchain']+called_method
        response = self.session.post(url, json=payload, headers=self.headers)
        msg = eval(response.text)
        
        if "未找到相关文档" not in msg['docs'][0]:
            content = msg['answer']
            self.update_mid_vars(kwargs['mid_vars'], 
                                 key=f"查询知识库", 
                                 input_text=query, 
                                 output_text=msg, 
                                 model=model_name)
        else:   # 知识库未查到,可能是阈值过高或者知识不匹配,使用搜索引擎做保底策略
            content = self.call_llm_with_search_engine(query)
        return content

    def call_llm_with_search_engine(self, *args, model_name="Qwen-14B-Chat", **kwargs) -> AnyStr:
        """llm + 搜索引擎
        
        使用src/pkgs/knowledge/config/prompt_config.py中定义的拼接模板 (from langchain-Chatchat)
        """
        query = args[0]
        search_result = asyncio.run(search_engine_chat(query, 
                                                       top_k=kwargs.get("top_k", 3), 
                                                       max_length=500,
                                                       session=self.session))

        template = get_template("search_engine_chat")
        if search_result:
            template = template["search"].strip().replace("{{ context }}", "\n"+search_result+"\n")
            self.update_mid_vars(kwargs['mid_vars'], 
                                 key=f"查询搜索引擎", 
                                 input_text=query, 
                                 output_text=search_result, 
                                 model="baidu crawler")
        else:
            template = template["Empty"]
        prompt = template.replace("{{ question }}", search_result)

        content = chat_qwen(prompt, model_name=model_name, temperature=0.7, top_p=0.8)
        self.update_mid_vars(kwargs['mid_vars'], 
                             key=f"搜索引擎 -> LLM", 
                             input_text=prompt, 
                             output_text=content, 
                             model=model_name)
        return content
    
    def call_llm_with_graph(self, *args, **kwargs) -> AnyStr:
        """
        """
        called_method = "/v1/subgraph_qa"
        payload = self.param_server.llm_with_graph
        payload['messages'][0]['content'] = args[0]['query']
        response = self.session.post(self.api_config['graph']+called_method,
                                     json=payload,
                                     headers=self.headers)
        res_js = eval(response.text)
        ret = res_js['result']
        return ret

    def _call(self, **kwargs):
        """"""
        history = kwargs["out_history"]
        function_call = history[-1]['function_call']
        func_name = function_call["name"]
        arguments = function_call["arguments"]
        assert self.funcmap.get(func_name) is not None, f"Unregistered function {{{func_name}}}, 请重试"
        content = self.funcmap[func_name]['func'](arguments, **kwargs)
        history.append({"role": "user","content": content})
        logger.debug(f"Observation: {content}")
        return history

if __name__ == "__main__":
    funcall = funcCall()
    function_call = {'name': 'searchKB', 'arguments': '后脑勺持续一个月的头疼'}
    funcall._call(out_history=[{"function_call":function_call}], verbose=True)
