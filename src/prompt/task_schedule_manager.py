# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-27 13:35:10
@desc    :   日程管理pipeline
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import AnyStr, Dict, List

import openai
import requests
import yaml
from langchain.prompts import PromptTemplate

sys.path.append(".")
from config.constrant import task_schedule_return_demo
from config.constrant_for_task_schedule import (
    query_schedule_template, task_schedule_parameter_description,
    task_schedule_parameter_description_for_qwen)
from src.prompt.model_init import ChatCompletionRequest, chat_qwen
from src.prompt.qwen_openai_api import create_chat_completion
from src.utils.Logger import logger


def call_qwen(messages, functions=None):
    model_name = "Qwen-14B-Chat"
    openai.api_base = "http://10.228.67.99:26926/v1"
    openai.api_key = "empty"
    # logger.debug(messages)
    if functions:
        response = openai.ChatCompletion.create(model=model_name, messages=messages, functions=functions)
    else:
        response = openai.ChatCompletion.create(model=model_name, messages=messages)
    logger.debug("Assistant:\t", response.choices[0].message.content)
    messages.append(json.loads(json.dumps(response.choices[0].message, ensure_ascii=False)))
    return messages

class taskSchedulaManager:
    headers: Dict = {"content-type": "application/json"}
    payload_template: Dict = {
        "customId": None,"orgCode": None,"taskName": None,"taskType": None,"taskDesc": None,
        "intentCode": None, "repeatType": None, "cronDate": None, "fromTime": None, "toTime": None
        }
    def __init__(self, 
                 api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local'],
                 prompt_meta_data: Dict[str, Dict] = {}
                 ):
        """日程管理模块
        - Args
            
            api_config (Dict, required)
                api配置, 包含llm, langchain, graph_qa, ai_backend
            prompt_meta_data (Dict[str, Dict], required)
                prompt meta data,分立的prompt元数据
        """
        self.api_config = api_config
        self.prompt_meta_data = prompt_meta_data
        # prompt = PromptTemplate(
        #     input_variables=["task_schedule_return_demo", "task_schedule_parameter_description", "curr_plan", "tmp_time"], 
        #     template=TEMPLATE_TASK_SCHEDULE_MANAGER
        # )
        prompt = PromptTemplate(
            input_variables=["task_schedule_return_demo", "task_schedule_parameter_description", "curr_plan", "tmp_time"], 
            template=self.prompt_meta_data['tool']['日程管理']['description']
        )
        self.sys_prompt = prompt.format(
            task_schedule_return_demo=task_schedule_return_demo,
            task_schedule_parameter_description=task_schedule_parameter_description,
            curr_plan=[],
            tmp_time=datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
            )
        self.conv_prompt = PromptTemplate(
            input_variables=["input"],
            template="用户输入: {input}\n你的输出(json):"
            )
        self.session = requests.Session()
    
    def run(self, query, **kwds):
        content = self.sys_prompt + self.conv_prompt.format(input=query)
        if kwds.get("verbose"):
            logger.debug(content)
        ret = chat_qwen(content)
        logger.debug(eval(ret))
        return ret

    def tool_ask_for_time(self, messages, msg):
        content = input(f"tool input(晚上7点半): ")
        messages.append(
            {
                "role": "assistant", 
                "content": msg.content, 
                "function_call": {"name":msg.function_call['name'],"arguments":  content},
                "schedule": self.get_init_schedule()
            }
        )
        return messages

    def tool_create_schedule(self, msg, **kwds):
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
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert cur_time, "time is None"

        url = self.api_config['ai_backend'] + "/alg-api/schedule/manage"
        input_payload = {
            "customId": customId,
            "orgCode": orgCode,
            "taskName": task,
            "taskType": "reminder",
            "intentCode": "CREATE",
            "cronDate": cur_time
        }
        payload = {**self.payload_template, **input_payload}
        headers = {"content-type": "application/json"}
        response = self.session.post(url, json=payload, headers=headers).text
        resp_js = json.loads(response)

        assert resp_js["code"] == 200, resp_js["msg"]
        logger.info(f"Create schedule org:{{{orgCode}}} - uid:{{{customId}}} - {cur_time} {task}")
        return 200
    
    def tool_cancel_schedule(self, msg, **kwds):
        """取消日程
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
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert customId, "customId is None"
        assert orgCode, "orgCode is None"

        url = self.api_config['ai_backend'] + "/alg-api/schedule/manage"
        input_payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CANCEL"
        }
        payload = {**self.payload_template, **input_payload}
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        assert resp_js["code"] == 200, resp_js["msg"]
        logger.info(f"Cancle schedule org:{{{orgCode}}} - uid:{{{customId}}} - {task}")
        return task

    def tool_modify_schedule(self, msg, schedule, **kwds):
        """修改日程时间， 当前算法后端逻辑应该是根据task和from time查询 都改为toTime
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
        arguments = eval(msg.function_call['arguments'])
        task = arguments.get("task")
        cur_time = arguments.get("time")
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        assert task, "task name is None"
        assert cur_time, "time is None"

        task_time_ori = [i for i in schedule if i['task']==arguments['task']][0]['time']

        url = self.api_config['ai_backend'] + "/alg-api/schedule/manage"
        input_payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CHANGE",
            "fromTime": task_time_ori,
            "toTime": cur_time
        }
        payload = {**self.payload_template, **input_payload}
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        assert resp_js["code"] == 200, resp_js["msg"]
        logger.info(f"Change schedule org:{{{orgCode}}} - uid:{{{customId}}} - {task} from {task_time_ori} to {cur_time}.")
        return 200
    
    def compose_today_schedule(self, schedule, **kwds):
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

    def tool_query_schedule(self, schedule, mid_vars_item, **kwds):
        """查询用户日程处理逻辑

        Note:
            1. 仅查询当日未完成日程
        """
        # prompt = ("以下是用户的日程及对应时间,请组织语言,告知用户,请遵循以下几点要求:\n"
        #           "1.尽可能语句通顺,上下文连贯且对话术对用户友好\n"
        #           "2.除了要告知用户的日程信息不要输出任何其他内容\n"
        #           "3.请按照时间戳的先后顺序输出\n\n")
        # prompt += f"{schedule}\n\n总结查询到的日程:"
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = query_schedule_template.replace("{{cur_time}}", cur_time)
        today_schedule = [i for i in schedule if i['time'][:10]==cur_time[:10] and i['time'] > cur_time]
        history = [{"role": "system", "content": prompt}]
        user_input = self.compose_today_schedule(today_schedule)
        history.append({"role": "user", "content": user_input})
        raw_content = chat_qwen(history=history, top_p=0.8, temperature=0.7)
        mid_vars_item.append({"key":"总结查询到的日程", "input_text": history, "output_text": raw_content})
        return raw_content, mid_vars_item

    def get_real_time_schedule(self, **kwds):
        """查询用户实时日程
        """
        assert kwds.get("orgCode"), KeyError("orgCode is required")
        assert kwds.get("customId"), KeyError("customId is required")

        url = self.api_config["ai_backend"] + "/alg-api/schedule/query"
        payload = {"orgCode": kwds.get("orgCode"),"customId": kwds.get("customId")}
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        data = resp_js['data']
        ret = [{"task": i['taskName'], "time": i['cronDate']} for i in data]
        set_str = set([json.dumps(i, ensure_ascii=False) for i in ret])
        ret = [json.loads(i) for i in set_str]
        return ret
    
    def _run(self, messages: List[Dict], **kwds):
        """对话过程以messages形式利用历史信息
        - Args
        
            messages (List[Dict])
                历史信息 包括user/assistant/function_call
            kwds (keyword arguments)
        
        - return
            output (str) 
                直接输出的文本
        """
        schedule = self.get_real_time_schedule(**kwds)
        request = ChatCompletionRequest(model="Qwen-14B-Chat", 
                                        functions=task_schedule_parameter_description_for_qwen,
                                        messages=messages,)
        response, mid_vars_item = create_chat_completion(request, schedule)
        msg = response.choices[0].message
        # if kwds.get("verbose"):
        logger.debug("Thought:" + msg.content[msg.content.rfind("Thought:"):])
        if msg.function_call:
            logger.debug(f"call func: {msg.function_call['name']}")
            logger.debug(f"arguments: {msg.function_call['arguments']}")
        if msg.function_call:
            if msg.function_call['name'] == "ask_for_time":
                content = eval(msg.function_call['arguments'])['ask'] if msg.function_call else msg.content
            elif msg.function_call['name'] == "create_schedule":
                self.tool_create_schedule(msg, **kwds)
                content = eval(msg.function_call['arguments'])['ask']
            elif msg.function_call['name'] == "modify_schedule":
                self.tool_modify_schedule(msg, schedule, **kwds)
                content = eval(msg.function_call['arguments'])['ask']
            elif msg.function_call['name'] == "cancel_schedule":
                task = self.tool_cancel_schedule(msg, **kwds)
                content = f"已为您取消{task}的提醒"
            elif msg.function_call['name'] == "query_schedule":
                content, mid_vars_item = self.tool_query_schedule(schedule, mid_vars_item, **kwds)
            else:
                content = "我不清楚你想做什么,请稍后重试"
        else:
            content = "我不清楚你想做什么,请稍后重试"
        return content, mid_vars_item

if __name__ == "__main__":
    t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    tsm = taskSchedulaManager()
    t_claim_str = f"现在的时间是{t}\n"
    orgCode="sf"
    customId="007"
    # debug 任务时间
    # content = f"{t_claim_str}下午开会,提前叫我"
    # debug 直接取消
    # content = f"{t_claim_str}5分钟后的日程取消"
    # debug 提醒规则
    # content = f"现在的时间是{t}\n明天下午3点24开会,提前叫我"
    # debug 创建日程
    # content = f"明天下午3点40开会"
    content = "开会时间改到明天下午4点"
    history = [{"role":"user", "content": content}]
    while True:
        content = tsm._run(history, verbose=True, orgCode=orgCode, customId=customId)
        history.append({"role":"assistant", "content": content})
        history.append({"role":"user", "content": input(f"{content}: ")})
    # tsm._run(f"开会时间改到明天下午3点", verbose=True, schedule=schedule)
    
