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
from typing import Any, AnyStr, Dict, List

import requests
from click import prompt
from langchain.prompts import PromptTemplate

sys.path.append(str(Path.cwd()))
from config.constrant import task_schedule_return_demo
from config.constrant_for_task_schedule import (_tspdfq, query_schedule_template,
                                                task_schedule_parameter_description,
                                                task_schedule_parameter_description_for_qwen)
from src.prompt.model_init import ChatCompletionRequest, ChatMessage, callLLM
from src.prompt.qwen_openai_api import create_chat_completion
from src.utils.Logger import logger
from src.utils.module import (accept_stream_response, clock, curr_time, curr_weekday,
                              date_after_days, initAllResource, make_meta_ret, this_sunday)


class taskSchedulaManager:
    headers: Dict = {"content-type": "application/json"}
    payload_template: Dict = {
        "customId": None,"orgCode": None,"taskName": None,"taskType": None,"taskDesc": None,
        "intentCode": None, "repeatType": None, "cronDate": None, "fromTime": None, "toTime": None
        }
    def __init__(self, global_share_resource):
        """日程管理模块
        - Args
            
            api_config (Dict, required)
                api配置, 包含llm, langchain, graph_qa, ai_backend
            prompt_meta_data (Dict[str, Dict], required)
                prompt meta data,分立的prompt元数据
        """
        self.api_config = global_share_resource.api_config
        self.prompt_meta_data = global_share_resource.prompt_meta_data
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
        ret = callLLM(content)
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
        logger.info(f"Create schedule org: {orgCode} - uid: {customId} - {cur_time} {task}")
        return 200
    
    def tool_cancel_schedule(self, msg, schedule, **kwds):
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
        assert len(schedule) != 0, "no schedule can be canceled"
        cronDate = [i for i in schedule if i['task'] == task][0]['time']

        url = self.api_config['ai_backend'] + "/alg-api/schedule/manage"
        input_payload = {
            "customId": kwds.get("customId"),
            "orgCode": kwds.get("orgCode"),
            "taskName": task,
            "intentCode": "CANCEL",
            "cronDate": cronDate
        }
        payload = {**self.payload_template, **input_payload}
        response = self.session.post(url, json=payload, headers=self.headers).text
        resp_js = json.loads(response)
        assert resp_js["code"] == 200, resp_js["msg"]
        logger.info(f"Cancle schedule org:{orgCode} - uid:{customId} - {task}")
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
        logger.info(f"Change schedule org:{orgCode} - uid:{customId} - {task} from {task_time_ori} to {cur_time}.")
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
        raw_content = callLLM(history=history, top_p=0.8, temperature=0.7)
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
        data = json.loads(response)['data']
        ret = [{"task": i['taskName'], "time": i['cronDate']} for i in data]
        set_str = set([json.dumps(i, ensure_ascii=False) for i in ret])
        ret = [json.loads(i) for i in set_str]
        logger.debug(f"Realtime schedule: {ret}")
        return ret
    
    def generate_modify_content(self, msg, **kwds):
        tool_args = eval(msg.function_call['arguments'])
        task = tool_args['task']
        changed_time = tool_args['time'][11:]
        content = f"{task}提醒时间修改为{changed_time}"
        return content
    
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
        intentCode = kwds.get("intentCode")
        schedule = self.get_real_time_schedule(**kwds)
        request = ChatCompletionRequest(model="Qwen-14B-Chat", 
                                        # functions=task_schedule_parameter_description_for_qwen,
                                        functions=_tspdfq,
                                        messages=messages)
        if not intentCode == 'schedule_qry_up':
            msg, mid_vars_item = create_chat_completion(request, schedule)
            logger.debug("Thought:" + msg.content)
        else:
            mid_vars_item = [{"key":"日程管理", "input_text": intentCode, "output_text": "日程查询"}]
            msg = ChatMessage(role="assistant", content="", function_call={"name": "query_schedule", "arguments": ""})
        if msg.function_call:
            tool_name, tool_args = msg.function_call['name'], msg.function_call['arguments']
            yield make_meta_ret(msg=tool_name, code=intentCode, type="Tool"), mid_vars_item
            if tool_args: 
                yield make_meta_ret(msg=msg.content, code=intentCode, type="Thought"), mid_vars_item

        if msg.function_call:
            if tool_name == "ask_for_time":
                content = eval(tool_args)['ask'] if msg.function_call else msg.content
            elif tool_name == "create_schedule":
                self.tool_create_schedule(msg, **kwds)
                content = eval(tool_args)['ask']
            elif tool_name == "modify_schedule":
                self.tool_modify_schedule(msg, schedule, **kwds)
                content = self.generate_modify_content(msg, **kwds)
                # content = eval(tool_args)['ask']
            elif tool_name == "cancel_schedule":
                task = self.tool_cancel_schedule(msg, schedule, **kwds)
                content = f"已为您取消{task}的提醒"
            elif tool_name == "query_schedule":
                content, mid_vars_item = self.tool_query_schedule(schedule, mid_vars_item, **kwds)
            else:
                content = "我不清楚你想做什么,请稍后重试"
        meta_ret = make_meta_ret(end=True, 
                                 msg=content, 
                                 code=intentCode, 
                                 type="Result", 
                                 init_intent=self.prompt_meta_data['init_intent'].get(tool_name, False))
        yield meta_ret, mid_vars_item

class scheduleManager:
    """细化版日程管理模块 2024年1月2日17:56:10
    """
    def __init__(self, gsr: initAllResource) -> None:
        self.api_config = gsr.api_config
        self.model_config = gsr.model_config
        self.prompt_meta_data = gsr.prompt_meta_data
    
    def __init_vars__(self, funcmap, session) -> None:
        """初始化funcmap 用来获取ai_backend 日程管理方法
        """
        self.funcmap = funcmap
        self.session = session

    def __update_mid_vars__(self, mid_vars, input_text: Any = None, output_text: Any = None, key="节点名", model="调用模型", **kwargs):
        """更新中间变量
        """
        lth = len(mid_vars) + 1
        mid_vars.append({"id": lth, "key":key, "input_text": input_text, "output_text":output_text, "model":model, **kwargs})
        return mid_vars
    
    def call_query_confirm_query_time_range(self, query: str, current: str=curr_time() + " " + curr_weekday()) -> Dict:
        """确定查询的时间范围
        """
        output_format = '{"startTime": "%Y-%m-%d %H:%M:%S", "endTime": "%Y-%m-%d %H:%M:%S"}'
        prompt = (
            "请你理解用户所说, 解析其描述的时间范围,以下是一些指导:\n"
            "1. 如果未指明范围但说了日期,默认为当天的00:00:00到23:59:59\n"
            "2. 如果是今天,默认为今天从现在的时间开始到23:59:59\n"
            "3. 如果说本周,则从本周一00:00:00开始至周日23:59:59\n"
            "4. 早晨指05:00:00-08:00:00, 上午指08:00:00至11:00:00, 中午指11:00:00至13:00:00, 下午指13:00:00至18:00:00, 晚上指18:00:00至24:00:00"
            f"5. 输出的格式参考: {output_format}\n\n"
            f"现在时间: {current}\n"
            f"用户输入: {query}\n"
            "输出:"
        )
        # logger.debug(prompt)
        model = self.model_config.get("schedular_time_understand", "Qwen-14B-Chat")
        response = callLLM(prompt, model=model, stop="\n\n", stream=True)
        text = accept_stream_response(response, verbose=False)
        output = text.strip()
        time_range = json.loads(output)
        logger.debug(f"{output}")
        return time_range

    def query(self, *args, **kwds):
        """查询用户日程处理逻辑

        Note:
            1. 仅查询当日未完成日程
        """
        current = curr_time() + " " + curr_weekday()
        model = self.model_config.get('call_schedule_query', 'Qwen-14B-Chat')
        schedule = self.funcmap["get_schedule"]['func'](**kwds)
        query = kwds['history'][-2]['content']
        query_schedule_template = self.prompt_meta_data['event']['schedule_qry_up']['description']
        try:
            time_range = self.call_query_confirm_query_time_range(query, current)
        except Exception as err:
            time_range = {"startTime": curr_time(), "endTime": date_after_days(2)}
   
        target_schedule = [i for i in schedule if time_range['endTime'] > i['time'] > time_range['startTime']]
        target_schedule_content = "\n".join([f"{i['task']}: {i['time']}" for i in target_schedule])
        if not target_schedule_content:
            target_schedule_content = "空"
        prompt = query_schedule_template.replace("{{cur_time}}", current)
        prompt = prompt.replace("{{user_schedule}}", target_schedule_content)
        prompt = prompt.replace("{{query}}", query)

        messages = [
            {"role": "user", "content": prompt}
        ]
        logger.debug(prompt)
        response = callLLM(history=messages, top_p=0.8, temperature=0.85, model=model, stream=True)
        content = accept_stream_response(response, verbose=False)
        self.__update_mid_vars__(kwds['mid_vars'], key=f"LLM回答查询query", input_text=prompt, output_text=content, model=model)
        return content
    
    def get_currct_time_from_desc(self, time_desc: str):
        """根据时间描述获取正确的时间
        - Args
            time_desc [str]: 时间的描述 如:今晚8点
        - Return
            target_time [str]: %Y-%m-%d %H:%M:%S格式的时间
        """
        model = self.model_config.get('schedular_time_understand', 'Qwen-14B-Chat')
        current_time = curr_time() + " " +  curr_weekday()
        prompt = (
                "你是一个功能强大的时间理解及推理工具,可以根据描述和现在的时间推算出正确的时间(%Y-%m-%d %H:%M:%S格式)\n"
                f"现在的时间是: {current_time}\n"
                f"{time_desc}对应的时间是: "
            )
        response = callLLM(prompt, model=model, temperature=0.7, top_p=0.5, stream=True, stop="\n")
        target_time = accept_stream_response(response, verbose=False)[:19]
        return target_time
    
    def call_create_extract_event_time_pair(self, query: str, **kwds):
        head_str = '''[["'''
        model = self.model_config.get("call_schedule_create_extract_event_time_pair", "Qwen-14B-Chat")
        prompt = (
            "请你扮演一个功能强大的日程管理助手，帮用户提取描述中的日程名称和时间，提取的数据将用于为用户创建日程提醒，下面是一些要求:\n"
            "1. 日程名称尽量简洁明了并包含用户所描述的事件信息，如果未明确，则默认为`提醒`\n"
            "2. 事件可能是一个或多个, 每个事件对应一个时间, 请你充分理解用户的意图, 提取每个事件-时间\n"
            '3. 输出格式: [["事件1", "时间1"], ["事件2", "时间2"]]\n'
            "# 示例\n"
            "用户输入: 3分钟后叫我一下,今晚8点提醒我们看联欢晚会,半个小时后提醒我喝牛奶\n"
            "输出: \n"
            '[["提醒", "3分钟后"],["看联欢晚会", "今晚8点"], ["喝牛奶提醒", "半个小时后"]]\n'
            f"用户输入: {query}\n"
            "输出: \n"
            f"{head_str}"
        )
        response = callLLM(prompt, model=model, temperature=0.7, top_p=0.8,stop="\n\n",stream=True)
        event_time_pair = head_str + accept_stream_response(response, verbose=False)
        event_time_pair = eval(event_time_pair)
        self.__update_mid_vars__(kwds['mid_vars'], 
                                 input_text=query, 
                                 output_text=event_time_pair, 
                                 key="extract_event_time_pair",
                                 model=model)
        return event_time_pair

    def call_create_parse_currect_event_time(self, event_time_pair: List, **kwds):
        """从时间描述解析正确时间
        """
        except_result, unexpcept_result = [], []
        for item in event_time_pair:
            event, tdesc = item
            if not (event and tdesc):
                item.append(None)
                unexpcept_result.append(item)
                continue
            current_time = self.get_currct_time_from_desc(tdesc)
            item.append(current_time)
            except_result.append(item)
        except_result.sort(key=lambda x: x[2])      # 对提取出的事件 - 时间 按时间排序
        self.__update_mid_vars__(kwds['mid_vars'], 
                                 output_text={"except_result": except_result, "unexpcept_result": unexpcept_result}, 
                                 key="parse_time_desc",
                                 model=self.model_config.get('schedular_time_understand', 'Qwen-14B-Chat'))
        return except_result, unexpcept_result
    
    def call_create_execute_create_schedule(self, url, query, result_to_create: List, result_unexpected: List, **kwds):
        """执行创建日程接口
        """
        customId = kwds.get("customId")
        orgCode = kwds.get("orgCode")
        create_schedule_success = []
        for item in result_to_create:           # 逐一创建日程并
            task, desc, cronDate = item
            payload = {"customId": customId, "orgCode": orgCode, "taskName": task, "cronDate": cronDate, "taskType": "reminder","intentCode": "CREATE"}
            responseJS = self.session.post(url, json=payload).json()
            if responseJS["code"] == 200 and responseJS['data'] is True:
                create_schedule_success.append(item)
                logger.success(f"Create schedule org: {orgCode}, uid: {customId}, task: {task}, time: {cronDate}, desc: {desc}")
                continue
            else:
                responseJS = self.session.post(url, json=payload).json()
                if responseJS["code"] == 200 and responseJS['data'] is True:
                    create_schedule_success.append(item)
                    logger.info(f"Create schedule org: {orgCode}, uid: {customId}, task: {task}, time: {cronDate}, desc: {desc}")
                else:
                    result_unexpected.append(item)
                    logger.error(f"日程创建失败: \npayload: {payload}\nresponse: {responseJS}")
        # TODO 回复内容待优化
        prompt_template = PromptTemplate.from_template(self.prompt_meta_data['event']['call_schedule_create_reply']['description'])
        model = self.model_config['call_schedule_create_reply']
        created_schedule_content = [i[1]+": "+i[0] for i in create_schedule_success]
        prompt = prompt_template.format(created_schedule_content=created_schedule_content)
        message = [{"role":"user", "content": prompt}]
        response = callLLM(history=message, model=model, temperature=0.6, top_p=0.5, stream=True)
        content = accept_stream_response(response, verbose=False)
        self.__update_mid_vars__(kwds['mid_vars'], input_text=message, output_text=content, key="schedule_create_reply", model=model)
        return content

    def create(self, *args, **kwds):
        """提取日程信息并创建日程，处理流程见 doc/日程管理/日程创建流程.drawio
        """        
        query = kwds['history'][-2]['content']
        func = self.funcmap['create_schedule']
        url = self.api_config["ai_backend"] + func['method']
        
        event_time_pair = self.call_create_extract_event_time_pair(query, **kwds)
        result_to_create, result_unexpected = self.call_create_parse_currect_event_time(event_time_pair, **kwds)
        content = self.call_create_execute_create_schedule(url, query, result_to_create, result_unexpected, **kwds)
        return content

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
    
