# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-01 11:30:10
@desc    :   业务处理流程
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys

sys.path.append('.')

from typing import Any, Dict, List

from langchain.prompts import PromptTemplate
from requests import Session

from chat.constant import EXT_USRINFO_TRANSFER_INTENTCODE, default_prompt, intentCode_desc_map
from chat.util import norm_userInfo_msg
from config.constrant import DEFAULT_DATA_SOURCE
from config.constrant import TOOL_CHOOSE_PROMPT_PIPELINE as TOOL_CHOOSE_PROMPT
from config.constrant import role_map
from data.test_param.test import testParam
from src.pkgs.knowledge.callback import funcCall
from src.prompt.factory import customPromptEngine
from src.prompt.model_init import callLLM
from src.prompt.react_demo import build_input_text
from src.utils.Logger import logger
from src.utils.module import (get_doc_role, get_intent, initAllResource, make_meta_ret,
                              parse_latest_plugin_call)


class Chat_v2:
    def __init__(self, global_share_resource: initAllResource) -> None:
        global_share_resource.chat_v2 = self
        self.gsr = global_share_resource
        self.prompt_meta_data = self.gsr.prompt_meta_data
        self.promptEngine = customPromptEngine(self.gsr)
        self.funcall = funcCall(self.gsr)
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        self.__initalize_intent_map__()
        self.session = Session()
    
    def __initalize_intent_map__(self):
        """初始化各类意图map
        """
        schedule_manager = ["schedule_qry_up", "schedule_manager"]
        tips_intent_code_list = [
            'dietary_eva', 'schedule_no', 'measure_bp',
            'meet_remind', 'medicine_remind', 'dietary_remind', 'sport_remind',
            'broadcast_bp', 'care_for', 'default_clock',
            'default_reminder', 'broadcast_bp_up'
        ]
        useinfo_intent_code_list = [
            'ask_name','ask_age','ask_exercise_taboo','sk_exercise_habbit','ask_food_alergy','ask_food_habbit','ask_taste_prefer',
            'ask_family_history','ask_labor_intensity','ask_nation','ask_disease','ask_weight','ask_height', 
            'ask_six', 'ask_mmol_drug', 'ask_exercise_taboo_degree', 'ask_exercise_taboo_xt'
        ]
        aiui_intent_code_list = ['websearch', 'KLLI3.captialInfo', 'lottery', 'dream', 'AIUI.calc', 'LEIQIAO.cityOfPro', 'ZUOMX.queryCapital', 'calendar', 'audioProgram', 'translation', 'garbageClassifyPro', 'AIUI.unitConversion', 'AIUI.forexPro', 'carNumber', 'datetimePro', 'AIUI.ocularGym', 'weather', 'cookbook', 'story', 'AIUI.Bible', 'drama', 'storyTelling', 'AIUI.audioBook', 'musicX', 'news', 'joke']
        callout_intent_code_list = ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']
        self.intent_map = {
            'schedule': {i:1 for i in schedule_manager},
            'tips': {i:1 for i in tips_intent_code_list},
            'userinfo': {i:1 for i in useinfo_intent_code_list},
            'aiui': {i:1 for i in aiui_intent_code_list},
            'callout': {i:1 for i in callout_intent_code_list}
        }

    def __get_default_reply__(self, intentCode):
        """针对不同的意图提供不同的回复指导话术
        """
        if intentCode == "schedule_manager" or intentCode == "other_schedule":
            content = "对不起，我没有理解您的需求，如果想进行日程提醒管理，您可以这样说: 查询一下我今天的日程, 提醒我明天下午3点去打羽毛球, 帮我把明天下午3点打羽毛球的日程改到后天下午5点, 取消今天的提醒"
        elif intentCode == "schedule_qry_up":
            content = "对不起，我没有理解您的需求，如果您想查询今天的待办日程，您可以这样说：查询一下我今天的日程"
        elif intentCode == "meeting_schedule":
            content = "对不起，我没有理解您的需求，如果您想管理会议日程，您可以这样说：帮我把明天下午4点的会议改到今天晚上7点"
        elif intentCode == "auxiliary_diagnosis":
            content = "对不起，我没有理解您的需求，如果您有健康问题想要咨询，建议您提供更明确的描述"
        else:
            content = "对不起, 我没有理解您的需求, 请在问题中提供明确的信息并重新尝试."
        return content
    
    def __check_query_valid__(self, query):
        prompt = (
            "你是一个功能强大的内容校验工具请你帮我判断下面输入的句子是否符合要求\n"
            "1. 是一句完整的可以向用户输出的话\n"
            "2. 不包含特殊符号\n"
            "3. 语义完整连贯\n"
            "要判断的句子: {query}\n\n"
            "你的结果(yes or no):\n"
        )
        prompt = prompt.replace("{query}", query)
        result = callLLM(query=prompt, temperature=0, top_p=0, max_tokens=3)
        if "yes" in result.lower():
            return True
        else:
            return False
    
    def __generate_content_verification__(self, out_text, list_of_plugin_info, **kwargs):
        """ReAct生成内容的校验

        1. 校验Tool
        2. 校验Tool Parameter格式
        """
        thought, tool, parameter = out_text
        possible_tool_map = {i['code']: 1 for i in list_of_plugin_info}

        try:
            parameter = json.loads(parameter)
        except Exception as err:
            ...

        # 校验Tool
        if not possible_tool_map.get(tool):     # 如果生成的Tool不对, parameter也必然不对
            tool = "AskHuman"
            parameter = self.__get_default_reply__(kwargs['intentCode'])
        
        if tool == "AskHuman":
            # TODO 如果生成的工具是AskHuman但参数是dict, 1. 尝试提取dict中的内容  2. 回复默认提示话术
            if isinstance(parameter, dict):
                for gkey, gcontent in parameter.items():
                    if self.__check_query_valid__(gcontent):
                        parameter = gcontent
                        break
                if isinstance(parameter, dict):
                    parameter = self.__get_default_reply__(kwargs['intentCode'])
        return [thought, tool, parameter]

    def chat_react(self, *args, **kwargs):
        """调用模型生成答案,解析ReAct生成的结果
        """
        max_tokens = kwargs.get("max_tokens", 200)
        _sys_prompt, list_of_plugin_info = self.compose_input_history(**kwargs)
        prompt = build_input_text(_sys_prompt, list_of_plugin_info, **kwargs)
        prompt += "Thought: "
        logger.debug(f"ReAct Prompt:\n{prompt}")
        model_output = callLLM(prompt, temperature=0.7, top_p=0.5, max_tokens=max_tokens, model="Qwen-14B-Chat", stop=["\nObservation"])
        model_output = "\nThought: " + model_output
        logger.debug(f"ReAct Generate: {model_output}")
        self.update_mid_vars(kwargs.get("mid_vars"), key="Chat ReAct", input_text=prompt, output_text=model_output, model="Qwen-14B-Chat")

        # model_output = """Thought: 任务名和时间都没有提供，无法创建日程。\nAction: AskHuman\nAction Input: {"message": "请提供任务名和时间。"}"""
        out_text = parse_latest_plugin_call(model_output)
        # if not self.prompt_meta_data['prompt_tool_code_map'].get(out_text[1]):
        #     out_text[1] = "AskHuman"
        
        # (optim) 对于react模式, 如果一个事件提供工具列表, 生成的Action不属于工具列表中, 不同的意图返回不同的话术指导和AskHuman工具 2024年1月9日15:50:18
        out_text = self.__generate_content_verification__(out_text, list_of_plugin_info, **kwargs)
        try:
            # gen_args = json.loads(out_text[2])
            tool = out_text[1]
            tool_zh = self.prompt_meta_data['prompt_tool_code_map'].get(tool)
            tool_param_msg = self.prompt_meta_data['tool'][tool_zh].get("params")
            # if self.prompt_meta_data['rollout_tool'].get(tool) and tool_param_msg and len(tool_param_msg) ==1:
            if tool_param_msg and len(tool_param_msg) == 1:
                # 对于直接输出的,此处判断改工具设定的参数,通常只有一项 为要输出的话,此时解析对应字段
                if tool_param_msg[0]['schema']['type'].startswith("str"):
                    out_text[2] = out_text[2][tool_param_msg[0]['name']]
        except Exception as err:
            # logger.exception(err)
            ...
        kwargs['history'].append({
            "intentCode": kwargs['intentCode'],
            "role": "assistant", 
            "content": out_text[0], 
            "function_call": {"name": out_text[1],"arguments": out_text[2]}
            })
        return kwargs['history']
    
    def compose_input_history(self, **kwargs):
        """拼装sys_prompt里
        """
        qprompt = kwargs.get("qprompt")

        sys_prompt, functions = self.promptEngine._call(**kwargs)

        if not qprompt:
            sys_prompt = self.sys_template.format(external_information=sys_prompt)
        else:
            sys_prompt = sys_prompt + "\n\n" + qprompt
        return sys_prompt, functions
    
    def update_mid_vars(self, mid_vars, input_text=Any, output_text=Any, key="节点名", model="调用模型", **kwargs):
        """更新中间变量
        """
        lth = len(mid_vars) + 1
        mid_vars.append({"id": lth, "key":key, "input_text": input_text, "output_text":output_text, "model":model, **kwargs})
        return mid_vars

    def get_parent_intent_name(self, text):
        if '五师' in text:
            return '呼叫五师意图'
        elif '音频' in text:
            return '音频播放意图'
        elif '生活' in text:
            return '生活工具查询意图'
        elif '医疗' in text:
            return '医疗健康意图'
        elif '饮食' in text:
            return '饮食营养意图'
        elif '运动' in text:
            return '运动咨询意图'
        elif '日程':
            return '日程管理意图'
        elif '食材采购' in text:
            return '食材采购意图'
        else:
            return '其它'
    
    def cls_intent(self, history, mid_vars, **kwargs):
        """意图识别
        """
        open_sch_list = ['打开','日程']
        market_list = ['打开','集市']
        home_list = ['打开','家居']
        bp_list = ['血压趋势图','血压录入','血压添加','入录血压','添加血压','历史血压','血压历史']
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        if sum([1 for i in bp_list if i in history[-1]['content']]) > 0:     
            return '打开功能页面'
        if sum([1 for i in open_sch_list if i in history[-1]['content']]) >= 2:
            return '打开功能页面'
        if sum([1 for i in market_list if i in history[-1]['content']]) >= 2:
            return '打开功能页面'
        if sum([1 for i in home_list if i in history[-1]['content']]) >= 2:
            return '打开功能页面'
        h_p = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history[-3:]])
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        if kwargs.get('intentPrompt', ''):
            prompt = kwargs.get('intentPrompt') + "\n\n" + h_p + "\nThought: "
        else:
            prompt = self.prompt_meta_data['tool']['父意图']['description'] + "\n\n" + h_p + "\nThought: "
        logger.debug('父意图模型输入：' + prompt)
        generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                temperature=0, do_sample=False)
        logger.debug('意图识别模型输出：' + generate_text)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        parant_intent = self.get_parent_intent_name(text)
        if parant_intent in ['呼叫五师意图', '音频播放意图', '生活工具查询意图', '医疗健康意图', '饮食营养意图', '日程管理意图', '食材采购意图']:
            sub_intent_prompt = self.prompt_meta_data['tool'][parant_intent]['description']
            if parant_intent in ['呼叫五师']:
                history = history[-1:]
                h_p = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
            if kwargs.get('subIntentPrompt', ''):
                prompt = kwargs.get('subIntentPrompt').format(sub_intent_prompt) + "\n\n" + h_p + "\nThought: "
            else:
                prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt) + "\n\n" + h_p + "\nThought: "
            logger.debug('子意图模型输入：' + prompt)
            generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                    temperature=0, do_sample=False)
            intentIdx = generate_text.find("\nIntent: ") + 9
            text = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text

    def chatter_gaily(self, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt
        """
        def compose_func_reply(messages):
            """拼接func中回复的内容到history中
            
            最终的history只有role/content字段
            """
            history = []
            for i in messages:
                if not i.get("function_call"):
                    history.append(i)
                else:
                    func_args = i['function_call']
                    role = i['role']
                    content = f"{func_args['arguments']}"
                    history.append({"role": role, "content": content})
            return history
        
        intentCode = kwargs.get("intentCode", 'other')
        messages = [i for i in kwargs['history'] if i.get("intentCode") == intentCode]
        
        desc = self.prompt_meta_data['event'][intentCode].get('description', '')
        process = self.prompt_meta_data['event'][intentCode].get('process', '')
        if desc or process:     # (optim) 无描述, 不添加system 2024年1月8日14:07:36, 针对需要走纯粹闲聊的问题
            ext_info = desc + "\n" + process
            messages = [{"role":"system", "content": ext_info}] + messages

        messages = compose_func_reply(messages)
        content = callLLM("", messages, temperature=0.7, top_p=0.8)
        self.update_mid_vars(mid_vars, key="闲聊", input_text=json.dumps(messages, ensure_ascii=False), output_text=content)
        if kwargs.get("return_his"):
            messages.append({
                "role": "assistant", 
                "content": "I know the answer.", 
                "function_call": {"name": "convComplete", "arguments": content}
            })
            return messages
        else:
            return content
    
    def chatter_gaily_knowledge(self, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt
        """
        def compose_func_reply(messages):
            """拼接func中回复的内容到history中
            
            最终的history只有role/content字段
            """
            payload = {"query": "","knowledge_base_name": "","top_k": 5,"score_threshold": 1,"history": [],
                "stream": False,"model_name": "Qwen-14B-Chat","temperature": 0.7,
                "top_p": 0.8,"max_tokens": 0,"prompt_name": "text"}
            history = []
            for i in messages:
                if not i.get("function_call"):
                    history.append(i)
                else:
                    func_args = i['function_call']
                    role = i['role']
                    content = f"{func_args['arguments']}"
                    history.append({"role": role, "content": content})
            payload['history'] = history[:-1]   
            payload['query'] = history[-1]['content']
            payload['knowledge_base_name'] = "新奥百科知识库"
            return payload

        url = self.gsr.api_config['langchain'] + '/chat/knowledge_base_chat'
        intentCode = kwargs.get("intentCode", 'other')

        messages = [i for i in kwargs['history'] if i.get("intentCode") == intentCode]
        
        desc = self.prompt_meta_data['event'][intentCode].get('description', '')
        process = self.prompt_meta_data['event'][intentCode].get('process', '')
        if desc or process:     # (optim) 无描述, 不添加system 2024年1月8日14:07:36, 针对需要走纯粹闲聊的问题
            ext_info = desc + "\n" + process
            messages = [{"role":"system", "content": ext_info}] + messages

        # event_msg = self.prompt_meta_data['event'][intentCode]
        # system_prompt = event_msg['description'] + "\n" + event_msg['process']

        # messages = [{"role":"system", "content": system_prompt}] + messages
        payload = compose_func_reply(messages)

        response = self.session.post(url, data=json.dumps(payload)).json()
        content, docs = response['answer'], response['docs']
        self.update_mid_vars(mid_vars, key="闲聊-知识库-新奥百科", input_text=payload, output_text=response, model="知识库-新奥百科知识库-Qwen-14B-Chat")
        if kwargs.get("return_his"):
            messages.append({
                "role": "assistant", 
                "content": "I know the answer.", 
                "function_call": {"name": "convComplete", "arguments": content}
            })
            return messages[1:]
        else:
            return content

    def intent_query(self, history, **kwargs):
        mid_vars = kwargs.get('mid_vars', [])
        task = kwargs.get('task', '')
        input_prompt = kwargs.get('prompt', [])
        if task == 'verify' and input_prompt:
            intent, desc = get_intent(self.cls_intent_verify(history, mid_vars,
                input_prompt))
        else:
            intent, desc = get_intent(self.cls_intent(history, mid_vars, **kwargs))
        if self.intent_map['callout'].get(intent):
            out_text = {'message':get_doc_role(intent),
                        'intentCode':'doc_role', 'processCode':'trans_back',
                        'intentDesc':desc}
        elif self.intent_map['aiui'].get(intent):
            out_text = {'message':'', 'intentCode':intent, 'processCode':'aiui', 'intentDesc':desc}
        elif intent in ['food_rec']:
            if not kwargs.get('userInfo', {}).get('askTastePrefer', ''):
                out_text = {'message':'', 'intentCode':intent,
                     'processCode':'trans_back', 'intentDesc':desc}
            else:
                out_text = {'message':'', 'intentCode':'food_rec',
                        'processCode':'alg', 'intentDesc':desc}
        #elif intent in ['sport_rec']:
        #    if kwargs.get('userInfo', {}).get('askExerciseHabbit', '') and kwargs.get('userInfo',{}).get('askExerciseTabooDegree', '') and kwargs.get('userInfo', {}).get('askExerciseTabooXt', ''):
        #        out_text = {'message':'',
        #                'intentCode':intent,'processCode':'alg', 'intentDesc':desc}
        #    else:
        #        out_text = {'message':'', 'intentCode':intent,
        #                'processCode':'trans_back', 'intentDesc':desc}
        else:
            out_text = {'message':'', 'intentCode':intent, 'processCode':'alg', 'intentDesc':desc}
        logger.debug('意图识别输出：' + json.dumps(out_text, ensure_ascii=False))
        return out_text

    def fetch_intent_code(self):
        """返回所有的intentCode"""
        intent_code_map = {
            "get_userInfo_msg": list(self.intent_map['userinfo'].keys()),
            "get_reminder_tips": list(self.intent_map['tips'].keys()),
            "other": ['BMI', 'food_rec', 'sport_rec', 'schedule_manager', 'schedule_qry_up', 'auxiliary_diagnosis', "other"]
        }
        return intent_code_map
    
    def pre_fill_param(self, *args, **kwargs):
        """结合业务逻辑，预构建输入
        """
        intentCode = kwargs.get("intentCode")
        if not self.prompt_meta_data['event'].get(intentCode):
            logger.debug(f"not support current event {intentCode}, change intentCode to other.")
            kwargs['intentCode'] = 'other'
        if intentCode == "schedule_qry_up" and not kwargs.get("history"):
            kwargs['history'] = [{"role": 0, "content": "帮我查询今天的日程"}]
        if "schedule" in intentCode:
            kwargs['schedule'] = self.funcall.call_get_schedule(*args, **kwargs)
        return args, kwargs

    def general_yield_result(self, *args, **kwargs):
        """处理最终的输出
        """
        args, kwargs = self.pre_fill_param(*args, **kwargs)
        if kwargs.get("history"):
            history = [{**i, "role": role_map.get(str(i['role']), "user")} for i in kwargs['history']]
            kwargs['history'] = kwargs['backend_history'] + [history[-1]]
            kwargs['history'][-1]['intentCode'] = kwargs['intentCode']

        if kwargs['intentCode'] == 'other':
            kwargs['prompt'] = None
            kwargs['sys_prompt'] = None
        
        _iterable = self.pipeline(*args, **kwargs)
        while True:
            try:
                yield_item = next(_iterable)
                if not yield_item['data'].get("type"):
                    yield_item['data']['type'] = "Result"
                if yield_item['data']['type'] == "Result" and not yield_item['data'].get("dataSource"):
                    yield_item['data']['dataSource'] = DEFAULT_DATA_SOURCE
                yield yield_item
            except StopIteration as err:
                break

    def __log_init(self, **kwargs):
        """初始打印日志
        """
        intentCode = kwargs.get('intentCode')
        history = kwargs.get('history')
        logger.info(f'intentCode: {intentCode}')
        if history:
            logger.info(f"Input: {history[-1]['content']}")
    
    def parse_last_history(self, history):
        tool = history[-1]['function_call']['name']
        content = history[-1]['function_call']['arguments']
        thought = history[-1]['content']
        logger.debug(f"Action: {tool}")
        logger.debug(f"Thought: {thought}")
        logger.debug(f"Action Input: {content}")
        return tool, content, thought

    def get_userInfo_msg(self, prompt, history, intentCode, mid_vars):
        """获取用户信息
        """
        logger.debug(f'信息提取prompt:\n{prompt}')
        model_output = callLLM(prompt,
                                 verbose=False,
                                 temperature=0,
                                 top_p=0.8,
                                 max_tokens=200,
                                 do_sample=False)
        logger.debug('信息提取模型输出：' + model_output)
        content = model_output
        self.update_mid_vars(mid_vars, key="获取用户信息 01", input_text=prompt, output_text=content, model="Qwen-14B-Chat")
        if sum([i in content for i in ["询问","提问","转移","结束", "未知","停止"]]) != 0:
            logger.debug('信息提取流程结束...')
            content = self.chatter_gaily(mid_vars, history=history)
            intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
        elif content:
            content = content.split('\n')[0].split('。')[0][:20]
            logger.debug('标签归一前提取内容：' + content)
            content = norm_userInfo_msg(intentCode, content)
            logger.debug('标签归一后提取内容：' + content)
        content = content if content else '未知'
        content = '未知' if 'Error' in content else content
        return content, intentCode

    def get_reminder_tips(self, prompt, history, intentCode, model='Baichuan2-7B-Chat', mid_vars=None):
        logger.debug('remind prompt: ' + prompt)
        content = callLLM(query=prompt, verbose=False, do_sample=False, temperature=0.1, top_p=0.2, max_tokens=500, model=model)
        self.update_mid_vars(mid_vars, key="", input_text=prompt, output_text=content, model=model)
        logger.debug('remind model output: ' + content)
        if content.startswith('（）'):
            content = content[2:].strip()
        return content, intentCode

    def open_page(self, mid_vars, **kwargs):
        """组装mysql中打开页面对应的prompt
        """
        add_bp_list = ['血压录入','血压添加','录入血压','添加血压']
        add_diet_list = ['打开记录','打开录入','打开添加']
        diet_record_list = ['饮食记录','饮食添加','打开推荐','饮食评估','食谱','我的饮食','食谱页面','餐食记录']
        market_list = ['集市']
        personal_list = ['我的设置']
        qr_code_list = ['二维码']
        input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in kwargs['history']]
        input_history = input_history[-3:]
        if '血压趋势' in input_history[-1]['content']:
            return 'pagename:"bloodPressure-trend-chart"'
        elif sum([1 for i in add_bp_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"add-blood-pressure"'
        elif sum([1 for i in personal_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"personal-setting"'
        elif sum([1 for i in qr_code_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"qr-code"'
        elif '血压历史' in input_history[-1]['content'] or '历史血压' in input_history[-1]['content']:
            return 'pagename:"record-list3"'
        elif sum([1 for i in add_diet_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"add-diet"'
        elif sum([1 for i in diet_record_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"diet-record"'
        elif sum([1 for i in market_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"my-market"'
        elif '打开' in input_history[-1]['content'] and '日程' in input_history[-1]['content']:
            return 'pagename:"my-schedule"'

        hp = [h['role'] + ' ' + h['content'] for h in input_history]
        hi = ''
        if len(input_history) > 1:
            hi = '用户历史会话如下，可以作为意图识别的参考，但不要过于依赖历史记录，因为它可能是狠久以前的记录：' + '\n' + '\n'.join([h["role"] + h["content"] for h in input_history[-3:-1]]) + '\n' + '当前用户输入：\n'
        hi += f'Question:{input_history[-1]["content"]}\nThought:'
        ext_info = self.prompt_meta_data['event']['open_Function']['description'] + "\n" + self.prompt_meta_data['event']['open_Function']['process'] + '\n' + hi + '\nThought:'
        input_history = [{"role":"system", "content": ext_info}]
        logger.debug('打开页面模型输入：' + json.dumps(input_history,ensure_ascii=False))
        content = callLLM("", input_history, temperature=0, top_p=0.8, do_sample=False)
        if content.find('Answer') != -1:
            content = content[content.find('Answer')+7:].split('\n')[0].strip()
        if content.find('Output') != -1:
            content = content[content.find('Response')+6:].split('\n')[0].strip()
        if content.find('Response') != -1:
            content = content[content.find('')+9:].split('\n')[0].strip()

        self.update_mid_vars(mid_vars, key="打开功能画面", input_text=json.dumps(input_history, ensure_ascii=False), output_text=content)
        return content 

    def get_pageName_code(self, text):
        logger.debug('页面生成内容：' + text)
        if 'bloodPressure-trend-chart' in text and 'pagename' in text:
            return 'bloodPressure-trend-chart'
        elif 'add-blood-pressure' in text and 'pagename' in text:
            return 'add-blood-pressure'
        elif 'record-list3' in text and 'pagename' in text:
            return 'record-list3'
        elif 'my-schedule' in text and 'pagename' in text:
            return 'my-schedule'
        elif 'add-diet' in text and 'pagename' in text:
            return 'add-diet'
        elif 'diet-record' in text and 'pagename' in text:
            return 'diet-record'
        elif 'my-market' in text and 'pagename' in text:
            return 'my-market'
        elif 'personal-setting' in text and 'pagename' in text:
            return 'personal-setting'
        elif 'qr-code' in text and 'pagename' in text:
            return 'qr-code'
        elif 'smart-home' in text and 'pagename' in text:
            return 'smart-home'
        else:
            return 'other'

    def complete(self, mid_vars: List[object], **kwargs):
        """only prompt模式的生成及相关逻辑
        """
        assert kwargs.get("prompt"), "Current process type is only_prompt, but not prompt passd."
        prompt = kwargs['prompt']
        chat_history = kwargs['history']
        intentCode = kwargs['intentCode']
        if self.intent_map['userinfo'].get(intentCode):
            content, intentCode = self.get_userInfo_msg(prompt, chat_history, intentCode, mid_vars)
        elif self.intent_map['tips'].get(intentCode): 
            content, intentCode = self.get_reminder_tips(prompt, chat_history, intentCode, mid_vars=mid_vars)
        elif intentCode == "open_Function":
            output_text = self.open_page(mid_vars, **kwargs)
            content = '稍等片刻，页面即将打开' if self.get_pageName_code(output_text) != 'other' else output_text
            intentCode = self.get_pageName_code(output_text)
            logger.debug('页面Code: ' + intentCode)
        else:
            content = self.chatter_gaily(mid_vars, return_his=False, **kwargs)
        
        assert type(content) == str, "only_prompt模式下，返回值必须为str类型"
        
        chat_history.append({
            "role": "assistant", 
            "content": "I know the answer.",
            "function_call": {"name": "convComplete", "arguments": content} 
        })
        return chat_history, intentCode
    
    def interact_first(self, mid_vars, **kwargs):
        """首次交互
        """
        intentCode = kwargs.get('intentCode')
        out_history = None
        if self.prompt_meta_data['event'].get(intentCode):
            if intentCode == "other" :
                # 2023年12月26日10:07:03 闲聊接入知识库 https://devops.aliyun.com/projex/task/VOSE-3715# 《模型中调用新奥百科的知识内容》
                out_history = self.chatter_gaily(mid_vars, **kwargs, return_his=True)                
            elif intentCode == "enn_wiki":
                out_history = self.chatter_gaily_knowledge(mid_vars, **kwargs, return_his=True)
            elif self.prompt_meta_data['event'][intentCode].get("process_type") == "only_prompt":
                out_history, intentCode = self.complete(mid_vars=mid_vars, **kwargs)
                kwargs['intentCode'] = intentCode
            elif self.prompt_meta_data['event'][intentCode].get("process_type") == "react":
                out_history = self.chat_react(mid_vars=mid_vars, **kwargs)
        if not out_history: 
            out_history = self.chat_react(mid_vars=mid_vars, return_his=True, max_tokens=100, **kwargs)
        return out_history, intentCode
    
    def if_init(self, tool):
        # XXX 不是所有的流程都会调用工具，比如未定义意图的闲聊
        return self.prompt_meta_data['init_intent'].get(tool, False)
    
    def pipeline(self, mid_vars=[], **kwargs):
        """
        ## 多轮交互流程
        1. 定义先验信息变量,拼装对应prompt
        2. 准备模型输入messages
        3. 模型生成结果

        - Args
            
            history (List[Dict[str, str]]) required 
                对话历史信息
            mid_vars (List[Dict])
                中间变量
            intentCode (str)
                意图编码,直接根据传入的intentCode进入对应的处理子流程

        - Return
            out_text (Dict[str, str])
                返回的输出结果
        """
        self.__log_init(**kwargs)
        intentCode = kwargs.get('intentCode')
        mid_vars = kwargs.get('mid_vars', [])
        dataSource = DEFAULT_DATA_SOURCE
        out_history, intentCode = self.interact_first(mid_vars=mid_vars, **kwargs)
        while True:
            tool, content, thought = self.parse_last_history(out_history)

            if self.prompt_meta_data['event'].get(intentCode) and self.prompt_meta_data['event'][intentCode]['process_type'] != "only_prompt": # 2023年12月13日15:35:50 only_prompt对应的事件不输出思考
                ret_tool = make_meta_ret(msg=tool, type="Tool", code=intentCode, gsr=self.gsr)
                ret_thought = make_meta_ret(msg=thought, type="Thought", code=intentCode, gsr=self.gsr)
                yield {"data": ret_tool, "mid_vars": mid_vars, "history": out_history}
                yield {"data": ret_thought, "mid_vars": mid_vars, "history": out_history}

            if self.prompt_meta_data['rollout_tool'].get(tool) or not self.funcall.funcmap.get(tool):  
                # 2023年12月17日17:19:06 增加判断是否支持对应函数 未定义支持的 即使未写rollout_tool也直接返回,不走函数调用
                break
            try:
                kwargs['history'], dataSource = self.funcall._call(out_history=out_history, mid_vars=mid_vars, **kwargs)
            except AssertionError as err:
                logger.error(err)
                kwargs['history'], dataSource = self.funcall._call(out_history=out_history, mid_vars=mid_vars, **kwargs)
            
            if self.prompt_meta_data['rollout_tool_after_complete'].get(tool):
                # 工具执行完成后输出
                content = kwargs['history'][-1]['content']
                break
            else:
                # function_call的结果, self_rag
                content = kwargs['history'][-1]['content']
                ret_function_call = make_meta_ret(msg=content, type="Observation", code=intentCode,gsr=self.gsr)
                yield {"data": ret_function_call, "mid_vars": mid_vars, "history": out_history}
                out_history = self.chat_react(mid_vars=mid_vars, **kwargs)

        ret_result = make_meta_ret(end=True, msg=content,code=intentCode, gsr=self.gsr,
                                   init_intent=self.if_init(tool), dataSource=dataSource)
        yield {"data": ret_result, "mid_vars": mid_vars, "history": out_history}

if __name__ == '__main__':
    chat = Chat_v2(initAllResource())
    ori_input_param = testParam.param_feat_schedular_not_today
    prompt = ori_input_param['prompt']
    history = ori_input_param['history']
    intentCode = ori_input_param['intentCode']
    customId = ori_input_param['customId']
    orgCode = ori_input_param['orgCode']
    while True:
        out_text, mid_vars = next(chat.general_yield_result(**ori_input_param))
