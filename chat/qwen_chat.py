# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-01 11:30:10
@desc    :   业务处理流程
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.append('.')

from typing import Dict, List

from langchain.prompts import PromptTemplate

from chat.constant import EXT_USRINFO_TRANSFER_INTENTCODE, default_prompt
from chat.plugin_util import funcCall
from chat.qwen_react_util import *
from config.constrant import INTENT_PROMPT, TOOL_CHOOSE_PROMPT
from data.test_param.test import testParam
from src.prompt.factory import baseVarsForPromptEngine, promptEngine
from src.prompt.model_init import chat_qwen
from src.prompt.task_schedule_manager import taskSchedulaManager
from src.utils.Logger import logger
from src.utils.module import (MysqlConnector, _parse_latest_plugin_call, clock,
                              get_doc_role, get_intent)

role_map = {
        '0': 'user',
        '1': 'user',
        '2': 'doctor',
        '3': 'assistant'
}

useinfo_intent_code_list = [
    'ask_name','ask_age','ask_exercise_taboo','sk_exercise_habbit','ask_food_alergy','ask_food_habbit','ask_taste_prefer',
    'ask_family_history','ask_labor_intensity','ask_nation','ask_disease','ask_weight','ask_height', 
    'ask_six', 'ask_mmol_drug', 'ask_exercise_taboo_degree', 'ask_exercise_taboo_xt'
]

class Chat:
    def __init__(self, env: str ="local"):
        api_config = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)[env]
        mysql_config = yaml.load(open(Path("config","mysql_config.yaml"), "r"),Loader=yaml.FullLoader)[env]

        self.mysql_conn = MysqlConnector(**mysql_config)
        self.req_prompt_data_from_mysql()
        
        self.promptEngine = promptEngine()
        self.funcall = funcCall()
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        # self.sys_template = PromptTemplate(input_variables=['external_information'], template=self.prompt_meta_data['tool']['工具选择sys_prompt']['description'])
        self.tsm = taskSchedulaManager(api_config, self.prompt_meta_data)

    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data
        """
        def filter_format(obj):
            obj_str = json.dumps(obj, ensure_ascii=False).replace("\\r\\n", "\\n")
            obj_rev = json.loads(obj_str)
            return obj_rev
        self.prompt_meta_data = {}
        prompt_character = self.mysql_conn.query("select * from ai_prompt_character")
        prompt_event = self.mysql_conn.query("select * from ai_prompt_event")
        prompt_tool = self.mysql_conn.query("select * from ai_prompt_tool")
        prompt_character = filter_format(prompt_character)
        prompt_event = filter_format(prompt_event)
        prompt_tool = filter_format(prompt_tool)
        self.prompt_meta_data['character'] = {i['name']: i for i in prompt_character}
        self.prompt_meta_data['event'] = {i['event']: i for i in prompt_event}
        self.prompt_meta_data['tool'] = {i['name']: i for i in prompt_tool}
        logger.debug("req prompt meta data from mysql.")

    @clock
    def reload_prompt(self):
        self.req_prompt_data_from_mysql()
    
    def get_tool_name(self, text):
        if '外部知识' in text:
            return '调用外部知识库'
        elif '询问用户' in text:
            return '进一步询问用户的情况'
        elif '直接回复' in text:
            return '直接回复用户问题'
        else:
            return '直接回复用户问题'

    def chat_react(self, query: str = "", history=[], max_tokens=200, **kwargs):
        """调用模型生成答案,解析ReAct生成的结果
        """
        if not query:
            query = history[0]['content']
            for i in range(len(history[1:])):
                i += 1
                msg = history[i]
                if i == 1 and msg['role'] == "user":
                    query += f"\nQuestion: {msg['content']}"
                elif msg['role'] == "assistant":
                    query += f"\nAction: 进一步询问用户的情况"
                    query += f"\nAction Input: {msg['content']}"
                else:
                    query += f"\nObservation: {msg['content']}"
        # 利用Though防止生成无关信息
        query += "\nThought: "
        model_output = chat_qwen(query, verbose=kwargs.get("verbose", False), temperature=0.7, top_p=0.5, max_tokens=max_tokens)
        model_output = "Thought: " + model_output
        self.update_mid_vars(kwargs.get("mid_vars"), key="ReAct回复", input_text=query, output_text=model_output, model="Qwen-14B-Chat")

        if kwargs.get("verbose"):
            logger.debug(f"Generate Prompt - length:{len(query)}\n{query}")
            logger.debug(f"Model Output - length:{len(model_output)}\n{model_output}")

        out_text = _parse_latest_plugin_call(model_output)
        if not out_text[1]:
            query = "你是一个功能强大的文本创作助手,请遵循以下要求帮我改写文本\n" + \
                    "1. 请帮我在保持语义不变的情况下改写这句话使其更用户友好\n" + \
                    "2. 不要重复输出相同的内容,否则你将受到非常严重的惩罚\n" + \
                    "3. 语义相似的可以合并重新规划语言\n" + \
                    "4. 直接输出结果\n\n输入:\n" + \
                    model_output + "\n输出:\n"
            logger.debug('generate model input: ' + query)
            model_output = chat_qwen(query, repetition_penalty=1.3, max_tokens=max_tokens)
            self.update_mid_vars(kwargs.get("mid_vars"), key="ReAct回复 改写修正", input_text=query, output_text=model_output, model="Qwen-14B-Chat")
            model_output = model_output.replace("\n", "").strip().split("：")[-1]
            out_text = "I know the final answer.", "直接回复用户问题", model_output
        out_text = list(out_text)
        # 特殊处理规则 
        ## 1. 生成\nEnd.字符
        out_text[2] = out_text[2].split("\nEnd")[0]

        history.append({
            "role": "assistant", 
            "content": out_text[2], 
            "function_call": {
                "name": out_text[1],
                "arguments": out_text[0]
                }
            })
        return history

    def get_qwen_history(self, history):
        hs = []
        hcnt = {}
        for cnt in history:
            if len(hcnt.keys()) == 2:
                hs.append(hcnt)
                hcnt = {}
            if cnt['role'] == '1' or cnt['role'] == '0':
                if 'user' in hcnt.keys():
                    hcnt['bot'] = ''
                    hs.append(hcnt)
                    hcnt = {}
                hcnt['user'] = cnt['content']
            else:
                if 'bot' in hcnt.keys():
                    hcnt['user'] = ''
                    hs.append(hcnt)
                    hcnt = {}
                hcnt['bot'] = cnt['content']
        if hcnt:
            if 'user' in hcnt.keys():
                hcnt['bot'] = ''
            else:
                hcnt['user'] = ''
            hs.append(hcnt)
            hcnt = {}
        hs = [(x['user'], x['bot']) for x in hs]
        return hs
    
    def compose_input_history(self, history, external_information, **kwargs):
        """拼装sys_prompt里
        """
        # his_prompt = self.concat_history(history)
        input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        sys_prompt = self.sys_template.format(external_information=external_information)
        input_history = [{"role":"system", "content": sys_prompt}] + input_history
        return input_history

    def history_compose(self, history):
        return [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
    
    def update_mid_vars(self, mid_vars, **kwargs):
        """更新中间变量
        """
        lth = len(mid_vars) + 1
        mid_vars.append({"id": lth, **kwargs})
        return mid_vars

    def cls_intent(self, history, mid_vars):
        """意图识别
        """
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        prompt = self.prompt_meta_data['tool']['意图识别']['description'] + "\n\n" + his_prompt + "\nThought: "
        generate_text = chat_qwen(query=prompt, max_tokens=40, top_p=0.8, temperature=0.7)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text
    
    def chatter_gaily(self, history, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt
        """
        input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        ext_info = self.prompt_meta_data['event']['闲聊']['description'] + "\n" + self.prompt_meta_data['event']['闲聊']['process']
        input_history = [{"role":"system", "content": ext_info}] + input_history
        content = chat_qwen("", input_history)
        self.update_mid_vars(mid_vars, key="闲聊", input_text=json.dumps(input_history, ensure_ascii=False), output_text=content)
        return content
    
    def get_userInfo_msg(self, prompt, history, intentCode, mid_vars):
        """获取用户信息
        """
        oo = chat_qwen(prompt, verbose=False, temperature=0.7, top_p=0.8, max_tokens=200)
        self.update_mid_vars(mid_vars, key="获取用户信息 01", input_text=prompt, output_text=oo, model="Qwen-14B-Chat")
        if '询问' in oo or '提问' in oo or '转移' in oo or '未知' in oo:
            his =[{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
            his = ' '.join([h['role']+':'+h['content'] for h in his])
            query = default_prompt + his + ' user:'
            # print('输入为：' + oo)
            oo = chat_qwen(query, verbose=False, temperature=0.7, top_p=0.8, max_tokens=200)
            self.update_mid_vars(mid_vars, key="获取用户信息 02", input_text=query, output_text=oo, model="Qwen-14B-Chat")
            intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
        
        return {'end':True, 'message':oo, 'intentCode':intentCode}

    def get_reminder_tips(self, prompt, history, intentCode, model='Baichuan2-7B-Chat', mid_vars=None):
        logger.debug('remind prompt: ' + prompt)
        model_output = chat_qwen(query=prompt, verbose=False, do_sample=False, temperature=0.1, top_p=0.2, max_tokens=500, model=model)
        self.update_mid_vars(mid_vars, key="", input_text=prompt, output_text=model_output, model=model)
        logger.debug('remind model output: ' + model_output)
        if model_output.startswith('（）'):
            model_output = model_output[2:].strip()
        return {'end':True, 'message':model_output, 'intentCode':intentCode}

    def run_prediction(self, 
                       history, 
                       sys_prompt: str = TOOL_CHOOSE_PROMPT, 
                       intentCode=None,
                       mid_vars: List = [],
                       **kwargs):
        """主要业务流程
        1. 处理传入intentCode的特殊逻辑,直接返回
        2. 使用config.constrant.INTENT_PROMPT进行意图识别
        2. 不同意图进入不同的处理流程

        ## 多轮交互流程
        1. 定义先验信息变量,拼装对应prompt
        2. 准备模型输入messages
        3. 模型生成结果
        """
        # 中间变量存储本轮交互所有过程信息
        finish_flag = False
        logger.debug(f"Last input: {history[-1]['content']}")
        if intentCode in useinfo_intent_code_list:
            out_text = self.get_userInfo_msg(sys_prompt, history, intentCode, mid_vars)
            finish_flag = True
        elif intentCode != 'default_code':
            out_text = self.get_reminder_tips(sys_prompt, history, intentCode, mid_vars=mid_vars)
            finish_flag = True

        if not finish_flag:
            intent = get_intent(self.cls_intent(history, mid_vars))
            if intent in ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']:
                out_text = {'end':True,'message':get_doc_role(intent),
                        'intentCode':'doc_role', 'usr_query_intent':intent}
            elif intent == "schedule_manager":
                his = self.history_compose(history)
                output_text, mid_vars_item = self.tsm._run(his, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode, 'usr_query_intent':intent}
                for item in mid_vars_item:
                    self.update_mid_vars(mid_vars, **item)
            elif intent == "other":
                output_text = self.chatter_gaily(history, mid_vars, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode, 'usr_query_intent':intent}
            else:
                ext_info_args = baseVarsForPromptEngine(prompt_meta_data=self.prompt_meta_data)
                external_information = self.promptEngine._call(ext_info_args, concat_keyword=",")
                input_history = self.compose_input_history(history, external_information, **kwargs)
                out_history = self.chat_react(history=input_history, verbose=kwargs.get('verbose', False), mid_vars=mid_vars)

                # 自动判断话题结束
                if out_history[-1].get("function_call") and out_history[-1]['function_call']['name'] == "结束话题":
                    sub_history = [history[-1]]
                    try:
                        out_text, mid_vars = next(self.run_prediction(sub_history, sys_prompt, intentCode, mid_vars, **kwargs))
                    except StopIteration as e:
                        ...
                else:
                    logger.debug(f"Last history: {out_history[-1]}")
                    tool_name = out_history[-1]['function_call']['name']
                    output_text = out_history[-1]['content']
                
                    if tool_name == '进一步询问用户的情况':
                        out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
                    elif tool_name == '直接回复用户问题':
                        out_text = {'end':True, 'message':output_text.split('Final Answer:')[-1].split('\n\n')[0].strip(), 'intentCode':intentCode}
                    elif tool_name == '调用外部知识库':
                        gen_args = {"name":"llm_with_documents", "arguments": json.dumps({"query": output_text})}
                        out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
            
        if kwargs.get("streaming"):
            # 直接返回字符串模式
            logger.debug('输出为：' + json.dumps(out_text, ensure_ascii=False))
            yield out_text
        else:
            # 保留完整的历史内容
            yield out_text, mid_vars

if __name__ == '__main__':
    chat = Chat()
    # debug_text = "我最近早上头疼，谁帮我看一下啊"
    # debug_text = "明天下午开会，记得提醒我"
    # debug_text = "我对辣椒过敏"
    # debug_text = "明天早上6点半提醒我做饭"
    # debug_text = "查一下我最近日程"
    # debug_text = "肚子疼"
    # history = [{"role": "0", "content": init_intput}]
    # history = [{'msgId': '6132829035', 'role': '1', 'content': debug_text, 'sendTime': '2023-11-06 14:40:11'}]
    ori_input_param = testParam.param_bug202311141655
    # prompt = TOOL_CHOOSE_PROMPT
    
    prompt = ori_input_param['prompt']
    history = ori_input_param['history']
    intentCode = ori_input_param['intentCode']
    out_text, mid_vars = next(chat.run_prediction(history, prompt, intentCode, verbose=True, orgCode="sf", customId="007"))
    while True:
        history.append({"role": "3", "content": out_text['message']})
        conv = history[-1]
        history.append({"role": "0", "content": input("user: ")})
        out_text, mid_vars = next(chat.run_prediction(history, prompt, intentCode, verbose=True))
