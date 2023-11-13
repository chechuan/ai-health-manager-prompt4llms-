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

from typing import Dict

from langchain.prompts import PromptTemplate

from chat.plugin_util import funcCall
from chat.qwen_react_util import *
from chat.special_intent_msg_util import get_reminder_tips, get_userInfo_msg
from config.constrant import INTENT_PROMPT, TOOL_CHOOSE_PROMPT
# from config.function_call_config import function_tools
from data.test_param.test import testParam
from src.prompt.factory import baseVarsForPromptEngine, promptEngine
from src.prompt.model_init import chat_qwen
from src.prompt.task_schedule_manager import taskSchedulaManager
from src.utils.Logger import logger
from src.utils.module import (MysqlConnector, _parse_latest_plugin_call,
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

class Chat(object):
    def __init__(self, env: str ="local"):
        api_config = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)[env]
        mysql_config = yaml.load(open(Path("config","mysql_config.yaml"), "r"),Loader=yaml.FullLoader)[env]
        
        self.promptEngine = promptEngine()
        self.funcall = funcCall()
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        self.tsm = taskSchedulaManager(api_config)
        # self.mysql_conn = MysqlConnector(**mysql_config)
        # self.req_prompt_data_from_mysql()

    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data
        """
        self.pMData = {}
        self.pMData['character'] = self.mysql_conn.query("select * from ai_prompt_character")
        self.pMData['event'] = self.mysql_conn.query("select * from ai_prompt_event")
        self.pMData['tool'] = self.mysql_conn.query("select * from ai_prompt_tool")
        logger.debug("req prompt meta data from mysql.")
    
    def get_tool_name(self, text):
        if '外部知识' in text:
            return '调用外部知识库'
        elif '询问用户' in text:
            return '进一步询问用户的情况'
        elif '直接回复' in text:
            return '直接回复用户问题'
        else:
            return '直接回复用户问题'

    def generate(self, query: str = "", history=[], max_tokens=200, **kwargs):
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

    def cls_intent(self, history):
        """意图识别"""
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        generate_text = chat_qwen(query=prompt, max_tokens=40, top_p=0.8, temperature=0.7)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        return text
    
    def chatter_gaily(self, history, **kwargs):
        """闲聊
        """
        # ext_info_args = baseVarsForPromptEngine()
        # ext_info_args.plan = "日常对话"
        # ext_info_args.role = "智能健康管家"
        # external_information = self.promptEngine._call(ext_info_args, concat_keyword=",")
        input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # sys_prompt = self.sys_template.format(external_information=external_information)
        ext_info = (
            "你是来康生命公司研发的智能健康管家, 为居家用户提供健康咨询和管理服务\n"
            "对于日常闲聊，有以下几点建议:\n"
            "1. 整体过程应该是轻松愉快的\n"
            "2. 你可以适当发挥一点幽默基因\n"
            "3. 对用户是友好的\n"
            "4. 当问你是谁或叫什么名字时,你应当说我是智能健康管家"
            "5. 当问你是什么公司或者组织机构研发的时,你应说我是由来康生命研发的")
        input_history = [{"role":"system", "content": ext_info}] + input_history
        content = chat_qwen("", input_history)
        return content

    def run_prediction(self, 
                       history, 
                       sys_prompt: str = TOOL_CHOOSE_PROMPT, 
                       intentCode=None,
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
        finish_flag = False
        if intentCode in useinfo_intent_code_list:
            yield get_userInfo_msg(sys_prompt, history, intentCode)
            finish_flag = True
        elif intentCode != 'default_code':
            yield get_reminder_tips(sys_prompt, history, intentCode) 
            finish_flag = True

        if not finish_flag:
            intent = get_intent(self.cls_intent(history))
            if intent in ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']:
                out_text = {'end':True,'message':get_doc_role(intent), 'intentCode':'doc_role'}
            elif intent == "schedule_manager":
                his = self.history_compose(history)
                output_text = self.tsm._run(his, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
            elif intent == "other":
                output_text = self.chatter_gaily(history, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
            else:
                ext_info_args = baseVarsForPromptEngine()
                external_information = self.promptEngine._call(ext_info_args, concat_keyword=",")
                input_history = self.compose_input_history(history, external_information, **kwargs)
                out_history = self.generate(history=input_history, verbose=kwargs.get('verbose', False))
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
            
            if kwargs.get("streaming", True):
                # 直接返回字符串模式
                logger.debug('输出为：' + json.dumps(out_text, ensure_ascii=False))
                yield out_text
            else:
                # 保留完整的历史内容
                return out_history

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
    ori_input_param = testParam.param_bug1111
    # prompt = TOOL_CHOOSE_PROMPT
    
    prompt = ori_input_param['prompt']
    history = ori_input_param['history']
    intentCode = "default_code"
    output_text = next(chat.run_prediction(history, prompt, intentCode, verbose=True, orgCode="sf", customId="007"))
    while True:
        history.append({"role": "3", "content": output_text['message']})
        conv = history[-1]
        history.append({"role": "0", "content": input("user: ")})
        output_text = next(chat.run_prediction(history, prompt, intentCode, verbose=True))
