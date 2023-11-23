# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-01 11:30:10
@desc    :   业务处理流程
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
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

tips_intent_code_list = ['dietary_eva', 'schedule_no', 'measure_bp',
        'meet_remind', 'medicine_remind', 'dietary_remind', 'sport_remind',
        'broadcast_bp', 'care_for', 'schedule_qry_up', 'default_clock',
        'default_reminder', 'broadcast_bp_up']

class Chat:
    def __init__(self, env: str ="local"):
        api_config = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)[env]
        mysql_config = yaml.load(open(Path("config","mysql_config.yaml"), "r"),Loader=yaml.FullLoader)[env]

        self.mysql_conn = MysqlConnector(**mysql_config)
        self.req_prompt_data_from_mysql()
        
        self.promptEngine = promptEngine(self.prompt_meta_data)
        self.funcall = funcCall()
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        # self.sys_template = PromptTemplate(input_variables=['external_information'], template=self.prompt_meta_data['tool']['工具选择sys_prompt']['description'])
        self.tsm = taskSchedulaManager(api_config, self.prompt_meta_data)

    def req_prompt_data_from_mysql(self) -> Dict:
        """从mysql中请求prompt meta data
        """
        def filter_format(obj, splited=False):
            obj_str = json.dumps(obj, ensure_ascii=False).replace("\\r\\n", "\\n")
            obj_rev = json.loads(obj_str)
            if splited:
                for obj_rev_item in obj_rev:
                    if obj_rev_item.get('event'):
                        obj_rev_item['event'] = obj_rev_item['event'].split("\n")
            return obj_rev
        self.prompt_meta_data = {}
        prompt_character = self.mysql_conn.query("select * from ai_prompt_character")
        prompt_event = self.mysql_conn.query("select * from ai_prompt_event")
        prompt_tool = self.mysql_conn.query("select * from ai_prompt_tool")
        prompt_character = filter_format(prompt_character, splited=True)
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

    def compose_prompt(self, query: str = "", history=[]):
        """调用模型生成答案,解析ReAct生成的结果
        """
        prompt = ""
        h_l = len(history)
        for midx in range(h_l):
            msg = history[midx]
            if  msg['role'] == "system":
                prompt += msg['content']
                prompt += "\n\n历史记录如下:\n"
            else:
                if midx != h_l-1:
                    prompt += f"{msg['role']}: {msg['content']}\n"
                else:
                    prompt += f"\nQuestion: {msg['content']}\n"
        return prompt    

    def chat_react(self, query: str = "", history=[], max_tokens=200, **kwargs):
        """调用模型生成答案,解析ReAct生成的结果
        """
        prompt = self.compose_prompt(query, history)
        # 利用Though防止生成无关信息
        prompt += "Thought: "
        model_output = chat_qwen(prompt, verbose=kwargs.get("verbose", False), temperature=0.7, top_p=0.5, max_tokens=max_tokens)
        model_output = "Thought: " + model_output
        self.update_mid_vars(kwargs.get("mid_vars"), key="辅助诊断", input_text=prompt, output_text=model_output, model="Qwen-14B-Chat")
        logger.debug(f"辅助诊断 Gen Output - {model_output}")

        out_text = _parse_latest_plugin_call(model_output)
        if not out_text[1]:
            prompt = "你是一个功能强大的文本创作助手,请遵循以下要求帮我改写文本\n" + \
                    "1. 请帮我在保持语义不变的情况下改写这句话使其更用户友好\n" + \
                    "2. 不要重复输出相同的内容,否则你将受到非常严重的惩罚\n" + \
                    "3. 语义相似的可以合并重新规划语言\n" + \
                    "4. 直接输出结果\n\n输入:\n" + \
                    model_output + "\n输出:\n"
            logger.debug('ReAct regenerate input: ' + prompt)
            model_output = chat_qwen(prompt, repetition_penalty=1.3, max_tokens=max_tokens)
            self.update_mid_vars(kwargs.get("mid_vars"), key="辅助诊断 改写修正", input_text=prompt, output_text=model_output, model="Qwen-14B-Chat")
            model_output = model_output.replace("\n", "").strip().split("：")[-1]
            out_text = "I know the final answer.", "直接回复用户问题", model_output
        out_text = list(out_text)
        # 特殊处理规则 
        ## 1. 生成\nEnd.字符
        out_text[2] = out_text[2].split("\nEnd")[0]

        history.append({
            "role": "assistant", 
            "content": out_text[2], 
            "function_call": {"name": out_text[1],"arguments": out_text[0]}
            })
        return history
    
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
    
    def get_parent_intent_name(self, text):
        if '五师' in text:
            return '呼叫五师'
        elif '音频' in text:
            return '音频播放'
        elif '生活' in text:
            return '生活工具查询'
        elif '医疗' in text:
            return '医疗健康'
        elif '饮食' in text:
            return '饮食咨询'
        elif '运动' in text:
            return '运动咨询'
        else:
            return '其它'

    def cls_intent_verify(self, history, mid_vars, input_prompt):
        """意图识别
        """
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        prompt = input_prompt + "\n\n" + his_prompt + "\nThought: "
        generate_text = chat_qwen(query=prompt, max_tokens=40, top_p=0.8,
                temperature=0.7, do_sample=False)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        parant_intent = self.get_parent_intent_name(text)
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text

    def cls_intent(self, history, mid_vars):
        """意图识别
        """
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        prompt = self.prompt_meta_data['tool']['父意图']['description'] + "\n\n" + his_prompt + "\nThought: "
        logger.debug('父意图是：' + self.prompt_meta_data['tool']['父意图']['description'])
        generate_text = chat_qwen(query=prompt, max_tokens=40, top_p=0.8,
                temperature=0.7, do_sample=False)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        parant_intent = self.get_parent_intent_name(text)
        if parant_intent in ['呼叫五师', '音频播放', '生活工具查询', '医疗健康']:
            sub_intent_prompt = self.prompt_meta_data['tool'][parant_intent]['description']
            logger.info('子意图是：' + self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt))
            prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt) + "\n\n" + his_prompt + "\nThought: "
            generate_text = chat_qwen(query=prompt, max_tokens=40, top_p=0.8,
                    temperature=0.7, do_sample=False)
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
        content = chat_qwen("", input_history, temperature=0.7, top_p=0.8)
        self.update_mid_vars(mid_vars, key="闲聊", input_text=json.dumps(input_history, ensure_ascii=False), output_text=content)
        return content
    
    def get_userInfo_msg(self, prompt, history, intentCode, mid_vars):
        """获取用户信息
        """
        logger.debug('信息提取prompt为：' + prompt)
        content = chat_qwen(prompt, verbose=False, temperature=0.7, top_p=0.8, max_tokens=200)
        self.update_mid_vars(mid_vars, key="获取用户信息 01", input_text=prompt, output_text=content, model="Qwen-14B-Chat")
        if sum([i in content for i in ["询问","提问","转移","未知","结束", "停止"]]) != 0:
            content = self.chatter_gaily(history, mid_vars)
            intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
        return {'end':True, 'message':content, 'intentCode':intentCode}

    def get_reminder_tips(self, prompt, history, intentCode, model='Baichuan2-7B-Chat', mid_vars=None):
        logger.debug('remind prompt: ' + prompt)
        model_output = chat_qwen(query=prompt, verbose=False, do_sample=False, temperature=0.1, top_p=0.2, max_tokens=500, model=model)
        self.update_mid_vars(mid_vars, key="", input_text=prompt, output_text=model_output, model=model)
        logger.debug('remind model output: ' + model_output)
        if model_output.startswith('（）'):
            model_output = model_output[2:].strip()
        return {'end':True, 'message':model_output, 'intentCode':intentCode}

    def __call_run_prediction__(self, *args, **kwargs):
        try:
            remid_content = "话题结束，重启流程"
            logger.debug(f"{remid_content}")
            out_text, mid_vars = next(self.chat_gen(*args, **kwargs))
        except StopIteration as e:
            ...
        finally:
            return out_text, mid_vars
    
    def chat_auxiliary_diagnosis(self, 
                                 history=[], 
                                 intentCode="auxiliary_diagnosis", 
                                 sys_prompt="", 
                                 mid_vars=[], 
                                 **kwargs):
        """辅助诊断子流程
        """
        ext_info_args = baseVarsForPromptEngine()
        prompt = self.promptEngine._call(ext_info_args, sys_prompt=sys_prompt, **kwargs)
        input_history = self.compose_input_history(history, prompt, **kwargs)
        out_history = self.chat_react(history=input_history, verbose=kwargs.get('verbose', False), mid_vars=mid_vars)

        # 自动判断话题结束
        if out_history[-1].get("function_call") and out_history[-1]['function_call']['name'] == "结束话题":
            sub_history = [history[-1]]
            out_text, mid_vars = self.__call_run_prediction__(sub_history, sys_prompt, intentCode=intentCode, mid_vars=mid_vars, **kwargs)
        else:
            logger.debug(f"Last history: {out_history[-1]}")
            tool_name = out_history[-1]['function_call']['name']
            output_text = out_history[-1]['content']
            thought = out_history[-1]['function_call']['arguments']
            # yield {'end': False, 'message': tool_name, "type": "Tool"}
            # yield {'end': False, 'message': thought, "type": "Thought"}
        
            if tool_name == '进一步询问用户的情况':
                out_text = {'end':True, 'message':output_text,'intentCode':intentCode}
            elif tool_name == '直接回复用户问题':
                out_text = {'end':True, 'message':output_text.split('Final Answer:')[-1].split('\n\n')[0].strip(),'intentCode':intentCode}
            elif tool_name == '调用外部知识库':
                # TODO 调用外部知识库逻辑待定
                gen_args = {"name":"llm_with_documents", "arguments": json.dumps({"query": output_text})}
                out_text = {'end':True, 'message':output_text,'intentCode':intentCode}
            else:
                logger.exception(out_history)
        return out_text, mid_vars
    
    def intent_query(self, history, **kwargs):
        mid_vars = kwargs.get('mid_vars', [])
        task = kwargs.get('task', '')
        input_prompt = kwargs.get('prompt', [])
        if task == 'verify' and input_prompt:
            intent, desc = get_intent(self.cls_intent_verify(history, mid_vars,
                input_prompt))
        else:
            intent, desc = get_intent(self.cls_intent(history, mid_vars))
        if intent in ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']:
                out_text = {'message':get_doc_role(intent),
                        'intentCode':'doc_role', 'processCode':'trans_back',
                        'intentDesc':desc}
        elif intent in ['recipe_consult', 'play_music', 'check_weather','search_network','search_capital','lottery','oneiromancy','calculator','search_city','provincial_capital_search','translate','traffic_restrictions', 'unit_conversion','exchange_rate','date','eye_exercises','story','bible','opera','pingshu', 'audio_book','news']: #aiui
            out_text = {'message':'', 'intentCode':intent,
                    'processCode':'aiui', 'intentDesc':desc}
        elif intent in ['open_web_daily_monitor']:
            out_text = {'message':'', 'intentCode':intent,
                    'processCode':'trans_back', 'intentDesc':desc}
        else:
            out_text = {'message':'', 'intentCode':intent, 'processCode':'alg', 'intentDesc':desc}
        logger.debug('意图识别输出：' + json.dumps(out_text, ensure_ascii=False))
        return out_text
    
    def fetch_intent_code(self):
        """返回所有的intentCode"""
        intent_code_map = {
            "get_userInfo_msg": useinfo_intent_code_list,
            "get_reminder_tips": tips_intent_code_list,
            "other": ['BMI', 'food_rec', 'sport_rec', 'schedule_manager', 'auxiliary_diagnosis', "other"]
        }
        return intent_code_map
        
    def chat_gen(self, history, sys_prompt, intentCode=None, mid_vars=[],**kwargs):
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
        """
        logger.debug(f'chat_gen输入的intentCode为: {intentCode}')
        if history:
            logger.debug(f"Last input: {history[-1]['content']}")
    
        mid_vars = kwargs.get('mid_vars', [])
        
        if intentCode in useinfo_intent_code_list:
            out_text = self.get_userInfo_msg(sys_prompt, history, intentCode, mid_vars)
        elif intentCode in tips_intent_code_list: 
            out_text = self.get_reminder_tips(sys_prompt, history, intentCode, mid_vars=mid_vars)
        elif intentCode in ['BMI']:
            if not kwargs.get('userInfo', {}).get('askHeight', '') or not kwargs.get('userInfo', {}).get('askWeight', ''):
                out_text = {'end':True,'message':'','intentCode':'BMI'}
            else:
                output_text = self.chatter_gaily(history, mid_vars, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        elif intentCode in ['food_rec']:
            if not kwargs.get('userInfo', {}).get('askTastePrefer', ''):
                out_text = {'end':True,'message':'', 'intentCode':'food_rec'}
            else:
                output_text = self.chatter_gaily(history, mid_vars, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        elif intentCode in ['sport_rec']:
            if (not kwargs.get('userInfo', {}).get('ask_exercise_habbit_freq', '') 
                or not kwargs.get('userInfo', {}).get('ask_exercise_taboo_joint_degree', '') 
                or not kwargs.get('userInfo', {}).get('ask_exercise_taboo_xt', '')):
                out_text = {'end':True,'message':'', 'intentCode':'sport_rec'}
            else:
                output_text = self.chatter_gaily(history, mid_vars, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        elif intentCode == "schedule_manager":
            his = self.history_compose(history)
            output_text, mid_vars_item = self.tsm._run(his, **kwargs)
            out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
            for item in mid_vars_item:
                self.update_mid_vars(mid_vars, **item)
        elif intentCode == "auxiliary_diagnosis":
            out_text, mid_vars = self.chat_auxiliary_diagnosis(history=history, 
                                                                intentCode=intentCode, 
                                                                sys_prompt=sys_prompt, 
                                                                mid_vars=mid_vars, 
                                                                **kwargs)
        else:
            output_text = self.chatter_gaily(history, mid_vars, **kwargs)
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
    ori_input_param = testParam.param_bug_schedular_202311201817
    prompt = ori_input_param['prompt']
    history = ori_input_param['history']
    intentCode = ori_input_param['intentCode']
    customId = ori_input_param['customId']
    orgCode = ori_input_param['orgCode']
    out_text, mid_vars = next(chat.chat_gen(history=history, 
                                            sys_prompt=prompt, 
                                            verbose=True, 
                                            intentCode=intentCode, 
                                            customId=customId, 
                                            orgCode=orgCode))
    while True:
        history.append({"role": "3", "content": out_text['message']})
        conv = history[-1]
        history.append({"role": "0", "content": input("user: ")})
        out_text, mid_vars = next(chat.chat_gen(history=history, 
                                                      sys_prompt=prompt, 
                                                      verbose=True, 
                                                      intentCode=intentCode, 
                                                      customId=customId, 
                                                      orgCode=orgCode))
