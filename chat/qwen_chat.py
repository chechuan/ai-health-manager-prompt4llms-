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

sys.path.append('.')
from typing import Tuple

from langchain.prompts import PromptTemplate

from chat.plugin_util import funcCall
from chat.qwen_react_util import *
from chat.special_intent_msg_util import *
from config.constrant import INTENT_PROMPT, TOOL_CHOOSE_PROMPT
from config.function_call_config import function_tools
from src.prompt.factory import baseVarsForPromptEngine, promptEngine
from src.prompt.model_init import chat_qwen

# role_map = {
#         '0': '<用户>',
#         '1': '<用户>',
#         '2': '<医生>',
#         '3': '<智能健康管家>'
# }
role_map = {
        '0': 'user',
        '1': 'user',
        '2': 'doctor',
        '3': 'assistant'
}

def get_intent(text):
    if '创建提醒' in text:
        return 'create_alert'
    elif '饮食' in text and '咨询' in text:
        return 'food'
    elif '菜谱' in text:
        return 'recipe_consult'
    elif '音乐' in text:
        return 'play_music'
    elif '天气' in text:
        return 'check_weather'
    elif '辅助诊断' in text:
        return 'auxiliary_diagnosis'
    elif '医师' in text:
        return 'call_doctor'
    elif '运动师' in text:
        return 'call_sportMaster'
    elif '心理' in text:
        return 'call_psychologist'
    elif '修改提醒' in text:
        return 'change_alert'
    elif '取消提醒' in text:
        return 'cancel_alert'
    elif '营养师' in text:
        return 'call_dietista'
    elif '健管师' in text:
        return 'call_health_manager'
    elif '其它意图' in text:
        return 'other'
    else:
        return 'other'

def get_doc_role(code):
    if code == 'call_dietista':
        return 'ROLE_NUTRITIONIST'
    elif code == 'call_sportMaster':
        return 'ROLE_EXERCISE_SPECIALIST'
    elif code == 'call_psychologist':
        return 'ROLE_EMOTIONAL_COUNSELOR'
    elif code == 'call_doctor':
        return 'ROLE_DOCTOR'
    elif code == 'call_health_manager':
        return 'ROLE_HEALTH_SPECIALIST'
    else:
        return 'ROLE_HEALTH_SPECIALIST'

useinfo_intent_code_list = [
    'ask_name','ask_age','ask_exercise_taboo','sk_exercise_habbit','ask_food_alergy','ask_food_habbit','ask_taste_prefer',
    'ask_family_history','ask_labor_intensity','ask_nation','ask_disease','ask_weight','ask_height', 
    'ask_six', 'ask_mmol_drug', 'ask_exercise_taboo_degree', 'ask_exercise_taboo_xt'
]

def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    h = text.find('Thought:')
    i = text.find('\nAction:')
    j = text.find('\nAction Input:')
    k = text.find('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_thought = text[h + len('Thought:'):i].strip()
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_thought, plugin_name, plugin_args
    return '', ''

class Chat(object):
    def __init__(self, ):
        self.promptEngine = promptEngine()
        self.funcall = funcCall()
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
    # def __init__(self, tokenizer, model):
    #     self.tokenizer = tokenizer
    #     self.model = model
    
    def get_tool_name(self, text):
        if '外部知识' in text:
            return '调用外部知识库'
        elif '询问用户' in text:
            return '进一步询问用户的情况'
        elif '直接回复' in text:
            return '直接回复用户问题'
        else:
            return '直接回复用户问题'

    def generate(self, query: str = "", history=[], **kwargs):
        """调用模型生成答案,解析"""
        # for i in range(10):
        model_output = chat_qwen(query, history, verbose=kwargs.get("verbose", False), temperature=0.7, top_p=0.8, max_tokens=200)
        out_text = parse_latest_plugin_call(model_output)
        if not out_text[1]:
            query = "帮我调整一下这句话直接给用户输出:"+ model_output + "输出结果:"
            model_output = chat_qwen(query)
            model_output = model_output.split("：")[-1]
            out_text = "I know the final answer.", "直接回复用户问题", model_output
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
    
    def cls_intent(self, history):
        """意图识别"""
        st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        prompt = INTENT_PROMPT + his_prompt
        output_text = chat_qwen(query=prompt, max_tokens=50, top_p=0.5, temperature=0.7)
        intent = output_text[output_text.find("Action: ")+8:]
        return intent

    def run_prediction(self, 
                       history, 
                       sys_prompt: str = TOOL_CHOOSE_PROMPT, 
                       intentCode=None,
                       **kwargs):
        """主要业务流程
        1. 意图识别
        2. 不同意图进入不同的处理流程

        ## 多轮交互流程
        1. 拼装外部信息
        2. 准备模型输入messages
        3. 模型生成结果
        """
        intent = get_intent(self.cls_intent(history))
        print('用户意图是：' + intent)
        
        if intentCode in useinfo_intent_code_list:
            yield get_userInfo_msg(sys_prompt, history, intentCode)
        elif intentCode != 'default_code':
            yield get_reminder_tips(sys_prompt, history, intentCode) 

        if intent in ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']:
            yield {'end':True,'message':get_doc_role(intent), 'intentCode':'doc_role'}
        else:
            ext_info_args = baseVarsForPromptEngine()
            external_information = self.promptEngine._call(ext_info_args, concat_keyword=",")
            input_history = self.compose_input_history(history, external_information, **kwargs)
            out_history = self.generate(history=input_history, verbose=kwargs.get('verbose', False))
        
            if out_history[-1].get("function_call"):
                print(f"Thought: {out_history[-1]['function_call']['arguments']}")
            print(out_history[-1])
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
                print('输出为：' + json.dumps(out_text, ensure_ascii=False))
                yield out_text
            else:
                # 保留完整的历史内容
                return out_history

if __name__ == '__main__':
    chat = Chat()
    a = "我最近早上头疼，谁帮我看一下啊"
    init_intput = input("init_input: ")
    history = [{"role": "0", "content": init_intput}]
    prompt = TOOL_CHOOSE_PROMPT
    intentCode = "default_code"
    output_text = next(chat.run_prediction(history, prompt, intentCode, verbose=False))
    while True:
        history.append({"role": "3", "content": output_text['message']})
        conv = history[-1]
        history.append({"role": "0", "content": input("user: ")})
        output_text = next(chat.run_prediction(history, prompt, intentCode, verbose=False))
