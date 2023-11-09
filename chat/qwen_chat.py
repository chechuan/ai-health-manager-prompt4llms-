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
from typing import Tuple

from langchain.prompts import PromptTemplate

from chat.plugin_util import funcCall
from chat.qwen_react_util import *
from chat.special_intent_msg_util import *
from config.constrant import INTENT_PROMPT, TOOL_CHOOSE_PROMPT
from config.function_call_config import function_tools
from src.prompt.factory import baseVarsForPromptEngine, promptEngine
from src.prompt.model_init import chat_qwen
from src.prompt.task_schedule_manager import taskSchedulaManager
from utils.Logger import logger

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
    """通过关键词解析意图->code
    """
    if '创建提醒' in text:
        code = 'create_alert'
    elif '饮食' in text and '咨询' in text:
        code = 'food'
    elif '菜谱' in text:
        code = 'recipe_consult'
    elif '音乐' in text:
        code = 'play_music'
    elif '天气' in text:
        code = 'check_weather'
    elif '辅助诊断' in text:
        code = 'auxiliary_diagnosis'
    elif '医师' in text:
        code = 'call_doctor'
    elif '运动师' in text:
        code = 'call_sportMaster'
    elif '心理' in text:
        code = 'call_psychologist'
    elif '修改提醒' in text:
        code = 'change_alert'
    elif '取消提醒' in text:
        code = 'cancel_alert'
    elif '营养师' in text:
        code = 'call_dietista'
    elif '健管师' in text:
        code = 'call_health_manager'
    elif '其它意图' in text:
        code = 'other'
    elif '日程管理'in text:
        code = 'schedule_manager'
    else:
        code = 'other'
    logger.debug(f'识别出的意图:{text} code:{code}')
    return code

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

def _parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    h = text.find('Thought:')
    i = text.find('\nAction:')
    j = text.find('\nAction Input:')
    k = text.find('\nObservation:')
    l = text.find('\nFinal Answer:')
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
    elif l > 0:
        if h > 0:
            plugin_thought = text[h + len('Thought:'):l].strip()
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            return plugin_thought, "直接回复用户问题", plugin_args
        else:
            plugin_args = text[l + len('\nFinal Answer:'):].strip()
            return "I know the final answer.", "直接回复用户问题", plugin_args
    return '', ''

class Chat(object):
    def __init__(self, env: str ="local"):
        api_config = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)[env]
        self.promptEngine = promptEngine()
        self.funcall = funcCall()
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        self.tsm = taskSchedulaManager(api_config)
        
        # self.sys_template_chatter_gaily = 
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
        if kwargs.get("verbose"):
            logger.debug(f"Generate Prompt - length:{len(query)}\n{query}")
        model_output = chat_qwen(query, verbose=kwargs.get("verbose", False), temperature=0.7, top_p=0.5, max_tokens=max_tokens)
        
        model_output = "\nThought: " + model_output
        out_text = _parse_latest_plugin_call(model_output)
        if not out_text[1]:
            query = "你是一个功能强大的文本创作助手,请遵循以下要求帮我改写文本\n" + \
                    "1. 请帮我在保持语义不变的情况下改写这句话使其更用户友好\n" + \
                    "2. 不要重复输出相同的内容,否则你将受到非常严重的惩罚\n" + \
                    "3. 语义相似的可以合并重新规划语言\n" + \
                    "4. 直接输出结果\n\n输入:\n" + \
                    model_output + "\n输出:\n"
            model_output = chat_qwen(query, repetition_penalty=1.3, max_tokens=max_tokens)
            model_output = model_output.replace("\n", "").strip().split("：")[-1]
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

    def history_compose(self, history):
        return [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]

    def cls_intent(self, history):
        """意图识别"""
        st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([f"{i['role']}: {i['content']}" for i in history])
        prompt = INTENT_PROMPT + his_prompt + "\n\n用户的意图是(只输出意图):"
        output_text = chat_qwen(query=prompt, max_tokens=5, top_p=0.5, temperature=0.7)
        return output_text
    
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
            "现在你是智能健康管家, 为居家用户提供健康咨询和管理服务\n"
            "对于日常闲聊，有以下几点建议:\n"
            "1. 整体过程应该是轻松愉快的\n"
            "2. 你可以适当发挥一点幽默基因\n"
            "3. 对用户是友好的\n"
            "4. 当问你是谁或叫什么名字时,你应当说我是智能健康管家")
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
        if intentCode in useinfo_intent_code_list:
            yield get_userInfo_msg(sys_prompt, history, intentCode)
        elif intentCode != 'default_code':
            yield get_reminder_tips(sys_prompt, history, intentCode) 

        intent = get_intent(self.cls_intent(history))
        if intent in ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista', 'call_health_manager']:
            yield {'end':True,'message':get_doc_role(intent), 'intentCode':'doc_role'}
        elif intent == "schedule_manager":
            his = self.history_compose(history)
            output_text = self.tsm._run(his, **kwargs)
            out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        elif intent == "other":
            output_text = self.chatter_gaily(history, external_information, **kwargs)
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
    debug_text = "肚子疼"
    # history = [{"role": "0", "content": init_intput}]
    # history = [{'msgId': '6132829035', 'role': '1', 'content': debug_text, 'sendTime': '2023-11-06 14:40:11'}]
    ori_input_param = {
        'orgCode': 'sf', 
        'customId': '1104a89793b43b76528', 
        'history': [
            {'msgId': '2655508861', 'role': '1', 'content': '你叫什么名字？', 'sendTime': '2023-11-08 09:22:38'}, 
            {'msgId': '8687772106', 'role': '3', 'content': '我是智能健康管家，您可以叫我小智。', 'sendTime': '2023-11-08 09:22:39'}, 
            {'msgId': '7139475722', 'role': '1', 'content': '我今天有点肚子疼。', 'sendTime': '2023-11-08 09:22:48'}, 
            # {'msgId': '5086931766', 'role': '3', 'content': '您好，我是小智，很高兴为您服务。您今天肚子疼，建议您及时就医，以便得到更好的治疗。如果您需要更多健康咨询和管理服务，欢迎随时联系我。期待为您提供更好的服务。', 'sendTime': '2023-11-08 09:22:51'}, 
            # {'msgId': '9756022981', 'role': '1', 'content': '那我现在肚子疼怎么办啊？你能给我点建议吗？', 'sendTime': '2023-11-08 09:23:14'}, 
            # {'msgId': '9078524669', 'role': '3', 'content': '这就为您播放《我通灵的那些年》', 'sendTime': '2023-11-08 09:23:15'}, 
            # {'msgId': '5587694936', 'role': '1', 'content': '不对呀，我是问你肚子疼怎么办？', 'sendTime': '2023-11-08 09:23:22'}, 
            # {'msgId': '2599541921', 'role': '3', 'content': '如果您现在肚子疼，建议您及时就医，以便得到更好的治疗。如果您需要更多健康咨询和管理服务，欢迎随时联系我。期待为您提供更好的服务。', 'sendTime': '2023-11-08 09:23:24'}, 
            # {'msgId': '8810390843', 'role': '1', 'content': '你不问问我肚子那块疼吗？', 'sendTime': '2023-11-08 09:23:34'}
        ], 
        'prompt': '你作为家庭智能健康管家，需要解答用户问题，如果用户回复了症状，则针对用户症状进行问诊；当用户提到高血压的相关症状时，如果对话历史中没有血压信息，则提示用户测量血压，如果对话中问过血压信息，就不要再问了，如果对话历史里有血压信息，则针对用户高血压症状问诊；如果用户提到其他疾病症状则对用户症状进行问诊。如果模型回复饮食情况则对用户饮食进行热量、膳食结构评价。今天日期是2023年11月08日。用户个人信息如下：\\n对话内容为：', 
        'intentCode': 'default_code'
        }
    
    prompt = ('你作为家庭智能健康管家，需要解答用户问题，如果用户回复了症状，则针对用户症状进行问诊；'
            '当用户提到高血压的相关症状时，如果对话历史中没有血压信息，则提示用户测量血压，如果对话中问过血压信息，'
            '就不要再问了，如果对话历史里有血压信息，则针对用户高血压症状问诊；如果用户提到其他疾病症状则对用户症状进行问诊。'
            '如果模型回复饮食情况则对用户饮食进行热量、膳食结构评价。今天日期是2023年11月06日。'
            '用户个人信息如下：\\n对话内容为：')
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
