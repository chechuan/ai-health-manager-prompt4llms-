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

TOOL_CHOOSE_PROMPT = """你是智能健康管家，可以根据用户的对话内容，从工具列表中选择对应工具完成用户的任务。

现提供以下工具:
- 调用外部知识库: 允许你在自身能力无法结局当前问题时调用外部知识库获取更专业的知识解决问题，提供帮助
- 进一步询问用户的情况: 当前用户提供的信息不足，需要进一步询问用户相关信息以提供更全面，具体的帮助
- 直接回复用户问题：问题过于简单，且无信息缺失，可以直接回复

请遵循以下格式回复:
Thought: 思考当前应该做什么
Action: 选择的解决用户当前问题的工具
Action Input: 当前工具需要用的参数,可以是调用知识库的参数,询问用户的问题,一次只针对一个主题询问
Observation: 工具返回的内容
...(Thought/Action/Action Input 可能会循环一次或多次直到解决问题)
Thought: bingo
Final Answer: the final answer to the original input question

[对话背景信息]
{external_information}
[对话背景信息]"""

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

    def run_prediction(self, 
                       history, 
                       sys_prompt: str = TOOL_CHOOSE_PROMPT, 
                       intentCode=None,
                       **kwargs):
        """主要业务流程

        1. 拼装外部信息
        2. 准备模型输入messages
        3. 模型生成结果
        """
        ext_info_args = baseVarsForPromptEngine()
        external_information = self.promptEngine._call(ext_info_args, concat_keyword=",")

        input_history = self.compose_input_history(history, external_information, **kwargs)

        history = self.generate(history=input_history, verbose=kwargs.get('verbose', False))
        
        if history[-1].get("function_call"):
            print(f"Thought: {history[-1]['function_call']['arguments']}")

        tool_name = history[-1]['function_call']['name']
        output_text = history[-1]['content']

        if tool_name == '进一步询问用户的情况' or tool_name == '直接回复用户问题':
            ...
        elif tool_name == '调用外部知识库':
            gen_args = {"name":"llm_with_documents", "arguments": json.dumps({"query": output_text})}
            output_text = self.funcall._call(gen_args, verbose=True)

        if not kwargs.get("streaming", False):
            # 直接返回字符串模式
            return output_text
        else:
            # 保留完整的历史内容
            return history

if __name__ == '__main__':
    chat = Chat()
    history = [{"role": "0", "content": "我最近早上头疼"}]
    prompt = TOOL_CHOOSE_PROMPT
    intentCode = None
    output_text = chat.run_prediction(history, prompt, intentCode, verbose=False)
    while True:
        history.append({"role": "3", "content": output_text})
        conv = history[-1]
        print(f"Role: {conv['role']}\nContent: {conv['content']}")
        history.append({"role": "0", "content": input("user: ")})
        output_text = chat.run_prediction(history, prompt, intentCode, verbose=False)