# -*- encoding: utf-8 -*-

import json, asyncio
import sys
from copy import deepcopy

sys.path.append('.')

from langchain.prompts import PromptTemplate

from chat.constant import *
from data.constrant import INTENT_PROMPT, TOOL_CHOOSE_PROMPT, role_map
from chat.util import *
from data.test_param.test import testParam
from src.prompt.factory import baseVarsForPromptEngine, promptEngine
from src.prompt.model_init import callLLM
from src.prompt.task_schedule_manager import taskSchedulaManager
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.database import MysqlConnector
from src.utils.module import (_parse_latest_plugin_call, clock,
                              get_doc_role, get_intent, make_meta_ret)
from src.prompt.utils import ChatterGailyAssistant


class Chat:
    def __init__(self, global_share_resource: InitAllResource) -> None:
        global_share_resource.chat = self
        self.global_share_resource = global_share_resource
        self.env = global_share_resource.args.env
        self.assistant = ChatterGailyAssistant(global_share_resource)

        self.mysql_conn = MysqlConnector(**global_share_resource.mysql_config)
        self.prompt_meta_data = global_share_resource.prompt_meta_data

        self.promptEngine = promptEngine(self.prompt_meta_data)
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        # self.sys_template = PromptTemplate(input_variables=['external_information'], template=self.prompt_meta_data['tool']['工具选择sys_prompt']['description'])
        self.tsm = taskSchedulaManager(global_share_resource)
        self.__initalize_intent_map__()

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
        logger.debug(f"辅助诊断 Input:\n{prompt}")
        model_output = callLLM(prompt, temperature=0.7, top_p=0.5, max_tokens=max_tokens)
        model_output = "\nThought: " + model_output
        self.update_mid_vars(kwargs.get("mid_vars"),
                             key="辅助诊断",
                             input_text=prompt,
                             output_text=model_output,
                             model="Qwen1.5-14B-Chat")
        logger.debug(f"辅助诊断 Gen Output:\n{model_output}")

        out_text = _parse_latest_plugin_call(model_output)
        if not out_text[1]:
            prompt = "你是一个功能强大的文本创作助手,请遵循以下要求帮我改写文本\n" + \
                    "1. 请帮我在保持语义不变的情况下改写这句话使其更用户友好\n" + \
                    "2. 不要重复输出相同的内容,否则你将受到非常严重的惩罚\n" + \
                    "3. 语义相似的可以合并重新规划语言\n" + \
                    "4. 直接输出结果\n\n输入:\n" + \
                    model_output + "\n输出:\n"

            logger.debug('ReAct regenerate input: ' + prompt)
            model_output = callLLM(prompt, repetition_penalty=1.3, max_tokens=max_tokens)

            self.update_mid_vars(kwargs.get("mid_vars"),
                                 key="辅助诊断 改写修正",
                                 input_text=prompt,
                                 output_text=model_output,
                                 model="Qwen1.5-14B-Chat")

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
            return '饮食营养'
        elif '运动咨询' in text:
            return '运动咨询'
        elif '日程' in text:
            return '日程管理'
        elif '健康咨询' in text:
            return '健康咨询'
        elif '健康处方' in text:
            return '健康处方'
        else:
            return '其它'

    def cls_intent_verify(self, history, mid_vars, input_prompt):
        """意图识别
        """
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        history = history[-1:]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        his_prompt = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        prompt = input_prompt + "\n\n" + his_prompt + "\nThought: "
        generate_text = callLLM(query=prompt, max_tokens=40, top_p=0.8,
                temperature=0.7, do_sample=False)
        logger.debug('意图识别模型输出：' + generate_text)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text

    def cls_intent_v2(self, history, mid_vars, **kwargs):
        """意图识别
        """

        if len(history) > 1:
            h_p = "\n".join([("Question" if i['role'] == "user" else "Answer")
                + f": {i['content']}" for i in history[-5:-1]])
        else:
            h_p = "无"
        prefix = "Question" if history[-1]['role'] == "user" else "Answer"
        query = f"{prefix}: {history[-1]['content']}"

        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        if kwargs.get('intentPrompt', ''):
            prompt = kwargs.get('intentPrompt').format(h_p) + "\n\n" + query + "\nThought: "
        else:
            scene_prompt = get_parent_scene_intent(self.prompt_meta_data['intent'], kwargs.get('scene_code') or 'default')
            prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "

            # if kwargs.get('scene_code', 'default') == 'exhibition_hall_exercise':
            #     scene_prompt = get_scene_intent(self.prompt_meta_data['tool'], 'exhibition_hall_exercise')
            #     prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "
            # else:
            #     prompt = self.prompt_meta_data['tool']['父意图']['description'].format(h_p) + "\n\n" + query + "\nThought: "
        logger.debug('父意图模型输入：' + prompt)
        generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                temperature=0, do_sample=False, stop=['\nThought'], model='Qwen1.5-14B-Chat')
        logger.debug('父意图识别模型输出：' + generate_text)
        intentIdx = 0
        if 'Intent:' in  generate_text:
            intentIdx = generate_text.find("\nIntent: ") + 9
        elif '意图:' in generate_text:
            intentIdx = generate_text.find("\n意图:") + 4
        elif '\nFunction:' in generate_text:
            intentIdx = generate_text.find("\nFunction:") + 10
        text = generate_text[intentIdx:].split("\n")[0].strip()
        intent = text
        while not self.prompt_meta_data['intent'][intent]['is_leaf'] and (not kwargs.get('intentPrompt', '') or (kwargs.get('intentPrompt', '') and kwargs.get('subIntentPrompt', ''))):
            # sub_intent_prompt = self.prompt_meta_data['intent'][parant_intent]['description']
            if ['外呼'] in intent:
                history = history[-1:]
                query = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
                h_p = '无'
            if kwargs.get('subIntentPrompt', ''):
                prompt = kwargs.get('subIntentPrompt').format(h_p) + "\n\n" + query + "\nThought: "
            else:
                scene_prompt = get_sub_scene_intent_v2(self.prompt_meta_data['intent'], kwargs.get('scene_code') or 'default', intent)
                prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "
                # prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt, h_p) + "\n\n" + query + "\nThought: "
            logger.debug('子意图模型输入：' + prompt)
            generate_text = callLLM(query=prompt, max_tokens=200, top_p=0.8,
                    temperature=0, do_sample=False, stop=['\nThought'], model='Qwen1.5-14B-Chat')
            logger.debug('子意图模型输出：' + generate_text)
            intentIdx = 0
            if 'Intent:' in  generate_text:
                intentIdx = generate_text.find("\nIntent: ") + 9
            elif '意图:' in generate_text:
                intentIdx = generate_text.find("\n意图:") + 4
            elif '\nFunction:' in generate_text:
                intentIdx = generate_text.find("\nFunction:") + 10
            intent = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return intent

    def cls_intent(self, history, mid_vars, **kwargs):
        """意图识别
        """
        kwargs = deepcopy(kwargs)
        logger.debug(kwargs)
        user_id = kwargs.get("user_id", "anonymous")
        session_id = kwargs.get("session_id", "anonymous")
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        if kwargs.get('scene_code', 'default') != 'doctor':
            open_sch_list = ['打开','日程']
            market_list = ['打开','集市']
            home_list = ['打开','家居']
            bp_list = ['血压趋势图','血压录入','血压添加','入录血压','添加血压','历史血压','血压历史']
            food_purch_list = ['打开食材采购','食材采购','买什么食材','买什么食物', '买点什么食材', '买点什么食物']
            inter_info_list = ['打开聊天','打开交流','信息交互页面','打开语音交互','语音交互页面','查看聊天','聊天页面', '我的聊天', '看看聊天']
            # st_key, ed_key = "<|im_start|>", "<|im_end|>"
            gc = ['商量', '医院', '去', '到']
            # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
            if sum([1 for i in gc if i in history[-1]['content']]) >= 3:
                return '拉群共策'
            if sum([1 for i in bp_list if i in history[-1]['content']]) > 0:
                return '打开功能页面'
            if sum([1 for i in inter_info_list if i in history[-1]['content']]) > 0:
                return '打开功能页面'
            if sum([1 for i in food_purch_list if i in history[-1]['content']]) > 0:
                return '生成食材采购清单'
            if sum([1 for i in open_sch_list if i in history[-1]['content']]) >= 2:
                return '打开功能页面'
            if sum([1 for i in market_list if i in history[-1]['content']]) >= 2:
                return '打开功能页面'
            if sum([1 for i in home_list if i in history[-1]['content']]) >= 2:
                return '打开功能页面'
            if '换回' in history[-1]['content'] and ('数字人' in history[-1]['content'] or '形象' in history[-1]['content'] or '皮肤' in history[-1]['content']):
                return '换回数字人皮肤'
        if len(history) > 0:
            h_p = "\n".join([("user" if i['role'] == "user" else "assistant")
                + f": {i['content']}" for i in history[-5:]])
        else:
            h_p = "无"
        prefix = "Question" if history[-1]['role'] == "user" else "Answer"
        query = f"{prefix}: {history[-1]['content']}"

        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        if kwargs.get('intentPrompt', ''):
            prompt = kwargs.get('intentPrompt').format(h_p)
        else:
            scene_code = kwargs.get('scene_code')
            if scene_code in ['doctor', 'dm_expert_scene', "elderly_cognitive_training"]:
                scene_prompt = get_parent_scene_intent(self.prompt_meta_data['intent'], kwargs.get('scene_code'))
                # logger.debug(f"Generated scene_prompt: {scene_prompt}")
                prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt, h_p)
                model = 'Qwen1.5-32B-Chat'
                # logger.debug(f"Generated Prompt: {prompt}")
            else:
                scene_prompt = get_parent_scene_intent(self.prompt_meta_data['intent_detect'],
                                                       kwargs.get('scene_code') or 'default')
                prompt = self.prompt_meta_data['intent_detect']['意图模版']['description'].format(scene_prompt, h_p)
                model = 'Intent-Detect-0.5B'

            # if kwargs.get('scene_code', 'default') == 'exhibition_hall_exercise':
            #     scene_prompt = get_scene_intent(self.prompt_meta_data['tool'], 'exhibition_hall_exercise')
            #     prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "
            # else:
            #     prompt = self.prompt_meta_data['tool']['父意图']['description'].format(h_p) + "\n\n" + query + "\nThought: "
        logger.debug('意图模型输入：' + prompt)
        # 调用模型（Langfuse 追踪：父意图）
        his = [{
            'role': 'user',
            'content': prompt
        }]
        generate_text = callLLM(
            history=his,
            # query=prompt,
            max_tokens=200,
            top_p=0.8,
            temperature=0.0,
            do_sample=False,
            stop=['\nThought'],
            model=model,
            extra_params={
                "langfuse": self.global_share_resource.langfuse_client,
                "user_id": user_id,
                "session_id": session_id,
                "tokenizer": self.global_share_resource.qwen_tokenizer,
                "intent_code": "intent_classification",
                "name": "parent_intent",
                "tags": [],
                "release": "v1.0.0",
                "metadata": {"source": "intent"},
            }
        )
        logger.debug('意图识别模型输出：' + generate_text)
        intentIdx = 0
        if 'Intent:' in generate_text:
            intentIdx = generate_text.find("\nIntent: ") + 9
        elif '意图:' in generate_text:
            intentIdx = generate_text.find("\n意图:") + 4
        elif '\nFunction:' in generate_text:
            intentIdx = generate_text.find("\nFunction:") + 10
        text = generate_text[intentIdx:].split("\n")[0].strip()
        parant_intent = self.get_parent_intent_name(text)
        # if parant_intent in ['呼叫五师', '音频播放', '生活工具查询', '医疗健康', '饮食营养', '运动咨询', '健康咨询', '健康处方'] and (not kwargs.get('intentPrompt', '') or (kwargs.get('intentPrompt', '') and kwargs.get('subIntentPrompt', ''))):
        #     # sub_intent_prompt = self.prompt_meta_data['intent'][parant_intent]['description']
        #     if parant_intent in ['呼叫五师意图']:
        #         history = history[-1:]
        #         query = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])
        #         h_p = '无'
        #     if kwargs.get('subIntentPrompt', ''):
        #         prompt = kwargs.get('subIntentPrompt').format(h_p) + "\n\n" + query + "\nThought: "
        #     else:
        #         scene_prompt = get_sub_scene_intent(self.prompt_meta_data['intent'], kwargs.get('scene_code') or 'default', parant_intent)
        #         prompt = self.prompt_meta_data['intent']['意图模版']['description'].format(scene_prompt, h_p) + "\n\n" + query + "\nThought: "
        #         # prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(sub_intent_prompt, h_p) + "\n\n" + query + "\nThought: "
        #     logger.debug('子意图模型输入：' + prompt)
        #     # 调用模型（Langfuse 追踪：子意图）
        #     generate_text = callLLM(
        #         query=prompt,
        #         max_tokens=200,
        #         top_p=0.8,
        #         temperature=0.7,
        #         do_sample=False,
        #         stop=['\nThought'],
        #         model='Qwen2-7B-Instruct',
        #         extra_params={
        #             "langfuse": self.global_share_resource.langfuse_client,
        #             "user_id": user_id,
        #             "session_id": session_id,
        #             "tokenizer": self.global_share_resource.qwen_tokenizer,
        #             "intent_code": "intent_classification",
        #             "name": "sub_intent",
        #             "tags": [parant_intent],
        #             "release": "v1.0.0",
        #             "metadata": {"source": "intent"},
        #         }
        #     )
        #     logger.debug('子意图模型输出：' + generate_text)
        #     intentIdx = 0
        #     if 'Intent:' in  generate_text:
        #         intentIdx = generate_text.find("\nIntent: ") + 9
        #     elif '意图:' in generate_text:
        #         intentIdx = generate_text.find("\n意图:") + 4
        #     elif '\nFunction:' in generate_text:
        #         intentIdx = generate_text.find("\nFunction:") + 10
        #     text = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text

    def chatter_gaily(self, *args, **kwargs):
        """组装mysql中闲聊对应的prompt
        """
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in kwargs['history']]
        if len(history) > 8:    # 每次history理论上为奇数个, 例: if len == 9 从第二轮的问题开始
            history = history[-7:]
        kwargs['history'] = history
        backend_history = self.global_share_resource.chat_v2.chat_react(*args, **kwargs)
        content = backend_history[-1]['function_call']['arguments']
        # ext_info = self.prompt_meta_data['event']['闲聊']['description'] + "\n" + self.prompt_meta_data['event']['闲聊']['process']
        # input_history = [{"role":"system", "content": ext_info}] + input_history
        # content = callLLM("", input_history, temperature=0.7, top_p=0.8)
        # self.update_mid_vars(mid_vars, key="闲聊", input_text=json.dumps(input_history, ensure_ascii=False), output_text=content)
        return content

    def open_page(self, history, mid_vars):
        """组装mysql中打开页面对应的prompt
        """
        add_diet_list = ['打开记录','打开录入','打开添加']
        diet_record_list = ['饮食记录','饮食添加','打开推荐','饮食评估','食谱','我的饮食','食谱页面','餐食记录']
        market_list = ['集市']
        home_list = ['智能','家居','面板']
        input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        input_history = input_history[-3:]
        if '血压趋势图' in history[-1]['content']:
            return 'pagename:"bloodPressure-trend-chart"'
        elif '血压录入' in history[-1]['content']:
            return 'pagename:"add-blood-pressure"'
        elif '血压历史页面' in history[-1]['content'] or '历史血压页面' in history[-1]['content']:
            return 'pagename:"record-list3"'
        elif '打开' in history[-1]['content'] and '日程' in history[-1]['content']:
            return 'pagename:"my_schedule"'
        elif sum([1 for i in add_diet_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"add-diet"'
        elif sum([1 for i in diet_record_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"diet-record"'
        elif sum([1 for i in market_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"my-market"'
        elif sum([1 for i in home_list if i in input_history[-1]['content']]) > 0:
            return 'pagename:"smart-home"'
        elif '打开' in input_history[-1]['content'] and '日程' in input_history[-1]['content']:
            return 'pagename:"my-schedule"'

        hp = [h['role'] + ' ' + h['content'] for h in input_history]
        #ext_info = self.prompt_meta_data['event']['open_Function']['description'] + "\n" + self.prompt_meta_data['event']['open_Function']['process'].format('\n'.join(hp))
        #last_h = [h['role'] + ' ' + h['content'] for h in input_history[-1:]]
        hi = ''
        if len(input_history) > 1:
            hi = '用户历史会话如下，可以作为意图识别的参考，但不要过于依赖历史记录，因为它可能是狠久以前的记录：' + '\n' + '\n'.join([h["role"] + h["content"] for h in input_history[-3:-1]]) + '\n' + '当前用户输入：\n'
        hi += f'Question:{input_history[-1]["content"]}\nThought:'
        ext_info = self.prompt_meta_data['event']['open_Function']['description'] + "\n" + self.prompt_meta_data['event']['open_Function']['process'] + '\n' + hi
        input_history = [{"role":"system", "content": ext_info}]
        logger.debug('打开页面模型输入：' + json.dumps(input_history,ensure_ascii=False))
        content = callLLM("", input_history, temperature=0.7, top_p=0.8)
        if content.find('Answer') != -1:
            content = content[content.find('Answer')+7:].split('\n')[0].strip()
        elif content.find('Output') != -1:
            content = content[content.find('Output')+6].split('\n')[0].strip()
        self.update_mid_vars(mid_vars, key="打开功能画面", input_text=json.dumps(input_history, ensure_ascii=False), output_text=content)
        return content

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
        self.update_mid_vars(mid_vars, key="获取用户信息 01", input_text=prompt, output_text=content, model="Qwen1.5-14B-Chat")
        if sum([i in content for i in ["询问","提问","转移","未知","结束", "停止"]]) != 0:
            logger.debug('信息提取流程结束...')
            content = self.chatter_gaily(history, mid_vars)
            intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
        elif content:
            content = content.split('\n')[0].split('。')[0][:20]
        content = content if content else '未知'
        return {'end':True, 'message':content, 'intentCode':intentCode}

    def get_reminder_tips(self, prompt, history, intentCode, model='Baichuan2-7B-Chat', mid_vars=None):
        logger.debug('remind prompt: ' + prompt)
        model_output = callLLM(query=prompt, verbose=False, do_sample=False, temperature=0.1, top_p=0.2, max_tokens=500, model=model)
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
            yield make_meta_ret(msg=tool_name, type="Tool", code=intentCode), mid_vars
            yield make_meta_ret(msg=thought, type="Thought", code=intentCode), mid_vars

            if tool_name == '进一步询问用户的情况':
                out_text = make_meta_ret(end=True, msg=output_text, code=intentCode)
            elif tool_name == '直接回复用户问题':
                out_text = make_meta_ret(end=True,
                                         msg=output_text.split('Final Answer:')[-1].split('\n\n')[0].strip(),
                                         code=intentCode,
                                         init_intent=True)
            elif tool_name == '调用外部知识库':
                output_text = self.global_share_resource.chat_v2.funcall.call_search_knowledge(output_text)
                out_text = make_meta_ret(end=False, msg=output_text, code=intentCode)
            elif tool_name == '结束话题':
                out_text = make_meta_ret(end=True, msg=output_text, code=intentCode, init_intent=True)
            else:
                out_text = make_meta_ret(end=True, msg=output_text, code=intentCode)
                # logger.exception(out_history)
        yield out_text, mid_vars

    def intent_query(self, history, **kwargs):
        mid_vars = kwargs.get('mid_vars', [])
        task = kwargs.get('task', '')
        input_prompt = kwargs.get('prompt', [])
        scene_code = kwargs.get('scene_code')

        if task == 'verify' and input_prompt:
            intent, desc = get_intent(
                self.cls_intent_verify(history, mid_vars, input_prompt),
                self.global_share_resource.all_intent,
                self.global_share_resource.com_intent,
                scene_code=scene_code
            )
        else:
            intent, desc = get_intent(
                self.cls_intent(history, mid_vars, **kwargs),
                self.global_share_resource.all_intent,
                self.global_share_resource.com_intent,
                scene_code=scene_code
            )
        if self.intent_map['callout'].get(intent):
            out_text = {'message':get_doc_role(intent),
                        'intentCode':'doc_role', 'processCode':'trans_back',
                        'intentDesc':desc}
        elif self.intent_map['aiui'].get(intent):
            out_text = {'message':'', 'intentCode':intent, 'processCode':'aiui', 'intentDesc':desc}
        #elif intent in ['food_rec']:
        #    if not kwargs.get('userInfo', {}).get('askTastePrefer', ''):
        #        out_text = {'message':'', 'intentCode':intent,
        #             'processCode':'alg', 'intentDesc':desc}
        #    else:
        #        out_text = {'message':'', 'intentCode':'food_rec',
        #                'processCode':'alg', 'intentDesc':desc}
        #elif intent in ['sport_rec']:
        #    if kwargs.get('userInfo', {}).get('askExerciseHabbit', '') and kwargs.get('userInfo',{}).get('askExerciseTabooDegree', '') and kwargs.get('userInfo', {}).get('askExerciseTabooXt', ''):
        #        out_text = {'message':'',
        #                'intentCode':intent,'processCode':'alg', 'intentDesc':desc}
        #    else:
        #        out_text = {'message':'', 'intentCode':intent,
        #                'processCode':'trans_back', 'intentDesc':desc}
        elif intent in ['shared_decision', 'create_food_purchasing_list', 'digital_image_back', 'digital_image_switch']:
            out_text = {'message':'', 'intentCode':intent,
                    'processCode':'trans_back', 'intentDesc':desc}
        elif intent == 'remind_take_blood_pressure':
            out_text = {'message':'好的，已通知张叔叔测量血压', 'intentCode':intent,
                    'processCode':'trans_back', 'intentDesc':desc}
        # elif intent in ["route_rec", "spa_rec"]:
        #     # # 直接使用 history 中的内容生成 messages
        #     # messages = [
        #     #     {"role": "user", "content": history[0]["content"]}
        #     # ]
        #     # # 调用 generate_chat_response 方法生成响应
        #     # content = self.assistant.generate_chat_response(messages)
        #     out_text = {'message': '', 'intentCode': intent, 'processCode': 'alg', 'intentDesc': desc}
        elif intent in ["spa_knowledge"]:
            # 直接使用 history 中的内容生成 messages
            messages = [
                {"role": "user", "content": history[0]["content"]}
            ]
            # 调用 generate_chat_response 方法生成响应
            content = self.assistant.generate_chat_response(messages)
            out_text = {'message': content, 'intentCode': intent, 'processCode': 'alg', 'intentDesc': desc}
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

    def yield_result(self, *args, **kwargs):
        """处理最终的输出
        """
        _iterable = self.chat_gen(*args, **kwargs)
        while True:
            try:
                out_text, mid_vars = next(_iterable)
                if not out_text.get("init_intent"):
                    out_text['init_intent'] = False
                if not out_text.get("type"):
                    out_text['type'] = "Result"
                if not kwargs.get("ret_mid"):
                    logger.debug('输出为：' + json.dumps(out_text, ensure_ascii=False))
                    yield out_text
                else:
                    yield out_text, mid_vars
            except StopIteration as err:
                break
            except Exception as err:
                logger.exception(err)

    def __init_log__(self, *args, **kwargs):
        history = kwargs.get('history', [])
        mid_vars = kwargs['mid_vars']
        sys_prompt = kwargs.get('sys_prompt', '')
        intentCode = kwargs.get('intentCode', 'other')

        logger.debug(f'chat_gen输入的intentCode为: {intentCode}')
        if history:
            logger.debug(f"Last input: {history[-1]['content']}")
        return history, mid_vars, intentCode, sys_prompt

    def chat_gen(self, *args, **kwargs):
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

        history, mid_vars, intentCode, sys_prompt = self.__init_log__(*args, **kwargs)

        if self.intent_map['tips'].get(intentCode):
            desc = '日程提醒'
        elif self.intent_map['userinfo'].get(intentCode):
            desc = '智能标签'
        else:
            desc = intentCode_desc_map.get(intentCode, '日常对话')

        if self.intent_map['userinfo'].get(intentCode):
            out_text = self.get_userInfo_msg(sys_prompt, history, intentCode, mid_vars)
        elif self.intent_map['tips'].get(intentCode):
            out_text = self.get_reminder_tips(sys_prompt, history, intentCode, mid_vars=mid_vars)
        elif intentCode in ['BMI']:
            if not kwargs.get('userInfo', {}).get('askHeight', '') or not kwargs.get('userInfo', {}).get('askWeight', ''):
                out_text = {'end':True,'message':'','intentCode':'BMI'}
            else:
                output_text = self.chatter_gaily(*args, **kwargs)
                out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        elif intentCode in ['food_rec']:
            logger.debug('进入饮食推荐的闲聊...')
            output_text = self.chatter_gaily(*args, **kwargs)
            logger.debug('医师推荐闲聊的模型输出：' + output_text)
            out_text = {'end':True, 'message':output_text, 'intentCode':'other'}
        elif intentCode in ['sport_rec']:
            output_text = self.chatter_gaily(*args, **kwargs)
            out_text = {'end':True, 'message':output_text, 'intentCode':'other'}
        elif self.intent_map['schedule'].get(intentCode):
            messages = self.history_compose(history)
            _iterable = self.tsm._run(messages, **kwargs)
            while True:
                out_text, mid_vars_item = next(_iterable)
                if not out_text.get('end', False):
                    yield out_text, mid_vars
                else:
                    for item in mid_vars_item:
                        self.update_mid_vars(mid_vars, **item)
                    break
        elif intentCode == "auxiliary_diagnosis":
            _iterable = self.chat_auxiliary_diagnosis(**kwargs)
            while True:
                out_text, mid_vars = next(_iterable)
                if not out_text.get('end', False):
                    yield out_text, mid_vars
                else:
                    break
        elif intentCode == "open_Function":
            output_text = self.open_page(history, mid_vars)
            logger.debug('打开页面模型输出：'  + output_text)
            msg = '稍等片刻，页面即将打开' if self.get_pageName_code(output_text) != 'other' else output_text
            out_text = {'end':True, 'message':msg, 'intentCode':self.get_pageName_code(output_text)}
        else:
            output_text = self.chatter_gaily(*args, **kwargs)
            out_text = {'end':True, 'message':output_text, 'intentCode':intentCode}
        out_text['intentDesc'] = desc
        yield out_text, mid_vars

    def get_pageName_code(self, text):
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
        elif 'smart-home' in text and 'pagename' in text:
            return 'smart-home'
        else:
            return 'other'

if __name__ == '__main__':
    chat = Chat()
    ori_input_param = testParam.param_bug_20231124_1018
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


