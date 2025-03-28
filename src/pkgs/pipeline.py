# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-01 11:30:10
@desc    :   业务处理流程
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Generator, Iterator
from functools import lru_cache

sys.path.append('.')

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
    """
    聊天处理系统，实现多种对话模式和意图处理
    """

    def __init__(self, global_share_resource: initAllResource) -> None:
        """
        初始化聊天处理系统

        Args:
            global_share_resource: 全局共享资源
        """
        global_share_resource.chat_v2 = self
        self.gsr = global_share_resource
        self.prompt_meta_data = self.gsr.prompt_meta_data
        self.promptEngine = customPromptEngine(self.gsr)
        self.funcall = funcCall(self.gsr)
        self.sys_template = PromptTemplate(input_variables=['external_information'], template=TOOL_CHOOSE_PROMPT)
        self.__initialize_intent_map__()
        self.session = Session()

    def __initialize_intent_map__(self) -> None:
        """初始化各类意图映射表
        """
        # 将意图列表整合为映射表以便快速查询
        schedule_manager = ["schedule_qry_up", "schedule_manager"]
        tips_intent_code_list = [
            'dietary_eva', 'schedule_no', 'measure_bp',
            'meet_remind', 'medicine_remind', 'dietary_remind', 'sport_remind',
            'broadcast_bp', 'care_for', 'default_clock',
            'default_reminder', 'broadcast_bp_up'
        ]
        userinfo_intent_code_list = [
            'ask_name', 'ask_age', 'ask_exercise_taboo', 'sk_exercise_habbit', 'ask_food_alergy', 'ask_food_habbit',
            'ask_taste_prefer',
            'ask_family_history', 'ask_labor_intensity', 'ask_nation', 'ask_disease', 'ask_weight', 'ask_height',
            'ask_six', 'ask_mmol_drug', 'ask_exercise_taboo_degree', 'ask_exercise_taboo_xt'
        ]
        aiui_intent_code_list = ['websearch', 'KLLI3.captialInfo', 'lottery', 'dream', 'AIUI.calc', 'LEIQIAO.cityOfPro',
                                 'ZUOMX.queryCapital', 'calendar', 'audioProgram', 'translation', 'garbageClassifyPro',
                                 'AIUI.unitConversion', 'AIUI.forexPro', 'carNumber', 'datetimePro', 'AIUI.ocularGym',
                                 'weather', 'cookbook', 'story', 'AIUI.Bible', 'drama', 'storyTelling',
                                 'AIUI.audioBook', 'musicX', 'news', 'joke']
        callout_intent_code_list = ['call_doctor', 'call_sportMaster', 'call_psychologist', 'call_dietista',
                                    'call_health_manager']

        # 使用字典推导式简化代码
        self.intent_map = {
            'schedule': {i: 1 for i in schedule_manager},
            'tips': {i: 1 for i in tips_intent_code_list},
            'userinfo': {i: 1 for i in userinfo_intent_code_list},
            'aiui': {i: 1 for i in aiui_intent_code_list},
            'callout': {i: 1 for i in callout_intent_code_list}
        }

    def __get_default_reply__(self, intentCode: str) -> str:
        """
        针对不同的意图提供不同的回复指导话术

        Args:
            intentCode: 意图代码

        Returns:
            str: 默认回复内容
        """
        # 使用字典映射替代多个if-elif语句，提高可维护性
        default_replies = {
            "schedule_manager": "对不起，我没有理解您的需求，如果想进行日程提醒管理，您可以这样说: 查询一下我今天的日程, 提醒我明天下午3点去打羽毛球, 帮我把明天下午3点打羽毛球的日程改到后天下午5点, 取消今天的提醒",
            "other_schedule": "对不起，我没有理解您的需求，如果想进行日程提醒管理，您可以这样说: 查询一下我今天的日程, 提醒我明天下午3点去打羽毛球, 帮我把明天下午3点打羽毛球的日程改到后天下午5点, 取消今天的提醒",
            "schedule_qry_up": "对不起，我没有理解您的需求，如果您想查询今天的待办日程，您可以这样说：查询一下我今天的日程",
            "meeting_schedule": "对不起，我没有理解您的需求，如果您想管理会议日程，您可以这样说：帮我把明天下午4点的会议改到今天晚上7点",
            "auxiliary_diagnosis": "对不起，我没有理解您的需求，如果您有健康问题想要咨询，建议您提供更明确的描述"
        }

        return default_replies.get(intentCode, "对不起, 我没有理解您的需求, 请在问题中提供明确的信息并重新尝试.")

    def __check_query_valid__(self, query: str) -> bool:
        """
        检验生成的回复内容是否有效

        Args:
            query: 需要检验的内容

        Returns:
            bool: 内容是否有效
        """
        prompt = (
            "你是一个功能强大的内容校验工具请你帮我判断下面输入的句子是否符合要求\n"
            "1. 是一句完整的可以向用户输出的话\n"
            "2. 不包含特殊符号\n"
            "3. 语义完整连贯\n"
            "要判断的句子: {query}\n\n"
            "你的结果(yes or no):\n"
        )
        prompt = prompt.replace("{query}", query)

        try:
            result = callLLM(query=prompt, temperature=0, top_p=0, max_tokens=3)
            return "yes" in result.lower()
        except Exception as e:
            logger.error(f"内容校验失败: {e}")
            return False

    def __generate_content_verification__(self, out_text: List, list_of_plugin_info: List, **kwargs) -> List:
        """
        ReAct生成内容的校验

        1. 校验Tool
        2. 校验Tool Parameter格式

        Args:
            out_text: 包含思考、工具和参数的列表
            list_of_plugin_info: 可用工具列表

        Returns:
            List: 校验后的思考、工具和参数列表
        """
        thought, tool, parameter = out_text
        possible_tool_map = {i['code']: 1 for i in list_of_plugin_info}

        # 尝试解析参数为JSON
        if isinstance(parameter, str):
            try:
                parameter = json.loads(parameter)
            except json.JSONDecodeError:
                # 保持原始字符串格式，不做处理
                pass

        # 校验Tool
        if not possible_tool_map.get(tool):
            # 如果生成的Tool不对, parameter也必然不对
            tool = "AskHuman"
            parameter = self.__get_default_reply__(kwargs.get('intentCode', ''))

        # 处理AskHuman工具的特殊情况
        if tool == "AskHuman" and isinstance(parameter, dict):
            # 尝试提取dict中有效的回复内容
            valid_content_found = False
            for _, content in parameter.items():
                if isinstance(content, str) and self.__check_query_valid__(content):
                    parameter = content
                    valid_content_found = True
                    break

            if not valid_content_found:
                parameter = self.__get_default_reply__(kwargs.get('intentCode', ''))

        return [thought, tool, parameter]

    def chat_react(self, *args, **kwargs) -> List[Dict]:
        """
        调用模型生成答案,解析ReAct生成的结果

        Args:
            kwargs: 包含历史记录等参数

        Returns:
            List[Dict]: 更新后的历史记录
        """
        max_tokens = kwargs.get("max_tokens", 200)
        mid_vars = kwargs.get("mid_vars", [])

        # 1. 构建系统提示词和工具列表
        _sys_prompt, list_of_plugin_info = self.compose_input_history(**kwargs)

        # 2. 构建完整的提示词
        prompt = build_input_text(_sys_prompt, list_of_plugin_info, **kwargs)
        prompt += "Thought: "
        logger.debug(f"ReAct Prompt:\n{prompt}")

        # 3. 调用模型生成回复
        try:
            model_output = callLLM(
                prompt,
                temperature=0.7,
                top_p=0.5,
                max_tokens=max_tokens,
                model="Qwen-14B-Chat",
                stop=["\nObservation"]
            )
            model_output = "\nThought: " + model_output
            logger.debug(f"ReAct Generate: {model_output}")
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            # 生成一个安全的备用响应
            model_output = "\nThought: 我无法完成当前请求。\nAction: AskHuman\nAction Input: 抱歉，我现在无法处理您的请求，请稍后再试。"

        # 4. 更新中间变量
        self.update_mid_vars(mid_vars, key="Chat ReAct", input_text=prompt, output_text=model_output,
                             model="Qwen-14B-Chat")

        # 5. 解析模型输出
        out_text = parse_latest_plugin_call(model_output)

        # 6. 校验内容
        out_text = self.__generate_content_verification__(out_text, list_of_plugin_info, **kwargs)

        # 7. 处理工具参数
        try:
            tool = out_text[1]
            tool_zh = self.prompt_meta_data['prompt_tool_code_map'].get(tool)

            if tool_zh and tool_zh in self.prompt_meta_data['tool']:
                tool_param_msg = self.prompt_meta_data['tool'][tool_zh].get("params", [])

                # 对于只有一个参数的工具，尝试提取该参数
                if tool_param_msg and len(tool_param_msg) == 1:
                    param_name = tool_param_msg[0].get('name')
                    param_type = tool_param_msg[0].get('schema', {}).get('type', '')

                    if param_type.startswith("str") and isinstance(out_text[2], dict) and param_name in out_text[2]:
                        out_text[2] = out_text[2][param_name]
        except Exception as err:
            logger.debug(f"处理工具参数时出错: {err}")
            # 继续处理，不中断流程

        # 8. 更新历史记录
        kwargs['history'].append({
            "intentCode": kwargs.get('intentCode', ''),
            "role": "assistant",
            "content": out_text[0],
            "function_call": {"name": out_text[1], "arguments": out_text[2]}
        })

        return kwargs['history']

    def compose_input_history(self, **kwargs) -> Tuple[str, List]:
        """
        拼装系统提示词

        Args:
            kwargs: 包含提示词等参数

        Returns:
            Tuple[str, List]: 系统提示词和工具列表
        """
        qprompt = kwargs.get("qprompt")
        sys_prompt, functions = self.promptEngine._call(**kwargs)

        if not qprompt:
            sys_prompt = self.sys_template.format(external_information=sys_prompt)
        else:
            sys_prompt = sys_prompt + "\n\n" + qprompt

        return sys_prompt, functions

    def update_mid_vars(self, mid_vars: List, input_text=Any, output_text=Any, key="节点名", model="调用模型",
                        **kwargs) -> List:
        """
        更新中间变量

        Args:
            mid_vars: 中间变量列表
            input_text: 输入文本
            output_text: 输出文本
            key: 节点名称
            model: 模型名称

        Returns:
            List: 更新后的中间变量列表
        """
        if mid_vars is None:
            mid_vars = []

        lth = len(mid_vars) + 1
        mid_vars.append({
            "id": lth,
            "key": key,
            "input_text": input_text,
            "output_text": output_text,
            "model": model,
            **kwargs
        })
        return mid_vars

    def get_parent_intent_name(self, text: str) -> str:
        """
        根据文本获取父意图名称

        Args:
            text: 识别结果文本

        Returns:
            str: 父意图名称
        """
        # 使用包含关系映射，更容易维护
        intent_keywords = {
            '五师': '呼叫五师意图',
            '音频': '音频播放意图',
            '生活': '生活工具查询意图',
            '医疗': '医疗健康意图',
            '饮食': '饮食营养意图',
            '运动': '运动咨询意图',
            '日程': '日程管理意图',
            '食材采购': '食材采购意图'
        }

        for keyword, intent in intent_keywords.items():
            if keyword in text:
                return intent

        return '其它'

    def cls_intent(self, history: List[Dict], mid_vars: List, **kwargs) -> str:
        """
        意图识别

        Args:
            history: 历史记录
            mid_vars: 中间变量

        Returns:
            str: 识别的意图
        """
        # 快速检测特定功能页面
        # 可以通过关键词匹配进行快速意图识别
        feature_keywords = {
            '血压趋势图': '打开功能页面',
            '血压录入': '打开功能页面',
            '血压添加': '打开功能页面',
            '入录血压': '打开功能页面',
            '添加血压': '打开功能页面',
            '历史血压': '打开功能页面',
            '血压历史': '打开功能页面',
            '打开聊天': '打开功能页面',
            '打开交流': '打开功能页面',
            '信息交互页面': '打开功能页面'
        }

        # 检查列表组合
        open_page_combos = [
            (['打开', '日程'], 2),
            (['打开', '集市'], 2),
            (['打开', '家居'], 2)
        ]

        # 格式化历史记录
        history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in history]
        latest_msg = history[-1]['content']

        # 检查关键词
        for keyword in feature_keywords:
            if keyword in latest_msg:
                return '打开功能页面'

        # 检查组合关键词
        for words, threshold in open_page_combos:
            if sum(word in latest_msg for word in words) >= threshold:
                return '打开功能页面'

        # 准备模型输入
        h_p = "\n".join([("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history[-3:]])

        if kwargs.get('intentPrompt', ''):
            prompt = kwargs.get('intentPrompt') + "\n\n" + h_p + "\nThought: "
        else:
            prompt = self.prompt_meta_data['tool']['父意图']['description'] + "\n\n" + h_p + "\nThought: "

        logger.debug('父意图模型输入：' + prompt)

        try:
            # 调用模型进行意图识别
            generate_text = callLLM(
                query=prompt,
                max_tokens=200,
                top_p=0.8,
                temperature=0,
                do_sample=False
            )
            logger.debug('意图识别模型输出：' + generate_text)

            # 提取意图
            intentIdx = generate_text.find("\nIntent: ")
            if intentIdx >= 0:
                text = generate_text[intentIdx + 9:].split("\n")[0]
                parant_intent = self.get_parent_intent_name(text)
            else:
                # 如果找不到Intent标记，使用安全的默认值
                parant_intent = '其它'
                text = '未能识别意图'
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            parant_intent = '其它'
            text = '意图识别错误'
            generate_text = f"Error: {str(e)}"

        # 处理子意图识别
        specific_intents = ['呼叫五师意图', '音频播放意图', '生活工具查询意图', '医疗健康意图', '饮食营养意图',
                            '日程管理意图', '食材采购意图']
        if parant_intent in specific_intents:
            try:
                sub_intent_prompt = self.prompt_meta_data['tool'][parant_intent]['description']

                # 对于呼叫五师，只考虑最后一条消息
                if parant_intent == '呼叫五师意图':
                    history = history[-1:]
                    h_p = "\n".join(
                        [("Question" if i['role'] == "user" else "Answer") + f": {i['content']}" for i in history])

                if kwargs.get('subIntentPrompt', ''):
                    prompt = kwargs.get('subIntentPrompt').format(sub_intent_prompt) + "\n\n" + h_p + "\nThought: "
                else:
                    prompt = self.prompt_meta_data['tool']['子意图模版']['description'].format(
                        sub_intent_prompt) + "\n\n" + h_p + "\nThought: "

                logger.debug('子意图模型输入：' + prompt)

                # 调用模型识别子意图
                generate_text = callLLM(
                    query=prompt,
                    max_tokens=200,
                    top_p=0.8,
                    temperature=0,
                    do_sample=False
                )

                intentIdx = generate_text.find("\nIntent: ")
                if intentIdx >= 0:
                    text = generate_text[intentIdx + 9:].split("\n")[0]
            except Exception as e:
                logger.error(f"子意图识别失败: {e}")
                # 保持原始父意图结果

        # 更新中间变量
        self.update_mid_vars(mid_vars, key="意图识别", input_text=prompt, output_text=generate_text, intent=text)
        return text

    def chatter_gaily(self, mid_vars: List, **kwargs) -> Any:
        """
        组装mysql中闲聊对应的prompt

        Args:
            mid_vars: 中间变量

        Returns:
            str 或 List[Dict]: 输出内容或更新的历史记录
        """

        def compose_func_reply(messages: List[Dict]) -> List[Dict]:
            """
            拼接func中回复的内容到history中

            最终的history只有role/content字段

            Args:
                messages: 消息列表

            Returns:
                List[Dict]: 格式化后的历史记录
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

        try:
            # 获取当前意图的消息
            intentCode = kwargs.get("intentCode", 'other')
            messages = [i for i in kwargs['history'] if i.get("intentCode") == intentCode]

            # 添加系统提示词
            desc = self.prompt_meta_data['event'][intentCode].get('description', '')
            process = self.prompt_meta_data['event'][intentCode].get('process', '')
            if desc or process:
                ext_info = desc + "\n" + process
                messages = [{"role": "system", "content": ext_info}] + messages

            # 处理函数调用
            messages = compose_func_reply(messages)

            # 调用模型生成回复
            content = callLLM("", messages, temperature=0.7, top_p=0.8)

            # 更新中间变量
            self.update_mid_vars(mid_vars, key="闲聊", input_text=json.dumps(messages, ensure_ascii=False),
                                 output_text=content)

            # 返回结果
            if kwargs.get("return_his"):
                messages.append({
                    "role": "assistant",
                    "content": "I know the answer.",
                    "function_call": {"name": "convComplete", "arguments": content}
                })
                return messages
            else:
                return content
        except Exception as e:
            logger.error(f"闲聊处理失败: {e}")
            error_content = "很抱歉，我现在无法回答您的问题，请稍后再试。"

            # 更新中间变量
            self.update_mid_vars(mid_vars, key="闲聊-错误", input_text=str(kwargs.get('history', [])),
                                 output_text=f"Error: {str(e)}")

            # 返回安全的结果
            if kwargs.get("return_his"):
                return [{"role": "assistant", "content": "I know the answer.",
                         "function_call": {"name": "convComplete", "arguments": error_content}}]
            else:
                return error_content

    def chatter_gaily_knowledge(self, mid_vars: List, **kwargs) -> Any:
        """
        组装mysql中闲聊对应的prompt，并结合知识库

        Args:
            mid_vars: 中间变量

        Returns:
            str 或 List[Dict]: 输出内容或更新的历史记录
        """

        def compose_func_reply(messages: List[Dict]) -> Dict:
            """
            拼接func中回复的内容到history中，并构建知识库查询载荷

            Args:
                messages: 消息列表

            Returns:
                Dict: 知识库查询载荷
            """
            payload = {
                "query": "",
                "knowledge_base_name": "新奥百科知识库",
                "top_k": 5,
                "score_threshold": 1,
                "history": [],
                "stream": False,
                "model_name": "Qwen-14B-Chat",
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 0,
                "prompt_name": "text"
            }

            history = []
            for i in messages:
                if not i.get("function_call"):
                    history.append(i)
                else:
                    func_args = i['function_call']
                    role = i['role']
                    content = f"{func_args['arguments']}"
                    history.append({"role": role, "content": content})

            # 设置查询参数
            if history:
                payload['history'] = history[:-1]
                payload['query'] = history[-1]['content']

            return payload

        try:
            url = self.gsr.api_config['langchain'] + '/chat/knowledge_base_chat'
            intentCode = kwargs.get("intentCode", 'other')

            # 获取当前意图的消息
            messages = [i for i in kwargs['history'] if i.get("intentCode") == intentCode]

            # 添加系统提示词
            desc = self.prompt_meta_data['event'][intentCode].get('description', '')
            process = self.prompt_meta_data['event'][intentCode].get('process', '')
            if desc or process:
                ext_info = desc + "\n" + process
                messages = [{"role": "system", "content": ext_info}] + messages

            # 构建知识库查询载荷
            payload = compose_func_reply(messages)

            # 调用知识库API
            response = self.session.post(url, data=json.dumps(payload))

            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"知识库API调用失败: {response.status_code}, {response.text}")
                raise Exception(f"知识库API调用失败: {response.status_code}")

            # 解析响应
            response_data = response.json()
            content = response_data.get('answer', '抱歉，无法获取知识库回答')
            docs = response_data.get('docs', [])

            # 更新中间变量
            self.update_mid_vars(
                mid_vars,
                key="闲聊-知识库-新奥百科",
                input_text=payload,
                output_text=response_data,
                model="知识库-新奥百科知识库-Qwen-14B-Chat"
            )

            # 返回结果
            if kwargs.get("return_his"):
                messages.append({
                    "role": "assistant",
                    "content": "I know the answer.",
                    "function_call": {"name": "convComplete", "arguments": content}
                })
                return messages[1:]  # 跳过system消息
            else:
                return content
        except Exception as e:
            logger.error(f"知识库查询失败: {e}")
            error_content = "很抱歉，我现在无法从知识库获取相关信息，请稍后再试。"

            # 更新中间变量
            self.update_mid_vars(mid_vars, key="闲聊-知识库-错误", input_text=str(kwargs.get('history', [])),
                                 output_text=f"Error: {str(e)}")

            # 返回安全的结果
            if kwargs.get("return_his"):
                return [{"role": "assistant", "content": "I know the answer.",
                         "function_call": {"name": "convComplete", "arguments": error_content}}]
            else:
                return error_content

    def intent_query(self, history: List[Dict], **kwargs) -> Dict:
        """
        意图查询处理

        Args:
            history: 历史记录

        Returns:
            Dict: 意图查询结果
        """
        mid_vars = kwargs.get('mid_vars', [])
        task = kwargs.get('task', '')
        input_prompt = kwargs.get('prompt', [])

        try:
            # 根据任务类型选择意图识别方法
            if task == 'verify' and input_prompt:
                # 验证模式下的意图识别
                intent, desc = get_intent(self.cls_intent_verify(history, mid_vars, input_prompt))
            else:
                # 常规意图识别
                intent, desc = get_intent(self.cls_intent(history, mid_vars, **kwargs))

            # 根据意图类型处理输出
            if self.intent_map['callout'].get(intent):
                out_text = {
                    'message': get_doc_role(intent),
                    'intentCode': 'doc_role',
                    'processCode': 'trans_back',
                    'intentDesc': desc
                }
            elif self.intent_map['aiui'].get(intent):
                out_text = {
                    'message': '',
                    'intentCode': intent,
                    'processCode': 'aiui',
                    'intentDesc': desc
                }
            elif intent in ['food_rec']:
                if not kwargs.get('userInfo', {}).get('askTastePrefer', ''):
                    out_text = {
                        'message': '',
                        'intentCode': intent,
                        'processCode': 'trans_back',
                        'intentDesc': desc
                    }
                else:
                    out_text = {
                        'message': '',
                        'intentCode': 'food_rec',
                        'processCode': 'alg',
                        'intentDesc': desc
                    }
            else:
                out_text = {
                    'message': '',
                    'intentCode': intent,
                    'processCode': 'alg',
                    'intentDesc': desc
                }

            logger.debug('意图识别输出：' + json.dumps(out_text, ensure_ascii=False))
            return out_text
        except Exception as e:
            logger.error(f"意图查询失败: {e}")
            # 返回安全的默认意图
            default_intent = {
                'message': '',
                'intentCode': 'other',
                'processCode': 'alg',
                'intentDesc': '通用对话'
            }
            return default_intent

    def fetch_intent_code(self) -> Dict[str, List[str]]:
        """
        返回所有的intentCode

        Returns:
            Dict[str, List[str]]: 意图代码映射
        """
        intent_code_map = {
            "get_userInfo_msg": list(self.intent_map['userinfo'].keys()),
            "get_reminder_tips": list(self.intent_map['tips'].keys()),
            "other": ['BMI', 'food_rec', 'sport_rec', 'schedule_manager', 'schedule_qry_up', 'auxiliary_diagnosis',
                      "other"]
        }
        return intent_code_map

    def pre_fill_param(self, *args, **kwargs) -> Tuple[tuple, dict]:
        """
        结合业务逻辑，预构建输入

        Args:
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            Tuple[tuple, dict]: 更新后的参数
        """
        intentCode = kwargs.get("intentCode")

        # 检查意图是否存在，不存在则默认为other
        if not self.prompt_meta_data['event'].get(intentCode):
            logger.debug(f"不支持当前事件 {intentCode}，将intentCode更改为other。")
            kwargs['intentCode'] = 'other'
            intentCode = 'other'

        # 处理特定意图的特殊需求
        if intentCode == "schedule_qry_up" and not kwargs.get("history"):
            kwargs['history'] = [{"role": 0, "content": "帮我查询今天的日程"}]

        # 日程相关意图需要获取日程信息
        if "schedule" in intentCode:
            try:
                kwargs['schedule'] = self.funcall.call_get_schedule(*args, **kwargs)
            except Exception as e:
                logger.error(f"获取日程失败: {e}")
                kwargs['schedule'] = []  # 使用空列表作为默认值

        return args, kwargs

    def general_yield_result(self, *args, **kwargs) -> Generator[Dict, None, None]:
        """
        处理最终的输出

        Args:
            args: 位置参数
            kwargs: 关键字参数

        Yields:
            Dict: 输出结果
        """
        try:
            # 预处理参数
            args, kwargs = self.pre_fill_param(*args, **kwargs)

            # 处理历史记录
            if kwargs.get("history"):
                history = [{**i, "role": role_map.get(str(i['role']), "user")} for i in kwargs['history']]
                kwargs['history'] = kwargs.get('backend_history', []) + [history[-1]]
                kwargs['history'][-1]['intentCode'] = kwargs['intentCode']

            # 对于其他意图，清除提示词
            if kwargs['intentCode'] == 'other':
                kwargs['prompt'] = None
                kwargs['sys_prompt'] = None

            # 执行处理流程
            _iterable = self.pipeline(*args, **kwargs)

            # 处理每个生成的结果
            while True:
                try:
                    yield_item = next(_iterable)

                    # 确保结果包含类型
                    if not yield_item['data'].get("type"):
                        yield_item['data']['type'] = "Result"

                    # 确保结果包含数据源
                    if yield_item['data']['type'] == "Result" and not yield_item['data'].get("dataSource"):
                        yield_item['data']['dataSource'] = DEFAULT_DATA_SOURCE

                    yield yield_item
                except StopIteration:
                    break
        except Exception as e:
            logger.error(f"生成结果处理失败: {e}")
            # 返回错误信息
            error_data = {
                "data": {
                    "type": "Result",
                    "dataSource": DEFAULT_DATA_SOURCE,
                    "message": "抱歉，处理您的请求时出现了问题，请稍后再试。",
                    "end": True
                },
                "mid_vars": kwargs.get('mid_vars', []),
                "history": kwargs.get('history', [])
            }
            yield error_data

    def __log_init(self, **kwargs) -> None:
        """
        初始打印日志

        Args:
            kwargs: 包含意图和历史的参数
        """
        intentCode = kwargs.get('intentCode', 'unknown')
        history = kwargs.get('history', [])

        logger.info(f'intentCode: {intentCode}')
        if history:
            logger.info(f"Input: {history[-1].get('content', '')}")

    def parse_last_history(self, history: List[Dict]) -> Tuple[str, Any, str]:
        """
        解析历史记录中的最后一条

        Args:
            history: 历史记录

        Returns:
            Tuple[str, Any, str]: 工具、内容和思考过程
        """
        if not history or len(history) == 0:
            return "AskHuman", "没有可用的历史记录", "无思考过程"

        last_record = history[-1]

        if not last_record.get('function_call'):
            return "AskHuman", last_record.get('content', ""), "无思考过程"

        tool = last_record['function_call'].get('name', "AskHuman")
        content = last_record['function_call'].get('arguments', "")
        thought = last_record.get('content', "")

        logger.debug(f"Action: {tool}")
        logger.debug(f"Thought: {thought}")
        logger.debug(f"Action Input: {content}")

        return tool, content, thought

    def get_userInfo_msg(self, prompt: str, history: List[Dict], intentCode: str, mid_vars: List) -> Tuple[str, str]:
        """
        获取用户信息

        Args:
            prompt: 提示词
            history: 历史记录
            intentCode: 意图代码
            mid_vars: 中间变量

        Returns:
            Tuple[str, str]: 内容和意图代码
        """
        logger.debug(f'信息提取prompt:\n{prompt}')

        try:
            # 调用模型提取信息
            model_output = callLLM(
                prompt,
                verbose=False,
                temperature=0,
                top_p=0.8,
                max_tokens=200,
                do_sample=False
            )
            logger.debug('信息提取模型输出：' + model_output)
            content = model_output

            # 更新中间变量
            self.update_mid_vars(mid_vars, key="获取用户信息 01", input_text=prompt, output_text=content,
                                 model="Qwen-14B-Chat")

            # 处理提取结果
            exit_keywords = ["询问", "提问", "转移", "结束", "未知", "停止"]
            if any(keyword in content for keyword in exit_keywords):
                logger.debug('信息提取流程结束...')
                content = self.chatter_gaily(mid_vars, history=history)
                intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
            elif content:
                # 只取第一行第一句，并限制长度
                content = content.split('\n')[0].split('。')[0][:20]
                logger.debug('标签归一前提取内容：' + content)
                content = norm_userInfo_msg(intentCode, content)
                logger.debug('标签归一后提取内容：' + content)

            # 处理空内容和错误
            content = content if content else '未知'
            content = '未知' if 'Error' in content else content

            return content, intentCode
        except Exception as e:
            logger.error(f"用户信息提取失败: {e}")
            return '未知', intentCode

    def get_reminder_tips(self, prompt: str, history: List[Dict], intentCode: str, model='Baichuan2-7B-Chat',
                          mid_vars=None) -> Tuple[str, str]:
        """
        获取提醒提示

        Args:
            prompt: 提示词
            history: 历史记录
            intentCode: 意图代码
            model: 模型名称
            mid_vars: 中间变量

        Returns:
            Tuple[str, str]: 内容和意图代码
        """
        logger.debug('remind prompt: ' + prompt)

        try:
            # 调用模型生成提醒内容
            content = callLLM(
                query=prompt,
                verbose=False,
                do_sample=False,
                temperature=0.1,
                top_p=0.2,
                max_tokens=500,
                model=model
            )

            # 更新中间变量
            if mid_vars is not None:
                self.update_mid_vars(mid_vars, key="提醒生成", input_text=prompt, output_text=content, model=model)

            logger.debug('remind model output: ' + content)

            # 清理输出中的特殊前缀
            if content.startswith('（）'):
                content = content[2:].strip()

            return content, intentCode
        except Exception as e:
            logger.error(f"提醒生成失败: {e}")
            return "抱歉，我现在无法为您提供提醒。", intentCode

    def open_page(self, mid_vars: List, **kwargs) -> str:
        """
        组装打开页面对应的prompt

        Args:
            mid_vars: 中间变量

        Returns:
            str: 页面名称
        """
        # 关键词映射列表
        keyword_to_page = {
            "血压趋势": 'pagename:"bloodPressure-trend-chart"',
            "血压录入": 'pagename:"add-blood-pressure"',
            "血压添加": 'pagename:"add-blood-pressure"',
            "录入血压": 'pagename:"add-blood-pressure"',
            "添加血压": 'pagename:"add-blood-pressure"',
            "交流": 'pagename:"interactive-information"',
            "聊天": 'pagename:"interactive-information"',
            "信息交互": 'pagename:"interactive-information"',
            "设置": 'pagename:"personal-setting"',
            "二维码": 'pagename:"qr-code"',
            "血压历史": 'pagename:"record-list3"',
            "历史血压": 'pagename:"record-list3"',
            "打开记录": 'pagename:"add-diet"',
            "打开录入": 'pagename:"add-diet"',
            "打开添加": 'pagename:"add-diet"',
            "饮食记录": 'pagename:"diet-record"',
            "饮食添加": 'pagename:"diet-record"',
            "打开推荐": 'pagename:"diet-record"',
            "饮食评估": 'pagename:"diet-record"',
            "食谱": 'pagename:"diet-record"',
            "我的饮食": 'pagename:"diet-record"',
            "食谱页面": 'pagename:"diet-record"',
            "餐食记录": 'pagename:"diet-record"',
            "集市": 'pagename:"my-market"'
        }

        try:
            # 格式化历史记录
            input_history = [{"role": role_map.get(str(i['role']), "user"), "content": i['content']} for i in
                             kwargs['history']]
            input_history = input_history[-3:]  # 只取最近3条记录
            latest_msg = input_history[-1]['content']

            # 检查特定关键词组合
            if "打开" in latest_msg and "日程" in latest_msg:
                return 'pagename:"my-schedule"'

            # 检查其他关键词
            for keyword, page in keyword_to_page.items():
                if keyword in latest_msg:
                    return page

            # 构建更复杂的提示，让模型判断要打开的页面
            hp = [h['role'] + ' ' + h['content'] for h in input_history]
            hi = ''

            if len(input_history) > 1:
                hi = '用户历史会话如下，可以作为意图识别的参考，但不要过于依赖历史记录，因为它可能是很久以前的记录：' + '\n' + '\n'.join(
                    [h["role"] + h["content"] for h in input_history[-3:-1]]) + '\n' + '当前用户输入：\n'

            hi += f'Question:{input_history[-1]["content"]}\nThought:'
            ext_info = self.prompt_meta_data['event']['open_Function']['description'] + "\n" + \
                       self.prompt_meta_data['event']['open_Function']['process'] + '\n' + hi + '\nThought:'

            input_history = [{"role": "system", "content": ext_info}]
            logger.debug('打开页面模型输入：' + json.dumps(input_history, ensure_ascii=False))

            # 调用模型获取页面
            content = callLLM("", input_history, temperature=0, top_p=0.8, do_sample=False)

            # 解析模型输出
            if 'Answer' in content:
                content = content[content.find('Answer') + 7:].split('\n')[0].strip()
            elif 'Output' in content:
                content = content[content.find('Response') + 6:].split('\n')[0].strip()
            elif 'Response' in content:
                content = content[content.find('') + 9:].split('\n')[0].strip()

            # 更新中间变量
            self.update_mid_vars(mid_vars, key="打开功能画面", input_text=json.dumps(input_history, ensure_ascii=False),
                                 output_text=content)

            return content
        except Exception as e:
            logger.error(f"打开页面处理失败: {e}")
            return "pagename:\"unknown\""

    def get_pageName_code(self, text: str) -> str:
        """
        从文本中提取页面名称代码

        Args:
            text: 包含页面信息的文本

        Returns:
            str: 页面代码
        """
        logger.debug('页面生成内容：' + text)

        # 页面名称映射
        page_mappings = [
            ('bloodPressure-trend-chart', 'bloodPressure-trend-chart'),
            ('add-blood-pressure', 'add-blood-pressure'),
            ('record-list3', 'record-list3'),
            ('my-schedule', 'my-schedule'),
            ('add-diet', 'add-diet'),
            ('diet-record', 'diet-record'),
            ('my-market', 'my-market'),
            ('personal-setting', 'personal-setting'),
            ('qr-code', 'qr-code'),
            ('smart-home', 'smart-home'),
            ('interactive-information', 'interactive-information')
        ]

        # 检查文本中是否包含页面名称
        for page_id, page_code in page_mappings:
            if page_id in text and 'pagename' in text:
                return page_code

        return 'other'

    def complete(self, mid_vars: List[object], **kwargs) -> Tuple[List[Dict], str]:
        """
        only prompt模式的生成及相关逻辑

        Args:
            mid_vars: 中间变量

        Returns:
            Tuple[List[Dict], str]: 更新的历史记录和意图代码
        """
        # 验证必要参数
        if not kwargs.get("prompt"):
            raise ValueError("当前处理类型为only_prompt，但未提供prompt参数。")

        prompt = kwargs['prompt']
        chat_history = kwargs['history']
        intentCode = kwargs['intentCode']

        try:
            # 根据意图类型选择处理方法
            if self.intent_map['userinfo'].get(intentCode):
                # 用户信息提取
                content, intentCode = self.get_userInfo_msg(prompt, chat_history, intentCode, mid_vars)
            elif self.intent_map['tips'].get(intentCode):
                # 提醒生成
                content, intentCode = self.get_reminder_tips(prompt, chat_history, intentCode, mid_vars=mid_vars)
            elif intentCode == "open_Function":
                # 打开功能页面
                output_text = self.open_page(mid_vars, **kwargs)
                page_code = self.get_pageName_code(output_text)
                content = '稍等片刻，页面即将打开' if page_code != 'other' else output_text
                intentCode = page_code
                logger.debug('页面Code: ' + intentCode)
            else:
                # 默认闲聊
                content = self.chatter_gaily(mid_vars, return_his=False, **kwargs)

            # 验证返回类型
            if not isinstance(content, str):
                logger.warning(f"only_prompt模式下，返回值必须为str类型，但获得了{type(content)}")
                content = str(content)

            # 更新历史记录
            chat_history.append({
                "role": "assistant",
                "content": "I know the answer.",
                "function_call": {"name": "convComplete", "arguments": content}
            })

            return chat_history, intentCode
        except Exception as e:
            logger.error(f"完成处理失败: {e}")
            # 生成安全的响应
            error_content = "抱歉，我暂时无法处理您的请求，请稍后再试。"
            chat_history.append({
                "role": "assistant",
                "content": "I know the answer.",
                "function_call": {"name": "convComplete", "arguments": error_content}
            })
            return chat_history, intentCode

    def interact_first(self, mid_vars: List, **kwargs) -> Tuple[List[Dict], str]:
        """
        首次交互处理

        Args:
            mid_vars: 中间变量

        Returns:
            Tuple[List[Dict], str]: 历史记录和意图代码
        """
        intentCode = kwargs.get('intentCode', '')
        out_history = None

        try:
            # 检查事件是否存在于prompt_meta_data中
            if self.prompt_meta_data['event'].get(intentCode):
                if intentCode == "other":
                    # 闲聊
                    out_history = self.chatter_gaily(mid_vars, **kwargs, return_his=True)
                elif intentCode == "enn_wiki":
                    # 知识库查询
                    out_history = self.chatter_gaily_knowledge(mid_vars, **kwargs, return_his=True)
                elif self.prompt_meta_data['event'][intentCode].get("process_type") == "only_prompt":
                    # 仅提示模式
                    out_history, intentCode = self.complete(mid_vars=mid_vars, **kwargs)
                    kwargs['intentCode'] = intentCode
                elif self.prompt_meta_data['event'][intentCode].get("process_type") == "react":
                    # ReAct模式
                    out_history = self.chat_react(mid_vars=mid_vars, **kwargs)

            # 如果没有处理结果，使用chat_react作为备选方案
            if not out_history:
                out_history = self.chat_react(mid_vars=mid_vars, return_his=True, max_tokens=100, **kwargs)

            return out_history, intentCode
        except Exception as e:
            logger.error(f"首次交互失败: {e}")
            # 生成安全的响应
            error_message = "抱歉，处理您的请求时出现了问题，请稍后再试。"
            safe_history = [{"role": "assistant", "content": "Error occurred.",
                             "function_call": {"name": "convComplete", "arguments": error_message}}]
            return safe_history, intentCode

    def if_init(self, tool: str) -> bool:
        """
        检查工具是否为初始化意图

        Args:
            tool: 工具名称

        Returns:
            bool: 是否为初始化意图
        """
        return self.prompt_meta_data['init_intent'].get(tool, False)

    def pipeline(self, mid_vars: List = [], **kwargs) -> Iterator[Dict]:
        """
        多轮交互流程

        1. 定义先验信息变量,拼装对应prompt
        2. 准备模型输入messages
        3. 模型生成结果

        Args:
            mid_vars: 中间变量列表
            kwargs: 包含历史记录和意图代码等参数

        Yields:
            Dict: 输出结果
        """
        # 初始化日志
        self.__log_init(**kwargs)

        # 获取意图代码和中间变量
        intentCode = kwargs.get('intentCode', '')
        mid_vars = kwargs.get('mid_vars', [])
        dataSource = DEFAULT_DATA_SOURCE

        try:
            # 首次交互
            out_history, intentCode = self.interact_first(mid_vars=mid_vars, **kwargs)

            # 循环处理工具调用
            while True:
                # 解析历史记录中的最后一条
                tool, content, thought = self.parse_last_history(out_history)

                # 输出思考过程和工具选择
                if self.prompt_meta_data['event'].get(intentCode) and self.prompt_meta_data['event'][intentCode].get(
                        'process_type') != "only_prompt":
                    ret_tool = make_meta_ret(msg=tool, type="Tool", code=intentCode, gsr=self.gsr)
                    ret_thought = make_meta_ret(msg=thought, type="Thought", code=intentCode, gsr=self.gsr)
                    yield {"data": ret_tool, "mid_vars": mid_vars, "history": out_history}
                    yield {"data": ret_thought, "mid_vars": mid_vars, "history": out_history}

                # 检查是否需要执行工具函数
                if self.prompt_meta_data['rollout_tool'].get(tool) or not self.funcall.funcmap.get(tool):
                    # 不需要执行工具或工具不存在，直接退出循环
                    break

                # 执行工具函数
                try:
                    kwargs['history'], dataSource = self.funcall._call(out_history=out_history, mid_vars=mid_vars,
                                                                       **kwargs)
                except AssertionError as err:
                    logger.error(f"工具函数执行失败: {err}")
                    # 重试工具函数
                    kwargs['history'], dataSource = self.funcall._call(out_history=out_history, mid_vars=mid_vars,
                                                                       **kwargs)
                except Exception as e:
                    logger.error(f"工具函数执行出现未预期的错误: {e}")
                    # 中断处理并返回错误信息
                    dataSource = DEFAULT_DATA_SOURCE
                    content = "抱歉，处理您的请求时出现了问题，请稍后再试。"
                    break

                # 检查是否在工具完成后直接输出
                if self.prompt_meta_data['rollout_tool_after_complete'].get(tool):
                    # 工具执行完成后输出
                    content = kwargs['history'][-1]['content']
                    break
                else:
                    # 输出函数调用结果，继续ReAct流程
                    content = kwargs['history'][-1]['content']
                    ret_function_call = make_meta_ret(msg=content, type="Observation", code=intentCode, gsr=self.gsr)
                    yield {"data": ret_function_call, "mid_vars": mid_vars, "history": out_history}
                    out_history = self.chat_react(mid_vars=mid_vars, **kwargs)

            # 生成最终结果
            ret_result = make_meta_ret(
                end=True,
                msg=content,
                code=intentCode,
                gsr=self.gsr,
                init_intent=self.if_init(tool),
                dataSource=dataSource
            )
            yield {"data": ret_result, "mid_vars": mid_vars, "history": out_history}

        except Exception as e:
            logger.error(f"Pipeline处理失败: {e}")
            # 生成错误结果
            error_result = make_meta_ret(
                end=True,
                msg="抱歉，处理您的请求时遇到了问题，请稍后再试。",
                code=intentCode,
                gsr=self.gsr,
                init_intent=False,
                dataSource=DEFAULT_DATA_SOURCE
            )
            yield {"data": error_result, "mid_vars": mid_vars,
                   "history": out_history if 'out_history' in locals() else []}


if __name__ == '__main__':
    # 初始化聊天实例并测试
    chat = Chat_v2(initAllResource())
    ori_input_param = testParam.param_feat_schedular_not_today
    prompt = ori_input_param['prompt']
    history = ori_input_param['history']
    intentCode = ori_input_param['intentCode']
    customId = ori_input_param['customId']
    orgCode = ori_input_param['orgCode']

    # 测试生成结果
    for result in chat.general_yield_result(**ori_input_param):
        # 在这里处理生成的结果
        pass