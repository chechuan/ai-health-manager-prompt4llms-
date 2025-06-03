# -*- encoding: utf-8 -*-
"""
@Time    :   2023-11-01 11:30:10
@desc    :   业务处理流程
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import sys
from random import choice

from src.pkgs.models.agents import Agents
from src.utils.api_protocal import AigcFunctionsDoctorRecommendRequest

sys.path.append(".")
import json
from typing import Any, AsyncGenerator, Dict, List

from langchain.prompts import PromptTemplate
from requests import Session

from chat.constant import (
    EXT_USRINFO_TRANSFER_INTENTCODE,
    default_prompt,
    intentCode_desc_map,
)
from chat.util import norm_userInfo_msg
from data.constrant import DEFAULT_DATA_SOURCE
from data.constrant import TOOL_CHOOSE_PROMPT_PIPELINE as TOOL_CHOOSE_PROMPT
from data.constrant import role_map
from data.test_param.test import testParam
from src.pkgs.knowledge.callback import FuncCall
from src.pkgs.models.custom_chat_model import (
    CustomChatAuxiliary,
    CustomChatModel,
    CustomChatReportInterpretationAnswer,
    CustomChatReportInterpretationAsk,
)
from src.prompt.factory import CustomPromptEngine
from src.prompt.model_init import callLLM
from src.prompt.react_demo import build_input_text
from src.prompt.utils import ChatterGailyAssistant
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    curr_time,
    date_after,
    get_doc_role,
    get_intent,
    make_meta_ret,
    parse_latest_plugin_call,
    detect_sensitive_all
)
from data.sensitive_data.regex_patterns import REGEX_PATTERNS
from src.pkgs.models.health_expert_model import HealthExpertModel


class Chat_v2:
    def __init__(self, global_share_resource: InitAllResource) -> None:
        global_share_resource.chat_v2 = self
        self.gsr = global_share_resource
        self.prompt_meta_data = self.gsr.prompt_meta_data
        self.promptEngine = CustomPromptEngine(self.gsr)
        self.funcall = FuncCall(self.gsr)
        self.sys_template = PromptTemplate(
            input_variables=["external_information"], template=TOOL_CHOOSE_PROMPT
        )
        self.custom_chat_auxiliary = CustomChatAuxiliary(self.gsr)
        self.custom_chat_model = CustomChatModel(self.gsr)
        self.custom_chat_report_interpretation_ask = CustomChatReportInterpretationAsk(
            self.gsr
        )
        self.custom_chat_report_interpretation_answer = (
            CustomChatReportInterpretationAnswer(self.gsr)
        )
        self.chatter_assistant = ChatterGailyAssistant(global_share_resource)
        self.__initalize_intent_map__()
        self.session = Session()

    def __initalize_intent_map__(self):
        """初始化各类意图map"""
        schedule_manager = ["schedule_qry_up", "schedule_manager"]
        tips_intent_code_list = [
            "dietary_eva",
            "schedule_no",
            "measure_bp",
            "meet_remind",
            "medicine_remind",
            "dietary_remind",
            "sport_remind",
            "broadcast_bp",
            "care_for",
            "default_clock",
            "default_reminder",
            "broadcast_bp_up",
        ]
        useinfo_intent_code_list = [
            "ask_name",
            "ask_age",
            "ask_exercise_taboo",
            "ask_exercise_habbit",
            "ask_food_alergy",
            "ask_food_habbit",
            "ask_taste_prefer",
            "ask_family_history",
            "ask_labor_intensity",
            "ask_nation",
            "ask_disease",
            "ask_weight",
            "ask_height",
            "ask_six",
            "ask_mmol_drug",
            "ask_exercise_taboo_degree",
            "ask_exercise_taboo_xt",
            "ask_goal_manage",
        ]
        aiui_intent_code_list = [
            "websearch",
            "KLLI3.captialInfo",
            "lottery",
            "dream",
            "AIUI.calc",
            "LEIQIAO.cityOfPro",
            "ZUOMX.queryCapital",
            "calendar",
            "audioProgram",
            "translation",
            "garbageClassifyPro",
            "AIUI.unitConversion",
            "AIUI.forexPro",
            "carNumber",
            "datetimePro",
            "AIUI.ocularGym",
            "weather",
            "cookbook",
            "story",
            "AIUI.Bible",
            "drama",
            "storyTelling",
            "AIUI.audioBook",
            "musicX",
            "news",
            "joke",
        ]
        callout_intent_code_list = [
            "call_doctor",
            "call_sportMaster",
            "call_psychologist",
            "call_dietista",
            "call_health_manager",
        ]
        self.intent_map = {
            "schedule": {i: 1 for i in schedule_manager},
            "tips": {i: 1 for i in tips_intent_code_list},
            "userinfo": {i: 1 for i in useinfo_intent_code_list},
            "aiui": {i: 1 for i in aiui_intent_code_list},
            "callout": {i: 1 for i in callout_intent_code_list},
        }

    def __get_default_reply__(self, intentCode):
        """针对不同的意图提供不同的回复指导话术"""
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

    def __generate_content_verification__(
        self, out_text, list_of_plugin_info, **kwargs
    ):
        """ReAct生成内容的校验

        1. 校验Tool
        2. 校验Tool Parameter格式
        """
        thought, tool, parameter = out_text
        possible_tool_map = {i["code"]: 1 for i in list_of_plugin_info}

        try:
            parameter = json.loads(parameter)
        except Exception as err:
            ...

        # 校验Tool
        if not possible_tool_map.get(tool):  # 如果生成的Tool不对, parameter也必然不对
            tool = "AskHuman"
            parameter = self.__get_default_reply__(kwargs["intentCode"])

        if tool == "AskHuman":
            # TODO 如果生成的工具是AskHuman但参数是dict, 1. 尝试提取dict中的内容  2. 回复默认提示话术
            if isinstance(parameter, dict):
                for gkey, gcontent in parameter.items():
                    if self.__check_query_valid__(gcontent):
                        parameter = gcontent
                        break
                if isinstance(parameter, dict):
                    parameter = self.__get_default_reply__(kwargs["intentCode"])
        return [thought, tool, parameter]

    def chat_react(self, *args, **kwargs):
        """调用模型生成答案,解析ReAct生成的结果"""
        max_tokens = kwargs.get("max_tokens", 500)
        _sys_prompt, list_of_plugin_info = self.compose_input_history(**kwargs)
        prompt = build_input_text(_sys_prompt, list_of_plugin_info, **kwargs)
        prompt += "Thought: "
        model = self.gsr.get_model("general_react_model")
        logger.debug(f"ReAct Prompt:\n{prompt}")
        model_output = callLLM(
            prompt,
            temperature=0.7,
            top_p=0.5,
            max_tokens=max_tokens,
            model=model,
            stop=["\nObservation"],
        )
        model_output = "\nThought: " + model_output
        logger.debug(f"ReAct Generate: {model_output}")
        self.update_mid_vars(
            kwargs.get("mid_vars"),
            key="Chat ReAct",
            input_text=prompt,
            output_text=model_output,
            model=model,
        )

        # model_output = """Thought: 任务名和时间都没有提供，无法创建日程。\nAction: AskHuman\nAction Input: {"message": "请提供任务名和时间。"}"""
        out_text = parse_latest_plugin_call(model_output)
        # if not self.prompt_meta_data['prompt_tool_code_map'].get(out_text[1]):
        #     out_text[1] = "AskHuman"

        # (optim) 对于react模式, 如果一个事件提供工具列表, 生成的Action不属于工具列表中, 不同的意图返回不同的话术指导和AskHuman工具 2024年1月9日15:50:18
        out_text = self.__generate_content_verification__(
            out_text, list_of_plugin_info, **kwargs
        )
        try:
            # gen_args = json.loads(out_text[2])
            tool = out_text[1]
            tool_zh = self.prompt_meta_data["prompt_tool_code_map"].get(tool)
            tool_param_msg = self.prompt_meta_data["tool"][tool_zh].get("params")
            # if self.prompt_meta_data['rollout_tool'].get(tool) and tool_param_msg and len(tool_param_msg) ==1:
            if tool_param_msg and len(tool_param_msg) == 1:
                # 对于直接输出的,此处判断改工具设定的参数,通常只有一项 为要输出的话,此时解析对应字段
                if tool_param_msg[0]["schema"]["type"].startswith("str"):
                    out_text[2] = out_text[2][tool_param_msg[0]["name"]]
        except Exception as err:
            # logger.exception(err)
            ...
        kwargs["history"].append(
            {
                "intentCode": kwargs["intentCode"],
                "role": "assistant",
                "content": out_text[0],
                "function_call": {"name": out_text[1], "arguments": out_text[2]},
            }
        )
        return kwargs["history"]

    def compose_input_history(self, **kwargs):
        """拼装sys_prompt里"""
        qprompt = kwargs.get("qprompt")

        sys_prompt, functions = self.promptEngine._call(**kwargs)

        if not qprompt:
            sys_prompt = self.sys_template.format(external_information=sys_prompt)
        else:
            sys_prompt = sys_prompt + "\n\n" + qprompt
        return sys_prompt, functions

    def update_mid_vars(
        self,
        mid_vars,
        input_text=Any,
        output_text=Any,
        key="节点名",
        model="调用模型",
        **kwargs,
    ):
        """更新中间变量"""
        lth = len(mid_vars) + 1
        mid_vars.append(
            {
                "id": lth,
                "key": key,
                "input_text": input_text,
                "output_text": output_text,
                "model": model,
                **kwargs,
            }
        )
        return mid_vars

    def get_parent_intent_name(self, text):
        if "五师" in text:
            return "呼叫五师意图"
        elif "音频" in text:
            return "音频播放意图"
        elif "生活" in text:
            return "生活工具查询意图"
        elif "医疗" in text:
            return "医疗健康意图"
        elif "饮食" in text:
            return "饮食营养意图"
        elif "运动" in text:
            return "运动咨询意图"
        elif "日程":
            return "日程管理意图"
        elif "食材采购" in text:
            return "食材采购意图"
        else:
            return "其它"

    def cls_intent(self, history, mid_vars, **kwargs):
        """意图识别"""
        open_sch_list = ["打开", "日程"]
        market_list = ["打开", "集市"]
        home_list = ["打开", "家居"]
        bp_list = [
            "血压趋势图",
            "血压录入",
            "血压添加",
            "入录血压",
            "添加血压",
            "历史血压",
            "血压历史",
        ]
        inter_info_list = ["打开聊天", "打开交流", "信息交互页面"]
        # st_key, ed_key = "<|im_start|>", "<|im_end|>"
        history = [
            {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
            for i in history
        ]
        # his_prompt = "\n".join([f"{st_key}{i['role']}\n{i['content']}{ed_key}" for i in history]) + f"\n{st_key}assistant\n"
        if sum([1 for i in bp_list if i in history[-1]["content"]]) > 0:
            return "打开功能页面"
        if sum([1 for i in inter_info_list if i in history[-1]["content"]]) > 0:
            return "打开功能页面"
        if sum([1 for i in open_sch_list if i in history[-1]["content"]]) >= 2:
            return "打开功能页面"
        if sum([1 for i in market_list if i in history[-1]["content"]]) >= 2:
            return "打开功能页面"
        if sum([1 for i in home_list if i in history[-1]["content"]]) >= 2:
            return "打开功能页面"
        h_p = "\n".join(
            [
                ("Question" if i["role"] == "user" else "Answer") + f": {i['content']}"
                for i in history[-3:]
            ]
        )
        # prompt = INTENT_PROMPT + his_prompt + "\nThought: "
        if kwargs.get("intentPrompt", ""):
            prompt = kwargs.get("intentPrompt") + "\n\n" + h_p + "\nThought: "
        else:
            prompt = (
                self.prompt_meta_data["tool"]["父意图"]["description"]
                + "\n\n"
                + h_p
                + "\nThought: "
            )
        logger.debug("父意图模型输入：" + prompt)
        generate_text = callLLM(
            query=prompt, max_tokens=200, top_p=0.8, temperature=0, do_sample=False
        )
        logger.debug("意图识别模型输出：" + generate_text)
        intentIdx = generate_text.find("\nIntent: ") + 9
        text = generate_text[intentIdx:].split("\n")[0]
        parant_intent = self.get_parent_intent_name(text)
        if parant_intent in [
            "呼叫五师意图",
            "音频播放意图",
            "生活工具查询意图",
            "医疗健康意图",
            "饮食营养意图",
            "日程管理意图",
            "食材采购意图",
        ]:
            sub_intent_prompt = self.prompt_meta_data["tool"][parant_intent][
                "description"
            ]
            if parant_intent in ["呼叫五师"]:
                history = history[-1:]
                h_p = "\n".join(
                    [
                        ("Question" if i["role"] == "user" else "Answer")
                        + f": {i['content']}"
                        for i in history
                    ]
                )
            if kwargs.get("subIntentPrompt", ""):
                prompt = (
                    kwargs.get("subIntentPrompt").format(sub_intent_prompt)
                    + "\n\n"
                    + h_p
                    + "\nThought: "
                )
            else:
                prompt = (
                    self.prompt_meta_data["tool"]["子意图模版"]["description"].format(
                        sub_intent_prompt
                    )
                    + "\n\n"
                    + h_p
                    + "\nThought: "
                )
            logger.debug("子意图模型输入：" + prompt)
            generate_text = callLLM(
                query=prompt, max_tokens=200, top_p=0.8, temperature=0, do_sample=False
            )
            intentIdx = generate_text.find("\nIntent: ") + 9
            text = generate_text[intentIdx:].split("\n")[0]
        self.update_mid_vars(
            mid_vars,
            key="意图识别",
            input_text=prompt,
            output_text=generate_text,
            intent=text,
        )
        return text

    def __chatter_gaily_compose_func_reply__(self, messages):
        """拼接func中回复的内容到history中, 最终的history只有role/content字段"""
        history = []
        for i in messages:
            if not i.get("function_call"):
                history.append(i)
            else:
                func_args = i["function_call"]
                role = i["role"]
                content = f"{func_args['arguments']}"
                history.append({"role": role, "content": content})
        # 2024年1月24日13:54:32 闲聊轮次太多 保留4轮历史
        history = history[-8:]
        return history

    def chatter_gaily(self, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt"""
        intentCode = kwargs.get("intentCode", "other")
        messages = [i for i in kwargs["history"] if i.get("intentCode") == intentCode]
        messages = self.__chatter_gaily_compose_func_reply__(messages)

        desc = self.prompt_meta_data["event"][intentCode].get("description", "")
        # process = self.prompt_meta_data["event"][intentCode].get("process", "")
        process = ""
        if (
            desc or process
        ):  # (optim) 无描述, 不添加system 2024年1月8日14:07:36, 针对需要走纯粹闲聊的问题
            ext_info = desc + "\n" + process
            messages = [{"role": "system", "content": ext_info}] + messages

        logger.debug(f"闲聊 LLM Input:\n{messages}")
        if not messages:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好"}
            ]
        content = callLLM("", messages, temperature=0.7, top_p=0.45)
        logger.debug(f"闲聊 LLM Output: {content}")
        self.update_mid_vars(
            mid_vars,
            key="闲聊",
            input_text=json.dumps(messages, ensure_ascii=False),
            output_text=content,
        )
        if kwargs.get("return_his"):
            messages.append(
                {
                    "intentCode": "other",
                    "role": "assistant",
                    "content": "I know the answer.",
                    "function_call": {"name": "convComplete", "arguments": content},
                }
            )
            return messages
        else:
            return content

    def chatter_gaily_new(self, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt"""

        def __chatter_gaily_compose_func_reply__(messages):
            """拼接func中回复的内容到history中, 最终的history只有role/content字段"""
            history = []
            for i in messages:
                if not i.get("function_call"):
                    history.append(i)
                else:
                    func_args = i["function_call"]
                    role = i["role"]
                    content = f"{func_args['arguments']}"
                    history.append({"role": role, "content": content})
            # 2024年1月24日13:54:32 闲聊轮次太多 保留4轮历史
            history = history[-8:]
            return history

        intentCode = kwargs.get("intentCode", "other")
        messages = [i for i in kwargs["history"] if i.get("intentCode") == intentCode]
        # messages = __chatter_gaily_compose_func_reply__(messages)

        next_step, messages = self.chatter_assistant.get_next_step(messages)
        self.update_mid_vars(
            mid_vars,
            key="日常闲聊-next_step",
            input_text=messages,
            output_text=next_step,
        )

        if next_step == "searchKB":
            history, dataSource = self.funcall._call(
                out_history=messages, intentCode=intentCode, mid_vars=mid_vars
            )
            content = history[-1]["content"]
            messages[-1]["function_call"]["name"] = "AskHuman"
            messages[-1]["function_call"]["arguments"] = content
        # elif next_step == "searchEngine":
        #     history, dataSource = self.funcall._call(out_history=messages, intentCode=intentCode, mid_vars=mid_vars)
        #     content = history[-1]["content"]
        #     messages[-1]['function_call']['name'] = 'AskHuman'
        #     messages[-1]['function_call']['arguments'] = content
        # elif next_step == "AskHuman":
        else:
            content, messages = self.chatter_assistant.run(messages, kwargs)
            self.update_mid_vars(
                mid_vars,
                key="日常闲聊",
                input_text=json.dumps(messages, ensure_ascii=False),
                output_text=content,
            )
        # else:
        #     content, messages = self.chatter_assistant.run(messages)

        if kwargs.get("return_his"):
            return messages
        else:
            return content

    def chatter_gaily_knowledge(self, mid_vars, **kwargs):
        """组装mysql中闲聊对应的prompt"""

        def compose_func_reply(messages):
            """拼接func中回复的内容到history中

            最终的history只有role/content字段
            """
            payload = {
                "query": "",
                "knowledge_base_name": "",
                "top_k": 5,
                "score_threshold": 1,
                "history": [],
                "stream": False,
                "model_name": "Qwen1.5-14B-Chat",
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 0,
                "prompt_name": "text",
            }
            history = []
            for i in messages:
                if not i.get("function_call"):
                    history.append(i)
                else:
                    func_args = i["function_call"]
                    role = i["role"]
                    content = f"{func_args['arguments']}"
                    history.append({"role": role, "content": content})
            payload["history"] = history[:-1]
            payload["query"] = history[-1]["content"]
            payload["knowledge_base_name"] = "新奥百科知识库"
            return payload

        url = self.gsr.api_config["langchain"] + "/chat/knowledge_base_chat"
        intentCode = kwargs.get("intentCode", "other")

        messages = [i for i in kwargs["history"] if i.get("intentCode") == intentCode]

        desc = self.prompt_meta_data["event"][intentCode].get("description", "")
        process = self.prompt_meta_data["event"][intentCode].get("process", "")
        if (
            desc or process
        ):  # (optim) 无描述, 不添加system 2024年1月8日14:07:36, 针对需要走纯粹闲聊的问题
            ext_info = desc + "\n" + process
            messages = [{"role": "system", "content": ext_info}] + messages

        # event_msg = self.prompt_meta_data['event'][intentCode]
        # system_prompt = event_msg['description'] + "\n" + event_msg['process']

        # messages = [{"role":"system", "content": system_prompt}] + messages
        payload = compose_func_reply(messages)

        response = self.session.post(url, data=json.dumps(payload)).json()
        content, docs = response["answer"], response["docs"]
        self.update_mid_vars(
            mid_vars,
            key="闲聊-知识库-新奥百科",
            input_text=payload,
            output_text=response,
            model="知识库-新奥百科知识库-Qwen1.5-14B-Chat",
        )
        if kwargs.get("return_his"):
            messages.append(
                {
                    "role": "assistant",
                    "content": "I know the answer.",
                    "function_call": {"name": "convComplete", "arguments": content},
                }
            )
            return messages[1:]
        else:
            return content

    def intent_query(self, history, **kwargs):
        mid_vars = kwargs.get("mid_vars", [])
        task = kwargs.get("task", "")
        input_prompt = kwargs.get("prompt", [])
        if task == "verify" and input_prompt:
            intent, desc = get_intent(
                self.cls_intent_verify(history, mid_vars, input_prompt),
                self.gsr.all_intent,self.gsr.com_intent)
        else:
            intent, desc = get_intent(self.cls_intent(history, mid_vars, **kwargs),
                                      self.gsr.all_intent,self.gsr.com_intent)
        if self.intent_map["callout"].get(intent):
            out_text = {
                "message": get_doc_role(intent),
                "intentCode": "doc_role",
                "processCode": "trans_back",
                "intentDesc": desc,
            }
        elif self.intent_map["aiui"].get(intent):
            out_text = {
                "message": "",
                "intentCode": intent,
                "processCode": "aiui",
                "intentDesc": desc,
            }
        elif intent in ["food_rec"]:
            if not kwargs.get("userInfo", {}).get("askTastePrefer", ""):
                out_text = {
                    "message": "",
                    "intentCode": intent,
                    "processCode": "trans_back",
                    "intentDesc": desc,
                }
            else:
                out_text = {
                    "message": "",
                    "intentCode": "food_rec",
                    "processCode": "alg",
                    "intentDesc": desc,
                }
        # elif intent in ['sport_rec']:
        #    if kwargs.get('userInfo', {}).get('askExerciseHabbit', '') and kwargs.get('userInfo',{}).get('askExerciseTabooDegree', '') and kwargs.get('userInfo', {}).get('askExerciseTabooXt', ''):
        #        out_text = {'message':'',
        #                'intentCode':intent,'processCode':'alg', 'intentDesc':desc}
        #    else:
        #        out_text = {'message':'', 'intentCode':intent,
        #                'processCode':'trans_back', 'intentDesc':desc}
        else:
            out_text = {
                "message": "",
                "intentCode": intent,
                "processCode": "alg",
                "intentDesc": desc,
            }
        logger.debug("意图识别输出：" + json.dumps(out_text, ensure_ascii=False))
        return out_text

    def fetch_intent_code(self):
        """返回所有的intentCode"""
        intent_code_map = {
            "get_userInfo_msg": list(self.intent_map["userinfo"].keys()),
            "get_reminder_tips": list(self.intent_map["tips"].keys()),
            "other": [
                "BMI",
                "food_rec",
                "sport_rec",
                "schedule_manager",
                "schedule_qry_up",
                "auxiliary_diagnosis",
                "other",
            ],
        }
        return intent_code_map

    def pre_fill_param(self, *args, **kwargs):
        """结合业务逻辑，预构建输入"""
        intentCode = kwargs.get("intentCode")
        if not self.prompt_meta_data["event"].get(intentCode) and not intentCode in [
            "weight_meas",
            "blood_meas",
            "glucose_diagnosis",
        ]:
            logger.debug(
                f"not support current event {intentCode}, change intentCode to other."
            )
            kwargs["intentCode"] = "other"
        if intentCode == "schedule_qry_up" and not kwargs.get("history"):
            kwargs["history"] = [{"role": 0, "content": "帮我查询今天的日程"}]
        if "schedule" in intentCode:
            kwargs["schedule"] = self.funcall.call_get_schedule(*args, **kwargs)
        return args, kwargs

    async def general_yield_result(self, *args, **kwargs) -> AsyncGenerator:
        """预处理,调用pipeline，返回结果
        1. 通过role_map转换role定义
        2. history 由backend_history拼接用户输入
        """
        try:
            args, kwargs = self.pre_fill_param(*args, **kwargs)
            kwargs["his"] = kwargs.get("history", [])
            if kwargs.get("history"):
                if kwargs.get("intentCode") == "auxiliary_diagnosis":
                    history = [
                        {**i, "role": role_map.get(str(i["role"]), "user")}
                        for i in kwargs["history"]
                    ]
                    logger.info(f"Process message: {history}")

                    kwargs["history"] = kwargs["backend_history"] + history
                    if not kwargs["history"][-1].get("intentCode"):
                        for i in kwargs["history"]:
                            i["intentCode"] = kwargs["intentCode"]
                else:
                    history = [
                        {**i, "role": role_map.get(str(i["role"]), "user")}
                        for i in kwargs["history"]
                    ]
                    kwargs["history"] = kwargs["backend_history"] + [history[-1]]
                    if not kwargs["history"][-1].get("intentCode"):
                        kwargs["history"][-1]["intentCode"] = kwargs["intentCode"]

            if kwargs["intentCode"] == "other":
                kwargs["prompt"] = None
                kwargs["sys_prompt"] = None

            logger.info(f"Process message {kwargs.get('history')}")

            if kwargs.get("mode") == "deepseek":
                _iterable = self.chatter_assistant.run_yield(history=kwargs["history"], kwargs=kwargs)
            else:
                _iterable = self.pipeline(*args, **kwargs)

            # 风控：从 history 中拿最近一条 user 输入
            user_input = ""
            for msg in reversed(kwargs.get("history", [])):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break

            # while True:
            async for yield_item in _iterable:
                try:
                    # yield_item = next(_iterable)
                    if kwargs.get("mode") == "deepseek":
                        # DeepSeek 模式：不做任何结构修改，直接 yield
                        yield yield_item
                    else:
                        if not yield_item["data"].get("type"):
                            yield_item["data"]["type"] = "Result"
                        if yield_item["data"]["type"] == "Result" and not yield_item[
                            "data"
                        ].get("dataSource"):
                            yield_item["data"]["dataSource"] = DEFAULT_DATA_SOURCE

                        # 🔒 风控判断，仅在 Result 类型时触发
                        if yield_item["data"]["type"] == "Result":
                            detect_result = await detect_sensitive_all(
                                user_input=user_input,
                                sensitive_words=self.gsr.sensitive_words,
                                regex_patterns=REGEX_PATTERNS
                            )
                            if detect_result["is_blocked"]:
                                logger.warning(f"⚠️ 命中敏感词，替换回答: {detect_result['matched']}")
                                yield_item["data"]["message"] = detect_result["response"]

                        yield yield_item
                except StopIteration as err:
                    break

        except Exception as e:
            logger.error(f"生成结果处理失败: {e}")

    def __log_init(self, **kwargs):
        """初始打印日志"""
        intentCode = kwargs.get("intentCode")
        history = kwargs.get("history")
        logger.info(f"intentCode: {intentCode}")
        if history:
            logger.info(f"Input: {history[-1]['content']}")

    def parse_last_history(self, history):
        tool = history[-1]["function_call"]["name"]
        content = history[-1]["function_call"]["arguments"]
        thought = history[-1]["content"]
        # logger.debug(f"Action: {tool}")
        # logger.debug(f"Thought: {thought}")
        # logger.debug(f"Action Input: {content}")
        return tool, content, thought

    def get_userInfo_msg(self, prompt, history, intentCode, mid_vars):
        """获取用户信息"""
        logger.debug(f"信息提取prompt:\n{prompt}")
        model_output = callLLM(
            prompt,
            verbose=False,
            temperature=0,
            top_p=0.8,
            max_tokens=200,
            do_sample=False,
        )
        logger.debug("信息提取模型输出：" + model_output)
        content = model_output.replace("___", "").strip()
        cnt = []
        for i in content.split("\n"):
            if i.startswith("问题") or not i.strip():
                continue
            cnt.append(i)
        content = cnt[0]
        # model_output = model_output[model_output.find('Output')+7:].split('\n')[0].strip()
        self.update_mid_vars(
            mid_vars,
            key="获取用户信息 01",
            input_text=prompt,
            output_text=model_output,
            model="Qwen1.5-14B-Chat",
        )

        if (
            sum(
                [i in content for i in ["询问", "提问", "转移", "结束", "未知", "停止"]]
            )
            != 0
        ):
            logger.debug("信息提取流程结束...")
            content = self.chatter_gaily(mid_vars, history=history)
            intentCode = EXT_USRINFO_TRANSFER_INTENTCODE
        else:
            """
            content = ''
            for i in content.strip().split('\n'):
                if i.startswith('标签值为：'):
                    content = i
                    break
            if not content:
                content = model_output.strip().split('\n')[0]
            """
            logger.debug("标签归一前提取内容：" + content)
            content = norm_userInfo_msg(intentCode, content)
            logger.debug("标签归一后提取内容：" + content)
            content = self.clean_userInfo(content)

        # content = model_output
        # content = self.clean_userInfo(content)

        content = content if content else "未知"
        content = "未知" if "Error" in content else content
        logger.debug(
            "信息提取返回内容数据：" + content + "     返回意图码：" + intentCode
        )

        return content, intentCode

    def clean_userInfo(self, content):
        content = (
            content.replace("'", "")
            .replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
            .replace("。", "")
        )
        content = (
            content.replace("用户昵称：", "")
            .replace("输出：", "")
            .replace("标签值为：", "")
        )
        return content

    def get_reminder_tips(
        self, prompt, history, intentCode, model="Qwen2-7B-Instruct", mid_vars=None
    ):
        logger.debug("remind prompt: " + prompt)
        content = callLLM(
            query=prompt,
            verbose=False,
            do_sample=False,
            temperature=0.1,
            top_p=0.2,
            max_tokens=500,
            model=model,
        )
        self.update_mid_vars(
            mid_vars, key="", input_text=prompt, output_text=content, model=model
        )
        logger.debug("remind model output: " + content)
        if content.startswith("（）"):
            content = content[2:].strip()
        return content, intentCode

    def open_page(self, mid_vars, **kwargs):
        """组装mysql中打开页面对应的prompt"""
        add_bp_list = ["血压录入", "血压添加", "录入血压", "添加血压"]
        add_diet_list = ["打开记录", "打开录入", "打开添加"]
        diet_record_list = [
            "饮食记录",
            "饮食添加",
            "打开推荐",
            "饮食评估",
            "食谱",
            "我的饮食",
            "食谱页面",
            "餐食记录",
        ]
        market_list = ["集市"]
        personal_list = ["我的设置"]
        qr_code_list = ["二维码"]
        inter_info_list = ["交流", "聊天", "信息交互"]
        input_history = [
            {"role": role_map.get(str(i["role"]), "user"), "content": i["content"]}
            for i in kwargs["history"]
        ]
        input_history = input_history[-3:]
        if "血压趋势" in input_history[-1]["content"]:
            return 'pagename:"bloodPressure-trend-chart"'
        elif sum([1 for i in add_bp_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"add-blood-pressure"'
        elif sum([1 for i in inter_info_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"interactive-information"'
        elif sum([1 for i in personal_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"personal-setting"'
        elif sum([1 for i in qr_code_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"qr-code"'
        elif (
            "血压历史" in input_history[-1]["content"]
            or "历史血压" in input_history[-1]["content"]
        ):
            return 'pagename:"record-list3"'
        elif sum([1 for i in add_diet_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"add-diet"'
        elif (
            sum([1 for i in diet_record_list if i in input_history[-1]["content"]]) > 0
        ):
            return 'pagename:"diet-record"'
        elif sum([1 for i in market_list if i in input_history[-1]["content"]]) > 0:
            return 'pagename:"my-market"'
        elif (
            "打开" in input_history[-1]["content"]
            and "日程" in input_history[-1]["content"]
        ):
            return 'pagename:"my-schedule"'

        hp = [h["role"] + " " + h["content"] for h in input_history]
        hi = ""
        if len(input_history) > 1:
            hi = (
                "用户历史会话如下，可以作为意图识别的参考，但不要过于依赖历史记录，因为它可能是狠久以前的记录："
                + "\n"
                + "\n".join([h["role"] + h["content"] for h in input_history[-3:-1]])
                + "\n"
                + "当前用户输入：\n"
            )
        hi += f'Question:{input_history[-1]["content"]}\nThought:'
        ext_info = (
            self.prompt_meta_data["event"]["open_Function"]["description"]
            + "\n"
            + self.prompt_meta_data["event"]["open_Function"]["process"]
            + "\n"
            + hi
            + "\nThought:"
        )
        input_history = [{"role": "system", "content": ext_info}]
        logger.debug(
            "打开页面模型输入：" + json.dumps(input_history, ensure_ascii=False)
        )
        content = callLLM("", input_history, temperature=0, top_p=0.8, do_sample=False)
        if content.find("Answer") != -1:
            content = content[content.find("Answer") + 7 :].split("\n")[0].strip()
        if content.find("Output") != -1:
            content = content[content.find("Response") + 6 :].split("\n")[0].strip()
        if content.find("Response") != -1:
            content = content[content.find("") + 9 :].split("\n")[0].strip()

        self.update_mid_vars(
            mid_vars,
            key="打开功能画面",
            input_text=json.dumps(input_history, ensure_ascii=False),
            output_text=content,
        )
        return content

    def get_pageName_code(self, text):
        logger.debug("页面生成内容：" + text)
        if "bloodPressure-trend-chart" in text and "pagename" in text:
            return "bloodPressure-trend-chart"
        elif "add-blood-pressure" in text and "pagename" in text:
            return "add-blood-pressure"
        elif "record-list3" in text and "pagename" in text:
            return "record-list3"
        elif "my-schedule" in text and "pagename" in text:
            return "my-schedule"
        elif "add-diet" in text and "pagename" in text:
            return "add-diet"
        elif "diet-record" in text and "pagename" in text:
            return "diet-record"
        elif "my-market" in text and "pagename" in text:
            return "my-market"
        elif "personal-setting" in text and "pagename" in text:
            return "personal-setting"
        elif "qr-code" in text and "pagename" in text:
            return "qr-code"
        elif "smart-home" in text and "pagename" in text:
            return "smart-home"
        elif "interactive-information" in text and "pagename" in text:
            return "interactive-information"
        else:
            return "other"

    def __raw_event_process_for_doctor_recommend_at_end__(
        self, intentCode, appendData, chat_history, ret_result
    ):
        """附加医生推荐 原流程追加信息"""
        if intentCode.startswith("auxiliary_diagnosis"):
            if self.__assert_diet_suggest_in_content__(chat_history[-1]["content"]):
                # 2024年4月24日11:06:51 增加新事件
                # 判断auxiliary_diagnosis_with_doctor_recommend结束
                append_content = "请问是否需要我帮您找一位医生？"
                appendData["contents"] = [append_content]
                # appendData["scheme_gen"] = 1
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": append_content,
                        "intentCode": "aigc_functions_doctor_recommend",
                    }
                )
        elif intentCode.startswith("report_interpretation_chat"):
            _content = (
                chat_history[-1]["function_call"]["arguments"]
                if chat_history[-1].get("function_call")
                else None
            )
            if (
                _content
                and "?" not in _content
                and "？" not in _content
                and not appendData.get("doctor_rec")
            ):
                _append_content = "请问是否需要我帮您找一位医生？"
                # appendData["scheme_gen"] =
                appendData["contents"].append(_append_content)
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": _append_content,
                        "intentCode": "aigc_functions_doctor_recommend",
                    }
                )
        # elif intentCode.startswith("blood_meas"):
        #     _content = (
        #         chat_history[-1]["function_call"]["arguments"]
        #         if chat_history[-1].get("function_call")
        #         else None
        #     )
        #     if ret_result.get("init_intent"):
        #         _append_content = "请问是否需要我帮您找一位医生？"
        #         appendData["contents"].append(_append_content)
        #         chat_history.append(
        #             {
        #                 "role": "assistant",
        #                 "content": _append_content,
        #                 "intentCode": "aigc_functions_doctor_recommend",
        #             }
        #         )

    async def complete(
        self, mid_vars: List[object], tool: str = "convComplete", **kwargs
    ):
        """only prompt模式的生成及相关逻辑"""
        # assert kwargs.get("prompt"), "Current process type is only_prompt, but not prompt passd."
        _appendData = {"doctor_rec": []}
        weight_res, blood_res, conts, notify_blood_pressure_contnets = {}, {}, [], []
        content, level, modi_scheme, thought, contact_doctor, visit_verbal_idx = "", "", "", "我知道如何回答", -1, -1
        prompt = kwargs.get("prompt")
        chat_history = kwargs["history"]
        intentCode = kwargs["intentCode"]
        sch = blood_trend_gen = call_120 = is_visit = exercise_video = (
            notifi_daughter_doctor
        ) = weight_trend_gen = False
        if self.intent_map["userinfo"].get(intentCode):
            content, intentCode = self.get_userInfo_msg(
                prompt, chat_history, intentCode, mid_vars
            )
        elif self.intent_map["tips"].get(intentCode):
            content, intentCode = self.get_reminder_tips(
                prompt, chat_history, intentCode, mid_vars=mid_vars
            )
        elif intentCode == "open_Function":
            output_text = self.open_page(mid_vars, **kwargs)
            content = (
                "稍等片刻，页面即将打开"
                if self.get_pageName_code(output_text) != "other"
                else output_text
            )
            intentCode = self.get_pageName_code(output_text)
            logger.debug("页面Code: " + intentCode)
        elif intentCode in [
            "auxiliary_diagnosis",
            "auxiliary_diagnosis_with_doctor_recommend",
        ]:
            doctor_rec_code = "aigc_functions_doctor_recommend"
            # 调用医生推荐
            if (
                len(chat_history) >= 2
                and chat_history[-2]["intentCode"] == doctor_rec_code
            ):
                # chat_history[-1] 为用户对医生的需求
                # chat_history[-2] 为询问是否需要推荐医生
                # chat_history[-3] 为当前事件的总结
                param = AigcFunctionsDoctorRecommendRequest(
                    intentCode=doctor_rec_code,
                    prompt=chat_history[-3]["content"],
                    messages=[
                        {"role": "assistant", "content": chat_history[-2]["content"]},
                        {"role": "user", "content": chat_history[-1]["content"]},
                    ],
                ).model_dump()
                doctor_rec = await self.gsr.agents.call_function(**param)
                _appendData["doctor_rec"] = doctor_rec
                thought = "判断是否需要医生推荐"
                if doctor_rec:
                    content = "根据您的问诊结果，我已为您匹配离您最近、最适于您的医师。"
                else:
                    content = (
                        "好的, 我还可以为您提供其他健康咨询服务, 请问您有什么问题吗?"
                    )
                tool = "convComplete"

            else:
                mid_vars, conts, (thought, content) = (
                    await self.custom_chat_auxiliary.chat(mid_vars=mid_vars, **kwargs)
                )
                tool = "askHuman"
            if conts and len(conts)>0:
                intentCode = "assert_whether_contact_family_doctor"
                tool = "convComplete"
                # content = content+conts[0]
        elif intentCode == "pressure_meas":
            pressure_res = self.custom_chat_model.chat(mid_vars=mid_vars, **kwargs)
            content = pressure_res["content"]
            # sch = pressure_res['scheme_gen']
            thought = pressure_res["thought"]
            sch = pressure_res["scheme_gen"]
            tool = (
                "askHuman" if pressure_res["scene_ending"] == False else "convComplete"
            )
        elif intentCode == "weight_meas":
            weight_res = self.custom_chat_model.chat(mid_vars=mid_vars, **kwargs)
            if weight_res["contents"]:
                content = weight_res["contents"][0]
                conts = weight_res["contents"][1:]
            sch = weight_res["scheme_gen"]
            thought = weight_res["thought"]
            weight_trend_gen = weight_res["weight_trend_gen"]
            modi_scheme = weight_res.get("modi_scheme", "scheme_no_change")

            level = ""
            tool = "askHuman" if weight_res["scene_ending"] == False else "convComplete"
        elif intentCode  =="blood_meas":
            blood_res = self.custom_chat_model.chat(mid_vars=mid_vars, **kwargs)
            content = blood_res["contents"][0]
            conts = blood_res["contents"][1:]
            sch = blood_res["scheme_gen"]
            thought = blood_res["thought"]
            level = blood_res["level"]
            blood_trend_gen = blood_res["blood_trend_gen"]
            notifi_daughter_doctor = blood_res["notifi_daughter_doctor"]
            call_120 = blood_res["call_120"]
            is_visit = blood_res["is_visit"]
            contact_doctor = blood_res['contact_doctor']
            visit_verbal_idx = blood_res["visit_verbal_idx"]
            # idx = blood_res.get('idx', 0)
            tool = "askHuman" if blood_res["scene_ending"] == False else "convComplete"
            if blood_res["scene_ending"] and (level == 0 or level == 1):
                conts.append(
                    "基于您的生命熵变化，我给您匹配一个降压小妙招，您可以试一下。然后建议联系家庭医生对您进行后续健康服务，我现在帮您邀请在线家庭医生吧。"
                )
                intentCode = "assert_whether_contact_family_doctor"
            notify_blood_pressure_contnets = blood_res.get("events", [])
            exercise_video = blood_res.get("exercise_video", False)
        # todo
        elif intentCode == "blood_interact":
            mid_vars, conts, (thought, content) = (
                    await self.custom_chat_auxiliary.chat(mid_vars=mid_vars, **kwargs)
                )
            if len(conts)==0:
                tool = "askHuman"
        elif intentCode == "glucose_consultation":
            mid_vars, conts, (thought, content) = (
                    await self.custom_chat_auxiliary.chat(mid_vars=mid_vars, **kwargs)
                )
            if len(conts)==0:
                tool = "askHuman"
            else:
                conts=[]
        elif intentCode == "sanji_glucose_diagnosis":
            mid_vars, conts, (thought, content) = (
                    await self.custom_chat_auxiliary.chat(mid_vars=mid_vars, **kwargs)
                )
            if len(conts)==0:
                tool = "askHuman"
        elif intentCode =="chat_start_with_weather":
            mid_vars, conts, (thought, content) = (
                    await self.custom_chat_auxiliary.chat(mid_vars=mid_vars, **kwargs)
                )
            if len(conts)==0:
                tool = "askHuman"
        # elif intentCode == "blood_meas_with_doctor_recommend":
        #     blood_res = self.custom_chat_model.chat(mid_vars=mid_vars, **kwargs)
        #     content = blood_res["contents"][0]
        #     conts = blood_res["contents"][1:]
        #     sch = blood_res["scheme_gen"]
        #     thought = blood_res["thought"]
        #     level = blood_res["level"]
        #     blood_trend_gen = blood_res["blood_trend_gen"]
        #     notifi_daughter_doctor = blood_res["notifi_daughter_doctor"]
        #     call_120 = blood_res["call_120"]
        #     is_visit = blood_res["is_visit"]
        #     # idx = blood_res.get('idx', 0)
        #     tool = "askHuman" if blood_res["scene_ending"] == False else "convComplete"
        #     if blood_res["scene_ending"]:
        #         conts.append(
        #             "我建议您联系家庭医生对你进行后续健康服务，我现在帮您邀请家庭医生吧？"
        #         )
        #         intentCode = "assert_whether_contact_family_doctor"
        #     notify_blood_pressure_contnets = blood_res.get("events", [])
        #     exercise_video = blood_res.get("exercise_video", False)
        elif intentCode in [
            "report_interpretation_chat",
            "report_interpretation_chat_with_doctor_recommend",
        ]:
            doctor_rec_code = "aigc_functions_doctor_recommend"
            # 调用医生推荐
            if (
                len(chat_history) >= 2
                and chat_history[-2]["intentCode"] == doctor_rec_code
            ):
                param = AigcFunctionsDoctorRecommendRequest(
                    intentCode=doctor_rec_code,
                    prompt=chat_history[-3]["content"],
                    messages=[
                        {"role": "assistant", "content": chat_history[-2]["content"]},
                        {"role": "user", "content": chat_history[-1]["content"]},
                    ],
                ).model_dump()
                doctor_rec = await self.gsr.agents.call_function(**param)
                _appendData["doctor_rec"] = doctor_rec
                thought = "判断是否需要医生推荐"
                if doctor_rec:
                    content = "根据您的问诊结果，我已为您匹配离您最近、最适于您的医师。"
                else:
                    content = (
                        "好的, 我还可以为您提供其他健康咨询服务, 请问您有什么问题吗?"
                    )
                tool = "convComplete"
            else:
                kwargs["history"] = [
                    i
                    for i in kwargs["history"]
                    if i.get("intentCode").startswith("report_interpretation_chat")
                ]
                mid_vars, chat_history, conts, sch, (thought, content, tool) = (
                    self.custom_chat_report_interpretation_ask.chat(
                        mid_vars=mid_vars, **kwargs
                    )
                )
        elif intentCode == "report_interpretation_answer":
            kwargs["history"] = [
                i
                for i in kwargs["history"]
                if i.get("intentCode") == "report_interpretation_answer"
            ]
            mid_vars, chat_history, conts, sch, (thought, content, tool) = (
                self.custom_chat_report_interpretation_answer.chat(
                    mid_vars=mid_vars, **kwargs
                )
            )
        elif intentCode == "assert_whether_contact_family_doctor":
            # 判断是否需要联系家庭医生
            user_input = chat_history[-1]["content"]
            # prompt_template = self.gsr.get_event_item(
            #     "assert_whether_contact_family_doctor"
            # )["description"]
            prompt_template = (
                "请你帮我做判断, 输出选项中给定的内容\n"
                "问: 请问是否需要帮您联系家庭医生\n"
                "用户: {user_input}"
            )
            prompt = prompt_template.format(user_input=user_input)
            contactFamilyDoctor = self.gsr.agents.aigc_functions_single_choice(
                prompt=prompt,
                options=["YES", "NO"],
            )
            if contactFamilyDoctor == "YES":
                contactFamilyDoctor = 1
                content = "好的，正为您联系你的家庭医生"
            else:
                contactFamilyDoctor = 0
                content = "好的，我还能为您提供什么帮助？"
            intentCode = ""
            _appendData["contactFamilyDoctor"] = contactFamilyDoctor
        else:
            content = self.chatter_gaily(mid_vars, return_his=False, **kwargs)

        assert type(content) == str, "only_prompt模式下，返回值必须为str类型"

        appendData = {
            "contents": conts,
            "scheme_gen": sch,
            "level": level,
            "blood_trend_gen": blood_trend_gen,
            "notifi_daughter_doctor": notifi_daughter_doctor,
            "call_120": call_120,
            "is_visit": is_visit,
            "modi_scheme": modi_scheme,
            "weight_trend_gen": weight_trend_gen,
            "events": notify_blood_pressure_contnets,
            "exercise_video": exercise_video,
            "contact_doctor":contact_doctor,
            "visit_verbal_idx": visit_verbal_idx,
            **_appendData,
        }
        chat_history.append(
            {
                "role": "assistant",
                "content": thought,
                "function_call": {"name": tool, "arguments": content},
                "intentCode": intentCode,
            }
        )
        return appendData, chat_history, intentCode

    def complete_temporary(self, mid_vars: List[object], **kwargs):
        # XXX 演示临时增加逻辑 2024年01月31日12:39:28
        # XXX 推出这句话同时调用创建日程（2个：体温监测、挂号）
        # assert kwargs.get("prompt"), "Current process type is only_prompt, but not prompt passd."
        # prompt = kwargs.get("prompt")
        history = kwargs["history"]
        intentCode = kwargs["intentCode"]
        content = (
            "请您时刻关注自己的病情变化，如果出现新症状（胸痛、呼吸困难、疲劳等）或者原有症状加重（如咳嗽频率增加、持续发热、症状持续时间超过3天），建议您线下就医。"
            + "依据病情若有需要推荐您在廊坊市人民医院呼吸内科就诊。廊坊市人民医院的公众号挂号渠道0点开始放号。我帮您设置一个23:55的挂号日程，您看可以吗？"
        )

        url = self.gsr.api_config["ai_backend"] + "/alg-api/schedule/manage"
        for task, cronData in [
            ["体温测量", "一个小时后"],
            ["挂号提醒", "今晚23点55分"],
        ]:
            if cronData == "一个小时后":
                cronData = date_after(hours=1)
            elif cronData == "今晚23点55分":
                cronData = curr_time()[:10] + " 23:55:00"
            payload = {
                "customId": kwargs.get("customId"),
                "orgCode": kwargs.get("orgCode"),
                "taskName": task,
                "cronDate": cronData,
                "taskType": "reminder",
                "intentCode": "CREATE",
            }
            resp_js = self.session.post(url, json=payload).json()
            if resp_js["code"] == 200 and resp_js["data"] is True:
                ...
            else:
                logger.error(f"Error to create schedule {task}")
        # reply = self.funcall.scheduleManager.create(
        #     history=_history,
        #     intentCode="schedule_manager",
        #     orgCode=kwargs.get("orgCode"),
        #     customId=kwargs.get("customId"),
        #     mid_vars=kwargs.get("mid_vars", []),
        # )
        history.append(
            {
                "role": "assistant",
                "content": "I know the answer.",
                "function_call": {"name": "convComplete", "arguments": content},
                "intentCode": intentCode,
            }
        )
        return history

    def complete_temporary_v1(self, mid_vars: List[object], **kwargs):
        # XXX 演示临时增加逻辑 2024年01月31日12:39:28
        # XXX 推出这句话同时调用创建日程（2个：体温监测、挂号）
        # assert kwargs.get("prompt"), "Current process type is only_prompt, but not prompt passd."
        # prompt = kwargs.get("prompt")
        history = kwargs["history"]
        intentCode = kwargs["intentCode"]
        content = "我1小时后会提醒您测量体温。"

        history.append(
            {
                "role": "assistant",
                "content": "I know the answer.",
                "function_call": {"name": "convComplete", "arguments": content},
                "intentCode": intentCode,
            }
        )
        return history

    async def interact_first(self, mid_vars, **kwargs):
        """首次交互"""
        intentCode = kwargs.get("intentCode")
        out_history = None
        appendData = {}
        if self.prompt_meta_data["event"].get(intentCode):
            # XXX 演示临时增加逻辑 2024年01月31日11:28:00
            # XXX 判断kwargs历史中最后一条的content字段和"我需要去医院吗？"是否一致，如果一致，则进入临时逻辑，否则进入正常流程
            if (
                kwargs["history"]
                and "我需要去医院吗" in kwargs["history"][-1]["content"]
            ):
                out_history = self.complete_temporary(mid_vars=mid_vars, **kwargs)
            elif (
                kwargs["history"]
                and len(kwargs["history"]) >= 2
                and kwargs["history"][-2].get("function_call")
                and isinstance(kwargs["history"][-2].get("function_call", ""), str)
                and kwargs["history"][-2]["function_call"]["arguments"].startswith(
                    "请您时刻关注自己的病情变化，"
                )
                and [
                    i
                    for i in ["好的", "行", "可以", "ok", "OK"]
                    if i in kwargs["history"][-1]["content"]
                ]
                # and set(kwargs["history"][-1]["content"]).intersection(set("好行可以"))
            ):
                out_history = self.complete_temporary_v1(mid_vars=mid_vars, **kwargs)
            elif intentCode == "other":
                # 2023年12月26日10:07:03 闲聊接入知识库 https://devops.aliyun.com/projex/task/VOSE-3715# 《模型中调用新奥百科的知识内容》
                # out_history = self.chatter_gaily(mid_vars, **kwargs, return_his=True)
                out_history = self.chatter_gaily_new(
                    mid_vars, **kwargs, return_his=True
                )
            elif intentCode == "enn_wiki":
                out_history = self.chatter_gaily_knowledge(
                    mid_vars, **kwargs, return_his=True
                )
            elif self.prompt_meta_data["event"][intentCode].get("process_type") in [
                "only_prompt",
                "custom_chat",
            ]:
                appendData, out_history, intentCode = await self.complete(
                    mid_vars=mid_vars, **kwargs
                )
                kwargs["intentCode"] = intentCode
            elif (
                self.prompt_meta_data["event"][intentCode].get("process_type")
                == "react"
            ):
                out_history = self.chat_react(mid_vars=mid_vars, **kwargs)
            elif intentCode == "jia_kang_bao":
                kwargs["user_question"] = kwargs.get("history", [{}])[-1].get("content", "")
                kwargs["messages"] = kwargs.get("history", [])

                health_expert = HealthExpertModel(self.gsr)
                response = await health_expert.aigc_functions_jia_kang_bao_support(**kwargs)
                if response is None or not response.get("answer"):
                    # 修改所有相关 intentCode 为 "other"
                    intentCode = "other"
                    kwargs["intentCode"] = "other"
                    for msg in kwargs.get("messages", []):
                        msg["intentCode"] = "other"
                    for history_item in kwargs.get("history", []):
                        history_item["intentCode"] = "other"

                    # 切换到闲聊逻辑
                    out_history = self.chatter_gaily_new(
                        mid_vars, **kwargs, return_his=True
                    )
                else:
                    response_content = response.get("answer")
                    out_history = kwargs["messages"] + [{
                        "role": "assistant",
                        "content": response.get("thought", "基于知识库检索"),
                        "function_call": {
                            "name": "convComplete",
                            "arguments": response_content
                        },
                        "intentCode": intentCode
                    }]
                    appendData = {"guessQuestions": response["guess_you_want"]} if response.get("guess_you_want") else {}
        if not out_history:
            out_history = self.chat_react(
                mid_vars=mid_vars, return_his=True, max_tokens=100, **kwargs
            )
        return appendData, out_history, intentCode

    def if_init(self, tool):
        # XXX 不是所有的流程都会调用工具，比如未定义意图的闲聊
        return self.prompt_meta_data["init_intent"].get(tool, False)

    def __assert_diet_suggest_in_content__(self, content):
        """判断是否有建议饮食"""
        model = self.gsr.model_config.get(
            "assert_diet_suggest_in_content", "Qwen1.5-14B-Chat"
        )
        promt = f"{content}\n请你理解以上文本内容，判断文本是否同时包含诊断结果、食物推荐，如果同时包含，输出“YES”，否则输出“NO”"
        logger.debug(f"判断是否有建议饮食 LLM Input: {promt}")
        messages = [{"role": "user", "content": promt}]
        flag = callLLM(
            model=model, history=messages, temperature=0, top_p=0.8, do_sample=False
        )
        logger.debug(f"判断是否有建议饮食 LLM Output: {flag}")

        if "yes" in flag.lower():
            return True
        else:
            return False

    async def pipeline(self, mid_vars=[], **kwargs) -> AsyncGenerator:
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
        intentCode = kwargs.get("intentCode")
        mid_vars = kwargs.get("mid_vars", [])
        dataSource = DEFAULT_DATA_SOURCE
        appendData, out_history, intentCode = await self.interact_first(
            mid_vars=mid_vars, **kwargs
        )
        while True:
            tool, content, thought = self.parse_last_history(out_history)

            if (
                self.prompt_meta_data["event"].get(intentCode)
                and self.prompt_meta_data["event"][intentCode]["process_type"]
                != "only_prompt"
            ):  # 2023年12月13日15:35:50 only_prompt对应的事件不输出思考
                ret_tool = make_meta_ret(
                    msg=tool, type="Tool", code=intentCode, gsr=self.gsr
                )
                ret_thought = make_meta_ret(
                    msg=thought, type="Thought", code=intentCode, gsr=self.gsr
                )
                yield {
                    "data": ret_tool,
                    "mid_vars": mid_vars,
                    "history": out_history,
                    "appendData": appendData,
                }
                yield {
                    "data": ret_thought,
                    "mid_vars": mid_vars,
                    "history": out_history,
                    "appendData": appendData,
                }

            if self.prompt_meta_data["rollout_tool"].get(
                tool
            ) or not self.funcall.funcmap.get(tool):
                # 2023年12月17日17:19:06 增加判断是否支持对应函数 未定义支持的 即使未写rollout_tool也直接返回,不走函数调用
                break
            try:
                kwargs["history"], dataSource = self.funcall._call(
                    out_history=out_history, mid_vars=mid_vars, **kwargs
                )
            except AssertionError as err:
                logger.error(err)
                kwargs["history"], dataSource = self.funcall._call(
                    out_history=out_history, mid_vars=mid_vars, **kwargs
                )

            if self.prompt_meta_data["rollout_tool_after_complete"].get(tool):
                # 工具执行完成后输出
                content = kwargs["history"][-1]["content"]
                break
            else:
                # function_call的结果, self_rag
                content = kwargs["history"][-1]["content"]
                ret_function_call = make_meta_ret(
                    msg=content, type="Observation", code=intentCode, gsr=self.gsr
                )
                yield {
                    "data": ret_function_call,
                    "mid_vars": mid_vars,
                    "history": out_history,
                    "appendData": appendData,
                }
                out_history = self.chat_react(mid_vars=mid_vars, **kwargs)

        ret_result = make_meta_ret(
            end=True,
            msg=content,
            code=intentCode,
            gsr=self.gsr,
            init_intent=self.if_init(tool),
            dataSource=dataSource,
        )
        if intentCode.endswith("_with_doctor_recommend"):
            self.__raw_event_process_for_doctor_recommend_at_end__(
                intentCode, appendData, out_history, ret_result
            )

        """
        # 演示临时增加逻辑 2024年01月31日11:28:00
        # 2024年4月23日18:37:06 注释掉食材采购清单的逻辑
        if (
            intentCode.endswith("_with_doctor_recommend")
            and not ret_result["init_intent"]
        ):
            # if len([i for i in ["根据", "描述", "水果", "建议", "注意休息", "可以吃"] if i in content]) >= 3:
            if self.__assert_diet_suggest_in_content__(content):
                purchasing_list = (
                    self.gsr.expert_model.food_purchasing_list_generate_by_content(
                        content
                    )
                )
                ret_result["intentCode"] = "create_food_purchasing_list"
                ret_result["appendData"] = purchasing_list
                ret_result["message"] += "\n为您生成了一份采购清单，请确认"
        """

        yield {
            "data": ret_result,
            "mid_vars": mid_vars,
            "history": out_history,
            "appendData": appendData,
        }


if __name__ == "__main__":
    chat = Chat_v2(InitAllResource())
    ori_input_param = testParam.param_dev_report_interpretation_chat
    prompt = ori_input_param["prompt"]
    history = ori_input_param["history"]
    intentCode = ori_input_param["intentCode"]
    customId = ori_input_param["customId"]
    orgCode = ori_input_param["orgCode"]
    while True:
        out_text, mid_vars = next(chat.general_yield_result(**ori_input_param))