# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-30 10:08:10
@desc    :   自定义对话模型
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
from typing import Any, AnyStr, Dict, List

from requests import Session
from sqlalchemy.engine import base

from data.constrant import (
    CUSTOM_CHAT_REPOR_TINTERPRETATION_ANSWER_SYS_PROMPT,
    CUSTOM_CHAT_REPOR_TINTERPRETATION_SYS_PROMPT_END_SUMMARY,
    CUSTOM_CHAT_REPOR_TINTERPRETATION_SYS_PROMPT_INIT,
)
from src.pkgs.models.small_expert_model import expertModel
from src.prompt.model_init import ChatMessage, DeltaMessage, callLLM
from src.test.exp.data.prompts import _auxiliary_diagnosis_judgment_repetition_prompt
from src.utils.Logger import logger
from src.utils.module import (
    InitAllResource,
    accept_stream_response,
    dumpJS,
    update_mid_vars,
)


class CustomChatModel:
    def __init__(self, gsr: InitAllResource):
        self.gsr = gsr
        self.code_func_map = {
            "blood_meas": expertModel.tool_rules_blood_pressure_level,
            "weight_meas": expertModel.fat_reduction,
            "pressure_meas": expertModel.emotions,
        }

    def __parameter_check__(self, **kwargs):
        """参数检查"""
        if "intentCode" not in kwargs:
            raise ValueError("intentCode is not in kwargs")
        if kwargs["intentCode"] not in self.code_func_map:
            raise ValueError("intentCode is not in CustomChatModel.code_func_map")
        # if not kwargs.get("history", []):
        #     raise ValueError("history is empty")

    def __extract_event_from_gsr__(
        self, gsr: InitAllResource, code: str
    ) -> Dict[str, Any]:
        """从global_share_resource中提取事件数据"""
        event = {}
        if gsr.prompt_meta_data["event"].get(code):
            event = gsr.prompt_meta_data["event"][code]
            return event
        else:
            raise ValueError(
                f"event code {code} not found in gsr.prompt_meta_data['event']"
            )

    def chat(self, **kwargs):
        """自定义对话"""
        # self.__parameter_check__(**kwargs)
        out = self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatAuxiliary(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["auxiliary_diagnosis"] = self.__chat_auxiliary_diagnosis__
        self.code_func_map["auxiliary_diagnosis_with_doctor_recommend"] = (
            self.__chat_auxiliary_diagnosis__
        )

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        try:
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            if thought_index == -1 or doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8 : doctor_index].strip()
            doctor = text[doctor_index + 8 :].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __parse_jr_response__(self, text):
        try:
            idx = text.find("\nResult:")
            if idx == -1:
                return ""
            out = text[idx:].split("\n")[0].strip()
            return out
        except Exception as err:
            logger.error(text)
            return "None"

    def __compose_auxiliary_diagnosis_message__(
        self, history: List[Dict[str, str]]
    ) -> List[DeltaMessage]:
        """组装辅助诊断消息"""
        event = self.__extract_event_from_gsr__(self.gsr, "auxiliary_diagnosis")
        sys_prompt = event["description"] + "\n" + event["process"]
        system_message = DeltaMessage(role="system", content=sys_prompt)
        messages = []
        for idx in range(len(history)):
            i = history[idx]
            if i["role"] == "assistant":
                if i["function_call"]:
                    content = f"Thought: {i['content']}\nDoctor: {i['function_call']['arguments']}"
                else:
                    content = f"Doctor: {i['content']}"
                messages.append(DeltaMessage(role="assistant", content=content))
            if i["role"] == "user":
                if idx == 0:
                    content = f"Question: {i['content']}"
                else:
                    content = f"Observation: {i['content']}"
                messages.append(DeltaMessage(role="user", content=content))
        messages = [system_message] + messages
        for idx, n in enumerate(messages):
            messages[idx] = n.dict()
        return messages

    def __chat_auxiliary_diagnosis_summary_diet_rec__(
        self, history: List[Dict]
    ) -> AnyStr:
        """辅助诊断总结、饮食建议

        Args:
            history (List[Dict]): 历史消息列表

        Returns:
            AnyStr: 辅助诊断总结、饮食建议
        """
        event = self.__extract_event_from_gsr__(
            self.gsr, "auxiliary_diagnosis_summary_diet_rec"
        )
        prompt_template_str = event["process"]
        compose_message = ""
        for i in history:
            role, content = i["role"], i["content"]
            if role == "assistant":
                compose_message += f"你: {content}\n"
            elif role == "user":
                compose_message += f"用户: {content}\n"
        prompt = prompt_template_str.replace("{MESSAGE}", compose_message)
        messages = [{"role": "user", "content": prompt}]
        chat_response = callLLM(
            model=self.gsr.model_config.get(
                "custom_chat_auxiliary_diagnosis_summary_diet_rec", "Qwen-14B-Chat"
            ),
            history=messages,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 辅助诊断总结、饮食建议 LLM Output: \n{content}")
        return content

    async def __chat_auxiliary_diagnosis__(self, **kwargs) -> ChatMessage:
        """辅助问诊"""
        # 过滤掉辅助诊断之外的历史消息
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]
        history = [
            i for i in kwargs["history"] if i.get("intentCode") == "auxiliary_diagnosis"
        ]
        messages = self.__compose_auxiliary_diagnosis_message__(history)
        logger.info(f"Custom Chat 辅助诊断 LLM Input: {dumpJS(messages)}")
        valid = True
        for _ in range(2):
            content = callLLM(
                model=model,
                history=messages,
                temperature=0,
                max_tokens=1024,
                top_p=0.8,
                n=1,
                presence_penalty=0,
                frequency_penalty=0.5,
                stop=["\nObservation:", "问诊Finished!\n\n","问诊Finished!\n"],
                stream=False,
            )

            logger.info(f"Custom Chat 辅助诊断 LLM Output: \n{content}")
            thought, doctor = self.__parse_response__(content)
            is_repeat = self.judge_repeat(history, doctor, model)
            logger.debug(f"辅助问诊 重复判断 结果: {is_repeat}")
            if is_repeat:
                valid = False
                continue
            else:
                valid = True
                break
        conts = []
        if thought == "None" or doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请您稍后再试。"
        elif not doctor:
            doctor = self.__chat_auxiliary_diagnosis_summary_diet_rec__(history)
            conts = ["请问是否需要帮您联系家庭医生?"]
        else:
            ...
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义辅助诊断对话",
        )
        return mid_vars, conts, (thought, doctor)

    def judge_repeat(self, history, content, model):
        his = [f"{i['role']}:{i['content']}" for i in history]
        his = "[" + "  ".join(his) + "]"
        judge_p = _auxiliary_diagnosis_judgment_repetition_prompt.format(his, content)
        logger.debug(f"问诊重复判断LLM输入：{judge_p}")
        h = [{"role": "user", "content": judge_p}]
        model = self.gsr.model_config["auxiliary_diagnosis_judgment_repetition"]
        content = callLLM(
            model=model,
            history=h,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            repetition_penalty=1,
            stop=["Observation", "问诊Finished!"],
            stream=False,
        )
        logger.debug(f"辅助问诊 重复判断 Output: \n{content}")
        output = self.__parse_jr_response__(content)
        if "yes" in output.lower():
            return True
        elif "no" in output.lower():
            return False
        elif (
            "没有回答" in content
            or "没有被回答" in content
            or "未回答" in content
            or "未被回答" in content
        ):
            return False
        elif "回答过" in content or "回答了" in content:
            return True
        return False

    async def chat(self, **kwargs):
        """自定义对话"""
        self.__parameter_check__(**kwargs)
        out = await self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatReportInterpretationAsk(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["report_interpretation_chat"] = (
            self.__chat_report_interpretation__
        )
        self.code_func_map["report_interpretation_chat_with_doctor_recommend"] = (
            self.__chat_report_interpretation__
        )

    def __compose_message__(
        self,
        history: List[Dict[str, str]],
        intentCode: str = "report_interpretation_chat",
        **kwargs,
    ):
        """组装消息"""
        messages = []
        if not history:
            content = kwargs["promptParam"]["report_ocr_result"]
            sysprompt = self.gsr.get_event_item("report_interpretation_chat")[
                "description"
            ]
            messages.append(
                {
                    "role": "system",
                    "content": sysprompt,
                    "intentCode": intentCode,
                }
            )
            messages.append(
                {"role": "user", "content": content, "intentCode": intentCode}
            )
        else:
            # 出现两次user的信息 == 传入报告一次 + 用户回答一次问题
            # 通过此处替换system_prompt控制问诊轮数
            if len([i for i in history if i["role"] == "user"]) >= 3:
                system_prompt = CUSTOM_CHAT_REPOR_TINTERPRETATION_SYS_PROMPT_END_SUMMARY
                history[0]["content"] = (
                    system_prompt if history[0]["role"] == "system" else ...
                )
                for idx in range(len(history)):
                    msg = history[idx]
                    if msg["role"] != "assistant":
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                    messages[-1]["intentCode"] = intentCode
            else:
                for idx in range(len(history)):
                    msg = history[idx]
                    if msg["role"] == "assistant" and msg.get("function_call"):
                        content = f"Thought: {msg['content']}\nDoctor: {msg['function_call']['arguments']}"
                        messages.append({"role": "assistant", "content": content})
                    elif msg["role"] == "assistant":
                        messages.append(
                            {"role": "assistant", "content": msg["content"]}
                        )
                    else:
                        messages.append(
                            {"role": msg["role"], "content": msg["content"]}
                        )
                    messages[-1]["intentCode"] = intentCode
        return messages

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        try:
            if text.count("Thought:") > 1:
                second_thought_index = text.find('Thought', text.find('Thought') + 1)
                text = text[second_thought_index:]
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            if thought_index == -1 or doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8 : doctor_index].strip()
            doctor = text[doctor_index + 8 :].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __chat_report_interpretation__(self, tool: str = "AskHuman", **kwargs):
        """报告解读"""
        model = self.gsr.model_config["report_interpretation_chat"]
        # model = "Qwen-72B-Chat"
        messages = self.__compose_message__(**kwargs)
        logger.info(f"Custom Chat 报告解读 LLM Input: {dumpJS(messages)}")
        content = callLLM(
            model=model,
            history=messages,
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
            # top_k=-1,
            # n=1,
            # presence_penalty=1.15,
            # frequency_penalty=2,
            # repetition_penalty=1,
            # length_penalty=1.2,
            stream=False,
        )
        # content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 报告解读 LLM Output: \n{content}")
        thought, content = self.__parse_response__(content)
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义报告解读对话",
        )
        _contents = []
        sch = -1
        if (
            kwargs["intentCode"] == "report_interpretation_chat"
            and "?" not in content
            and "？" not in content
        ):
            tool = "convComplete"
            sch = 1
            if kwargs["promptParam"]["report_type"] == "口腔报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。别担心，我已经帮你智能匹配到奉华林社区卫生服务中心口腔科的滑波医生，他可是廊坊最好的齿科医生了，并告诉了你妈妈，让她尽快带你去看医生。我还为你智能匹配了一个非常适合你的口腔保健服务包，里面有全套的牙齿问诊和保健服务。你近期一定要认真刷牙，我每天早晚会给你按时播放一个专业的刷牙视频，超级专业有趣的，我陪你一起保护牙齿！"
                ]
            elif kwargs["promptParam"]["report_type"] == "胸部报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。我已经帮你智能匹配到廊坊市人民医院呼吸内科汪医生，并告诉了你的家人，让她尽快带你去看医生。根据你的情况，我为你智能匹配了一个适合你的健康保险计划，里面包含门诊和住院绿通服务、陪诊服务。可针对常见病如肺炎、中耳炎和20种传染病可以报销。帮助守护你的健康。"
                ]
            elif kwargs["promptParam"]["report_type"] == "腹部报告":
                _contents = [
                    "健康报告显示你的健康处于平衡状态。我已经帮你智能匹配到河北中石油中心医院肝胆内科赵医生，请你尽快去看医生。根据您的情况，我为您智能匹配了一个健康体检保险计划，其中包含全面体检服务、门诊挂号和陪诊服务，可针对规定的12个项目内的检查化验项目进行门诊报销。"
                ]
        return mid_vars, messages, _contents, sch, (thought, content, tool)


class CustomChatReportInterpretationAnswer(CustomChatModel):
    session: Session = Session()

    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["report_interpretation_answer"] = (
            self.__chat_report_interpretation_answer__
        )

    def __search_docs__(
        self,
        query: str = "用户query",
        knowledge_base_name: str = "新奥百科知识库",
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> str:
        """从指定知识库搜索相关文档"""
        url = self.gsr.api_config["langchain"] + "/knowledge_base/search_docs"
        payload = dumpJS(
            {
                "query": query,
                "knowledge_base_name": knowledge_base_name,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
            ensure_ascii=True,
        )
        docs = self.session.post(url, data=payload).json()
        if docs:
            content = "## 相关知识\n"
            for doc in docs:
                content += f"{doc['page_content']}\n\n"
        else:
            content = "## 相关知识\n无"
        return content.strip()

    def __compose_message__(
        self,
        history: List[Dict[str, str]],
        intentCode: str = "report_interpretation_chat",
        **kwargs,
    ):
        """组装消息"""
        messages = []
        # 首次补充system信息
        if len(history) == 1 and history[0]["role"] == "user":
            # base_info = kwargs["promptParam"]["report_ocr_result"]
            base_info = (
                "患者姓名：张叔叔{life_entropy}\n"
                "张叔叔：年龄68岁，身高170cm，体重70kg，所患疾病：高血压；饮食喜好：喜甜\n"
                "用户：张叔叔的女儿\n"
            )
            life_entropy = kwargs["promptParam"].get("life_entropy", "60.85")
            base_info = base_info.format(life_entropy=f", 生命熵：{life_entropy}")
            external_knowledge = self.__search_docs__(query=history[-1]["content"])
            logger.debug(f"查询到知识库的内容:\n{external_knowledge}")
            system_prompt = CUSTOM_CHAT_REPOR_TINTERPRETATION_ANSWER_SYS_PROMPT.format(
                base_info=base_info, external_knowledge=external_knowledge
            )
            messages = [
                {"role": "system", "content": system_prompt, "intentCode": intentCode}
            ] + messages
        for idx in range(len(history)):
            msg = history[idx]
            if msg["role"] == "assistant" and msg.get("function_call"):
                content = f"Thought: {msg['content']}\nDoctor: {msg['function_call']['arguments']}"
                messages.append({"role": "assistant", "content": content})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages[-1]["intentCode"] = intentCode
        return messages

    def __parse_response__(self, text):
        # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
        try:
            thought_index = text.find("Thought:")
            doctor_index = text.find("\nDoctor:")
            if thought_index == -1 or doctor_index == -1:
                return "None", text
            thought = text[thought_index + 8 : doctor_index].strip()
            doctor = text[doctor_index + 8 :].strip()
            return thought, doctor
        except Exception as err:
            logger.error(text)
            return "None", text

    def __chat_report_interpretation_answer__(self, tool: str = "AskHuman", **kwargs):
        """报告解读"""
        model = self.gsr.model_config["report_interpretation_chat"]
        # model = "Qwen-72B-Chat"
        messages = self.__compose_message__(**kwargs)
        logger.info(f"Custom Chat 报告解读Answer LLM Input: {dumpJS(messages)}")
        chat_response = callLLM(
            model=model,
            history=messages,
            temperature=0.7,
            max_tokens=4096,
            top_p=0.5,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=False)
        logger.info(f"Custom Chat 报告解读Answer LLM Output: \n{content}")
        thought, content = self.__parse_response__(content)
        mid_vars = update_mid_vars(
            kwargs["mid_vars"],
            input_text=messages,
            output_text=content,
            model=model,
            key="自定义报告解读对话Answer",
        )
        _contents = []
        sch = -1
        return mid_vars, messages, _contents, sch, (thought, content, tool)
