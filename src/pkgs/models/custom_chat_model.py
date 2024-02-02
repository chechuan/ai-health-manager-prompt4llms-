# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-30 10:08:10
@desc    :   自定义对话模型
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
import json
from pydoc import doc
from typing import Any, AnyStr, Dict, List
from urllib import response

from src.prompt.model_init import ChatMessage, DeltaMessage, callLLM
from src.utils.Logger import logger
from src.utils.module import InitAllResource, accept_stream_response, dumpJS, update_mid_vars
from src.test.exp.data.prompts import _auxiliary_diagnosis_judgment_repetition_prompt

class CustomChatModel:
    def __init__(self, gsr: InitAllResource):
        self.gsr = gsr
        self.code_func_map = {"auxiliary_diagnosis": self.__chat_auxiliary_diagnosis__}

    def __parameter_check__(self, **kwargs):
        """参数检查"""
        if "intentCode" not in kwargs:
            raise ValueError("intentCode is not in kwargs")
        if kwargs["intentCode"] not in self.code_func_map:
            raise ValueError("intentCode is not in CustomChatModel.code_func_map")
        if not kwargs.get("history", []):
            raise ValueError("history is empty")

    def __extract_event_from_gsr__(self, gsr: InitAllResource, code: str) -> Dict[str, Any]:
        """从global_share_resource中提取事件数据"""
        event = {}
        if gsr.prompt_meta_data["event"].get(code):
            event = gsr.prompt_meta_data["event"][code]
            return event
        else:
            raise ValueError(f"event code {code} not found in gsr.prompt_meta_data['event']")

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
            idx = text.find("\nOutput:")
            if idx == -1:
                return "None"
            out = text[idx + 8 :].split('\n')[0].strip()
            return out
        except Exception as err:
            logger.error(text)
            return "None"

    def __compose_auxiliary_diagnosis_message__(self, history: List[Dict[str, str]]) -> List[DeltaMessage]:
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

    def __chat_auxiliary_diagnosis_summary_diet_rec__(self, history: List[Dict]) -> AnyStr:
        """辅助诊断总结、饮食建议

        Args:
            history (List[Dict]): 历史消息列表

        Returns:
            AnyStr: 辅助诊断总结、饮食建议
        """
        event = self.__extract_event_from_gsr__(self.gsr, "auxiliary_diagnosis_summary_diet_rec")
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
            model=self.gsr.model_config.get("custom_chat_auxiliary_diagnosis_summary_diet_rec", "Qwen-14B-Chat"),
            history=messages,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            repetition_penalty=1,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 辅助诊断总结、饮食建议 LLM Output: \n{content}")
        return content

    def __chat_auxiliary_diagnosis__(self, **kwargs) -> ChatMessage:
        """辅助问诊"""
        # 过滤掉辅助诊断之外的历史消息
        model = self.gsr.model_config["custom_chat_auxiliary_diagnosis"]
        history = [i for i in kwargs["history"] if i.get("intentCode") == "auxiliary_diagnosis"]
        messages = self.__compose_auxiliary_diagnosis_message__(history)
        logger.info(f"Custom Chat 辅助诊断 LLM Input: {dumpJS(messages)}")
        for _ in range(2):
            chat_response = callLLM(
                model=model,
                history=messages,
                temperature=0,
                max_tokens=512,
                top_p=0.8,
                n=1,
                presence_penalty=0,
                frequency_penalty=0.5,
                repetition_penalty=1,
                stop=["Observation", "问诊Finished!"],
                stream=True,
            )
            content = accept_stream_response(chat_response, verbose=True)

            logger.info(f"Custom Chat 辅助诊断 LLM Output: \n{content}")
            thought, doctor = self.__parse_response__(content)
            is_repeat = self.judge_repeat(history, doctor, model)
            if is_repeat:
                continue
            else:
                break

        if thought == "None" or doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请您稍后再试。"
        elif not doctor:
            doctor = self.__chat_auxiliary_diagnosis_summary_diet_rec__(history)
        else:
            ...
        mid_vars = update_mid_vars(
            kwargs["mid_vars"], input_text=messages, output_text=content, model=model, key="自定义辅助诊断对话"
        )
        return mid_vars, (thought, doctor)


    def judge_repeat(self, history, content, model):
        his = json.dumps(history, ensure_ascii=False)
        judgge_p = _auxiliary_diagnosis_judgment_repetition_prompt.format(his, content)
        logger.debug(f'问诊重复判断LLM输入：{judgge_p}')
        chat_response = callLLM(
            model=model,
            prompt=judgge_p,
            temperature=0,
            max_tokens=512,
            top_p=0.8,
            n=1,
            presence_penalty=0,
            frequency_penalty=0.5,
            repetition_penalty=1,
            stop=["Observation", "问诊Finished!"],
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.debug(f"辅助问诊 重复判断 Output: \n{content}")
        output = self.__parse_jr_response__(content)
        if 'YES' in output:
            return True
        elif 'No' in output:
            return False
        return False



    def chat(self, **kwargs):
        """自定义对话"""
        self.__parameter_check__(**kwargs)
        out = self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out
