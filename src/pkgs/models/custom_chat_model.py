# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-30 10:08:10
@desc    :   自定义对话模型
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
from typing import Any, AnyStr, Dict, List

from src.prompt.model_init import ChatMessage, DeltaMessage, callLLM
from src.test.exp.data.prompts import _auxiliary_diagnosis_judgment_repetition_prompt
from src.utils.Logger import logger
from src.utils.module import InitAllResource, accept_stream_response, dumpJS, update_mid_vars
from src.test.exp.data.prompts import _auxiliary_diagnosis_judgment_repetition_prompt
from src.pkgs.models.small_expert_model import expertModel

class CustomChatModel:
    def __init__(self, gsr: InitAllResource):
        self.gsr = gsr
        self.code_func_map = { "blood_meas": expertModel.tool_rules_blood_pressure_level, "weight_meas": expertModel.fat_reduction}

    def __parameter_check__(self, **kwargs):
        """参数检查"""
        if "intentCode" not in kwargs:
            raise ValueError("intentCode is not in kwargs")
        if kwargs["intentCode"] not in self.code_func_map:
            raise ValueError("intentCode is not in CustomChatModel.code_func_map")
        # if not kwargs.get("history", []):
        #     raise ValueError("history is empty")

    def __extract_event_from_gsr__(self, gsr: InitAllResource, code: str) -> Dict[str, Any]:
        """从global_share_resource中提取事件数据"""
        event = {}
        if gsr.prompt_meta_data["event"].get(code):
            event = gsr.prompt_meta_data["event"][code]
            return event
        else:
            raise ValueError(f"event code {code} not found in gsr.prompt_meta_data['event']")

    def chat(self, **kwargs):
        """自定义对话"""
        self.__parameter_check__(**kwargs)
        out = self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatAuxiliary(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["auxiliary_diagnosis"] = self.__chat_auxiliary_diagnosis__

    # def __init__(self, gsr: InitAllResource):
    #     self.gsr = gsr
    #     self.code_func_map = {"auxiliary_diagnosis": self.__chat_auxiliary_diagnosis__}

    # def __parameter_check__(self, **kwargs):
    #     """参数检查"""
    #     if "intentCode" not in kwargs:
    #         raise ValueError("intentCode is not in kwargs")
    #     if kwargs["intentCode"] not in self.code_func_map:
    #         raise ValueError("intentCode is not in CustomChatModel.code_func_map")
    #     if not kwargs.get("history", []):
    #         raise ValueError("history is empty")

    # def __extract_event_from_gsr__(self, gsr: InitAllResource, code: str) -> Dict[str, Any]:
    #     """从global_share_resource中提取事件数据"""
    #     event = {}
    #     if gsr.prompt_meta_data["event"].get(code):
    #         event = gsr.prompt_meta_data["event"][code]
    #         return event
    #     else:
    #         raise ValueError(f"event code {code} not found in gsr.prompt_meta_data['event']")

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
        valid = True
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
            logger.debug(f"辅助问诊 重复判断 结果: \n{is_repeat}")
            if is_repeat:
                valid = False
                continue
            else:
                valid = True
                break

        if thought == "None" or doctor == "None":
            thought = "对不起，这儿可能出现了一些问题，请您稍后再试。"
        elif not doctor or not valid:
            doctor = self.__chat_auxiliary_diagnosis_summary_diet_rec__(history)
        else:
            ...
        mid_vars = update_mid_vars(
            kwargs["mid_vars"], input_text=messages, output_text=content, model=model, key="自定义辅助诊断对话"
        )
        return mid_vars, (thought, doctor)

    def judge_repeat(self, history, content, model):
        his = [f"{i['role']}:{i['content']}" for i in history]
        his = "[" + "  ".join(his) + "]"
        judge_p = _auxiliary_diagnosis_judgment_repetition_prompt.format(his, content)
        logger.debug(f"问诊重复判断LLM输入：{judge_p}")
        h = [{"role": "user", "content": judge_p}]
        chat_response = callLLM(
            model="Qwen-72B-Chat",
            history=h,
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
        if "YES" in output:
            return True
        elif "No" in output:
            return False
        elif "没有回答" in content or "没有被回答" in content or "未回答" in content or "未被回答" in content:
            return False
        elif "回答过" in content or "回答了" in content:
            return True
        return False

    def chat(self, **kwargs):
        """自定义对话"""
        self.__parameter_check__(**kwargs)
        out = self.code_func_map[kwargs["intentCode"]](**kwargs)
        return out


class CustomChatReportInterpretation(CustomChatModel):
    def __init__(self, gsr: InitAllResource):
        super().__init__(gsr)
        self.code_func_map["report_interpretation_chat"] = self.__chat_report_interpretation__

    def __compose_message__(self, history: List[Dict[str, str]], **kwargs):
        """组装消息"""
        messages = []
        system_prompt = """【问诊和出具报告解读的提示】：
# 任务描述
你是一个经验丰富的医生,请你协助我对一份医疗检查报告的情况进行问诊
# 问诊流程专业性要求
1.我会提供我的医疗检查报告, 请你根据自身经验分析,针对我的个人情况提出相应的问题，每次最多提出两个问题
2.问题关键点可以包括: 是否出现报告中提及的疾病相关的症状，是否存在该疾病的诱因，平素生活习惯等,同类问题可以总结在一起问
3.多轮问诊最多2轮，不要打招呼，直接开始问诊，每次输出不超过200字 
4.当你收集到足够信息后，输出以下内容：概括报告中的异常点；分析导致该影响学表现的可能原因；该疾病可能的病因，可能的症状，生活注意事项、后续建议患者完善的检查项目。
5.最终输出内容要求通俗易懂、温柔亲切、符合科学性、整体字数在250字以内。
6.如果报告显示是肺部问题，可以建议进一步检查如：血常规、C反应蛋白等
如果报告是显示是胆囊炎问题，可以建议进一步检查血常规、肝功能等
 
# 已知信息
{prompt}
# 格式要求：请遵循以下格式回复
 
Thought: 思考针对当前问题应该做什么
Doctor: 你作为一个医生,分析思考的内容,提出当前想了解我的问题，不要出现序号数字
 
Begins!"""
        if not history:
            content = system_prompt.format(prompt=kwargs["promptParam"]["report_ocr_result"])
            messages.append({"role": "user", "content": content})
        else:
            for idx in range(len(history)):
                msg = history[idx]
                # if idx == 0 and msg["role"] == "user":
                #     content = system_prompt.format(prompt=msg["content"])
                #     messages.append({"role": "user", "content": content})
                if msg["role"] == "assistant" and msg["function_call"]:
                    content = f"Thought: {msg['content']}\nDoctor: {msg['function_call']['arguments']}"
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})
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

    def __chat_report_interpretation__(self, tool: str = "AskHuman", **kwargs):
        """报告解读"""
        model = self.gsr.model_config["report_interpretation_chat"]
        # model = "Qwen-72B-Chat"
        messages = self.__compose_message__(**kwargs)
        logger.info(f"Custom Chat 报告解读 LLM Input: {dumpJS(messages)}")
        chat_response = callLLM(
            model=model,
            history=messages,
            temperature=0.7,
            max_tokens=512,
            top_p=0.8,
            top_k=-1,
            n=1,
            presence_penalty=0,
            frequency_penalty=0,
            repetition_penalty=1,
            length_penalty=1,
            stream=True,
        )
        content = accept_stream_response(chat_response, verbose=True)
        logger.info(f"Custom Chat 报告解读 LLM Output: \n{content}")
        thought, content = self.__parse_response__(content)
        mid_vars = update_mid_vars(
            kwargs["mid_vars"], input_text=messages, output_text=content, model=model, key="自定义报告解读对话"
        )
        if "?" not in content and "？" not in content:
            tool = "convComplete"
        return mid_vars, messages, (thought, content, tool)