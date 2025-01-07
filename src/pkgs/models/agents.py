# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-05 15:14:07
@desc    :   专家模型 & 独立功能
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

import asyncio
import re
import sys
from os.path import basename
from pathlib import Path
from typing import AsyncGenerator, Generator
from copy import deepcopy

import json5
from fastapi.exceptions import ValidationException
from requests import Session
from langchain.prompts.prompt import PromptTemplate

sys.path.append(Path(__file__).parents[4].as_posix())
from data.jiahe_util import *
from data.test_param.test import testParam
from src.prompt.model_init import acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    construct_naive_response_generator, download_from_oss, dumpJS, param_check
)
from PIL import Image, ImageDraw, ImageFont


class Agents:
    session = Session()

    # ocr = RapidOCR()

    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr: InitAllResource = gsr
        setattr(gsr, "agents", self)
        self.regist_aigc_functions()
        self.__load_image_config__()
        self.client = openai.OpenAI()
        self.aclient = openai.AsyncOpenAI()

    def __load_image_config__(self):
        self.image_font_path = Path(__file__).parent.parent.parent.parent.joinpath(
            "data/font/simsun.ttc"
        )
        if not self.image_font_path.exists():
            logger.error(f"font file not found: {self.image_font_path}")
            exit(1)

    async def get_ocr(self, payload):
        import requests

        url = "http://10.228.67.99:26927/ocr"
        # payload = {'image_url': 'http://ai-health-manager-algorithm.oss-cn-beijing.aliyuncs.com/reportUpload/e7339bfc-3033-4200-a03f-9bc828004da3.jpg'}
        files = []
        headers = {}
        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files
        )
        return response.json()

    async def __ocr_report__(self, **kwargs):
        """报告OCR功能"""
        payload = {"image_url": kwargs.get("url", "")}
        raw_result = await self.get_ocr(payload)
        docs = ""
        if raw_result:
            process_ocr_result = [line[1] for line in raw_result]
            logger.debug(
                f"Report interpretation OCR result: {dumpJS(process_ocr_result)}"
            )
            docs += "\n".join(process_ocr_result)
        else:
            logger.error(f"Report interpretation OCR result is empty")
        return docs, raw_result

    def __report_interpretation_result__(
            self,
            ocr_result: Union[str, List[str]] = "",
            msg: str = "Unknown Error",
            report_type: str = "Unknown Type",
            remote_image_url: str = "",
    ):
        """报告解读结果

        - Args:
            ocr_result (List[str]): OCR结果
            msg (str): 报告解读内容
            report_type (str): 报告类型

        - Returns:
            Dict: 报告解读结果
        """
        return {
            "ocr_result": ocr_result,
            "report_interpretation": msg,
            "report_type": report_type,
            "image_retc": remote_image_url,
        }

    def __plot_rectangle__(self, tmp_path, file_path, rectangles_with_text):
        """为识别的报告内容画出矩形框"""
        image_io = Image.open(file_path)
        draw = ImageDraw.Draw(image_io)
        for rectangle, text in rectangles_with_text:
            line_value = int(0.002 * sum(image_io.size))
            fontsize = line_value * 6
            image_font = ImageFont.truetype(str(self.image_font_path), fontsize)
            draw.rectangle(rectangle, outline="blue", width=line_value)
            draw.text(
                (rectangle[0] - fontsize * 2, rectangle[1] - fontsize - 15),
                text,
                font=image_font,
                fill="red",
            )
        save_path = tmp_path.joinpath(file_path.stem + "_rect" + file_path.suffix)
        image_io.save(save_path)
        logger.debug(f"Plot rectangle image saved to {save_path}")
        return save_path

    def __upload_image__(self, save_path):
        """上传图片到服务器"""
        url = self.gsr.api_config["ai_backend"] + "/file/uploadFile"
        payload = {"businessType": "reportAnalysis"}
        if save_path.suffix.lower() in [".jpg", ".jpeg"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/jpeg"))]
        elif save_path.suffix.lower() in [".png"]:
            files = [("file", (save_path.name, open(save_path, "rb"), "image/png"))]
        else:
            files = [
                (
                    "file",
                    (
                        save_path.name,
                        open(save_path, "rb"),
                        f"image/{save_path.suffix.lower()[1:]}",
                    ),
                )
            ]
        resp = self.session.post(url, data=payload, files=files)
        if resp.status_code == 200:
            remote_image_url = resp.json()["data"]
        else:
            logger.error(f"Upload image error: {resp.text}")
            remote_image_url = ""
        return remote_image_url

    async def __report_ocr_classification_make_text_group__(
            self, file_path: Union[str, Path], raw_result, tmp_path, **kwargs
    ) -> str:
        """报告OCR结果分组"""

        # "3. 可选类别有[报告标题,基础信息,影像图片,影像所见,诊断意见,医疗建议,检查方法,检查医生]\n"
        sysprompt = (
            "You are a helpful assistant.\n"
            "# 任务描述\n"
            "1. 下面我将给你报告OCR提取的内容，它是有序的，优先从上到下从左到右\n"
            "2. 请你参考给出的内容的前后信息，按内容的前后顺序对报告的内容进行归类，类别最多5个\n"
            "3. 只给出各类别开始内容和结尾内容对应的index, 所有内容的index都应当被包含\n"
            "# 输出格式要求\n"
            "## 格式参考\n"
            "```json\n"
            '{"分类1": [start_idx_1, end_idx_1], "分类2": [start_idx_2, end_idx_2], "分类3": [start_idx_3, end_idx_3],...}\n'
            "```\n"
            "## 特殊要求\n"
            "其中start_idx_2=end_idx_1+1, start_idx_3=end_idx_2+1\n"
        )
        content_index = {
            idx: text for idx, text in enumerate([i[1] for i in raw_result])
        }
        messages = [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": "```json\n" + str(content_index) + "\n```"},
        ]
        model = self.gsr.get_model("aigc_functions_report_interpretation_text_classify")

        logger.debug(f"报告解读文本分组 LLM Input:\n{dumpJS(messages)}")
        response = await self.aclient.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
        )
        content = response.choices[0].message.content
        logger.debug(f"报告解读文本分组 LLM Output: {content}")

        content = re.findall("```json(.*?)```", content, re.S)[0].strip()
        try:
            loc = json.loads(content)
        except:
            loc = {}
        try:
            rectangles_with_text = []
            for topic, index_range in loc.items():
                if index_range[0] > index_range[1]:
                    index_range[0], index_range[1] = index_range[1], index_range[0]
                if index_range[0] < 0:
                    index_range[0] = 0
                elif index_range[0] > len(content_index):
                    continue
                if index_range[1] >= len(content_index):
                    index_range[1] = len(content_index) - 1
                msgs = raw_result[index_range[0]: index_range[1] + 1]
                coordinates = [item[0] for item in msgs]
                left = min([j for i in coordinates for j in [i[0][0], i[3][0]]])
                top = min([j for i in coordinates for j in [i[0][1], i[1][1]]])
                right = max([j for i in coordinates for j in [i[1][0], i[2][0]]])
                bottom = max([j for i in coordinates for j in [i[2][1], i[3][1]]])
                rectangles_with_text.append(((left, top, right, bottom), topic))
            save_path = self.__plot_rectangle__(
                tmp_path, file_path, rectangles_with_text
            )
        except Exception as e:
            logger.exception(f"Report interpretation error: {e}")
            pass
        remote_image_url = self.__upload_image__(save_path)
        return remote_image_url

    def __compose_user_msg__(
            self,
            mode: Literal[
                "user_profile",
                "messages",
                "drug_plan",
                "medical_records",
                "ietary_guidelines",
                "key_indicators",
                "group_report",
                "interaction_data",
            ],
            user_profile: UserProfile = None,
            medical_records: MedicalRecords = None,
            ietary_guidelines: DietaryGuidelinesDetails = None,
            messages: List[ChatMessage] = [],
            key_indicators: "List[KeyIndicators]" = "[]",
            drug_plan: "List[DrugPlanItem]" = "[]",
            role_map: Dict = {},
    ) -> str:
        content = ""
        if mode == "user_profile":
            if user_profile:
                for key, value in user_profile.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, Union[float, int, str]) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "messages":
            assert messages is not None, "messages can't be None"
            assert messages is not [], "messages can't be empty list"
            role_map = (
                {"assistant": "医生", "user": "患者"} if not role_map else role_map
            )
            for message in messages:
                if message.get("role", "other") == "other":
                    content += f"other: {message['content']}\n"
                elif role_map.get(message.get("role", "other")):
                    content += f"{role_map[message['role']]}: {message['content']}\n"
                else:
                    content += f"{message['content']}\n"
        elif mode == "drug_plan":
            if drug_plan:
                for item in json5.loads(drug_plan):
                    content += (
                            ", ".join(
                                [
                                    f"{USER_PROFILE_KEY_MAP.get(k)}: {v}"
                                    for k, v in item.items()
                                ]
                            )
                            + "\n"
                    )
                content = content.strip()
        elif mode == "medical_records":
            if medical_records:
                for key, value in medical_records.items():
                    if value and USER_PROFILE_KEY_MAP.get(key):
                        content += f"{USER_PROFILE_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "ietary_guidelines":
            if ietary_guidelines:
                for key, value in ietary_guidelines.items():
                    if value and DIETARY_GUIDELINES_KEY_MAP.get(key):
                        content += f"{DIETARY_GUIDELINES_KEY_MAP[key]}: {value if isinstance(value, (float, int, str)) else json.dumps(value, ensure_ascii=False)}\n"
        elif mode == "key_indicators":
            # 创建一个字典来存储按日期聚合的数据
            aggregated_data = {}

            # 遍历数据并聚合
            for item in key_indicators:
                for entry in item["data"]:
                    date, time = entry["time"].split(" ")
                    value = entry["value"]
                    if date not in aggregated_data:
                        aggregated_data[date] = {}
                    aggregated_data[date][item["key"]] = {"time": time, "value": value}

            # 创建 Markdown 表格
            content = "| 测量日期 | 测量时间 | 体重 | BMI | 体脂率 |\n"
            content += "| ------ | ------ | ---- | ----- | ------ |\n"

            # 填充表格
            for date, measurements in aggregated_data.items():
                time = measurements.get("体重", {}).get("time", "")
                weight = measurements.get("体重", {}).get("value", "")
                bmi = measurements.get("bmi", {}).get("value", "")
                body_fat_rate = measurements.get("体脂率", {}).get("value", "")
                row = f"| {date} | {time} | {weight} | {bmi} | {body_fat_rate} |\n"
                content += row
        elif mode == "group_report":
            if user_profile:
                for key, value in user_profile.items():
                    if value and GROUP_REPORT_MAPPING.get(key):
                        content += f"{GROUP_REPORT_MAPPING[key]}: {value}\n"
        elif mode == "interaction_data":
            if user_profile:
                for key, value in user_profile.items():
                    if value and INTERACTION_DATA_MAPPING.get(key):
                        content += f"{INTERACTION_DATA_MAPPING[key]}: {value}\n"
        else:
            logger.error(f"Compose user profile error: mode {mode} not supported")
        return content

    async def __update_model_args__(self, kwargs, **args) -> Dict:
        if "model_args" in kwargs:
            if kwargs.get("model_args"):
                args = {
                    **args,
                    **kwargs["model_args"],
                }
            del kwargs["model_args"]
        return args

    def regist_aigc_functions(self) -> None:
        self.funcmap = {
            v: getattr(self, v) for k, v in self.gsr.intent_aigcfunc_map.items()
        }
        for obj_str in dir(self):
            if (
                    obj_str.startswith("aigc_functions_") or obj_str.startswith("sanji_")
            ) and not self.funcmap.get(obj_str):
                self.funcmap[obj_str] = getattr(self, obj_str)

    def aigc_functions_single_choice(self, prompt: str, options: List[str], **kwargs):
        """单项选择功能

        - Args:
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        model = self.gsr.model_config.get(
            "aigc_functions_single_choice", "Qwen-14B-Chat"
        )
        prompt_template_str = self.gsr.prompt_meta_data["event"][
            "aigc_functions_single_choice"
        ]["description"]
        prompt_template = PromptTemplate.from_template(prompt_template_str)
        query = prompt_template.format(options=options, prompt=prompt)
        messages = [{"role": "user", "content": query}]
        logger.debug(
            f"Single choice LLM Input: {json.dumps(messages, ensure_ascii=False)}"
        )
        content = callLLM(
            history=messages,
            model=model,
            temperature=0,
            repetition_penalty=1.0,
        )
        logger.debug(f"Single choice LLM Output: {content}")
        if content == "选项与要求不符":
            return content
        else:
            if content not in options:
                logger.error(f"Single choice error: {content} not in options")
                return "选项与要求不符"
        return content

    async def aigc_functions_report_interpretation(
            self, options: List[str] = ["口腔报告", "胸部报告", "腹部报告"], **kwargs
    ) -> Dict:
        """报告解读功能

        - Args:
            file_path (str): 报告文件路径

        - Returns:
            str: 报告解读内容
        """

        def prepare_file(**kwargs):

            file_path = None
            image_url = kwargs.get("url")

            if not tmp_path.exists():
                tmp_path.mkdir(parents=True)

            if image_url:
                logger.info(f"start: image_url")
                r = self.session.get(image_url)
                logger.info(f"stop: image_url")
                file_path = tmp_path.joinpath(basename(image_url))
                logger.info(f"start2: image_url")
                with open(file_path, mode="wb") as f:
                    f.write(r.content)
                logger.info(f"stop2: image_url")
            elif kwargs.get("file_path"):
                file_path = kwargs.get("file_path")
                image_url = self.__upload_image__(file_path)
            else:
                logger.error(f"Report interpretation error: file_path or url not found")
            return image_url, file_path

        async def jude_report_type(docs: str, options: List[str]) -> str:
            query = f"{docs}\n\n请你判断以上报告属于哪个类型,从给出的选项中选择: {options}, 要求只输出选项答案, 请不要输出其他内容\n\nOutput:"
            messages = [{"role": "user", "content": query}]
            report_type = callLLM(
                history=messages, model="Qwen1.5-72B-Chat", temperature=0.7, top_p=0.5
            )
            logger.debug(f"Report interpretation report type: {report_type}")
            if report_type not in options:
                if "口腔" in docs and "口腔报告" in options:
                    report_type = "口腔报告"
                elif "胸部" in docs and "胸部报告" in options:
                    report_type = "胸部报告"
                elif "腹部" in docs and "腹部报告" in options:
                    report_type = "腹部报告"
            if report_type not in options:
                report_type = "其他"
            return report_type

        tmp_path = Path(f".tmp/images")
        image_url, file_path = prepare_file(**kwargs)
        if not file_path:
            return self.__report_interpretation_result__(msg="请输入信息源")
        docs, raw_result = await self.__ocr_report__(**kwargs)
        if not docs:
            return self.__report_interpretation_result__(
                msg="未识别出报告内容，请重新尝试",
                ocr_result="您的报告内容无法解析，请重新尝试.",
            )
        try:
            remote_image_url = await self.__report_ocr_classification_make_text_group__(
                file_path, raw_result, tmp_path
            )
            if not remote_image_url:
                remote_image_url = image_url
        except Exception as e:
            logger.exception(f"Report interpretation error: {e}")
            remote_image_url = image_url
        # 报告异常信息解读
        # prompt_template_str = "You are a helpful assistant.\n" "# 任务描述\n" "请你为我解读报告中的异常信息"
        # messages = [{"role": "system", "content": prompt_template_str}, {"role": "user", "content": docs}]
        # logger.debug(f"Report interpretation LLM Input: {dumpJS(messages)}")
        # response = callLLM(history=messages, model="Qwen-14B-Chat", temperature=0.7, top_p=0.5, stream=True)
        # content = accept_stream_response(response, verbose=False)
        # logger.debug(f"Report interpretation LLM Output: {content}")

        # 增加报告类型判断
        if options:
            report_type = await jude_report_type(docs, options)
        else:
            report_type = "其他"
        return self.__report_interpretation_result__(
            ocr_result=docs, report_type=report_type, remote_image_url=remote_image_url
        )

    async def aigc_functions_report_summary(self, **kwargs):
        """报告内容总结
        循环
        """
        chunk_size = kwargs.get("chunk_size", 1000)
        assert kwargs["report_content"] is not None, "report_content is None"
        system_prompt = (
            "You are a helpful assistant.\n"
            "# 任务描述\n"
            "你是一个经验丰富的医生,你要清楚了解患者的生命熵检查报告内容，根据报告内容给出总结话术，其中包含生命熵熵值，哪些处于失衡状态，哪些有异常\n"
            "1. 请你根据自身经验，结合生命熵报告内容，给出摘要总结\n"
            "2. 请你根据总结，给出建议，并说明原因\n"
        )
        summary_list = []
        if isinstance(kwargs["report_content"], list):
            report_content = "\n".join(kwargs["report_content"])
        else:
            report_content = kwargs["report_content"]
        for i in range(0, len(report_content), chunk_size):
            chunk_text = kwargs["report_content"][i: i + chunk_size]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text},
            ]
            content = await acallLLM(
                history=messages,
                model="Qwen-14B-Chat",
                temperature=0.7,
                top_p=0.8,
            )
            summary_list.append(content)
        summary = "\n".join(summary_list)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary},
        ]
        content: str = await acallLLM(
            history=messages,
            model="Qwen1.5-72B-Chat",
            temperature=0.7,
            top_p=0.8,
        )
        return {"report_summary": content}

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary(self, **kwargs) -> str:
        """问诊摘要"""

        _event = "问诊摘要"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary_to_group(self, **kwargs) -> str:
        """问诊摘要"""
        _event = "问诊摘要"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_consultation_summary_chief_disease(self, **kwargs) -> str:
        """问诊摘要"""
        _event = "问诊摘要-主诉/现病史"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {"user_profile": user_profile, "messages": messages}

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.8
        )
        content: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    @param_check(check_params=["messages", "user_profile"])
    async def aigc_functions_diagnosis(self, **kwargs) -> str:
        """诊断"""
        _event = "诊断"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        model_args = await self.__update_model_args__(
            kwargs, temperature=0, top_p=0.8, repetition_penalty=1.0
        )

        prompt_vars = {"user_profile": user_profile, "messages": messages}
        # 诊断1阶段必须直接返回字符串用于判断下一步逻辑
        content: str = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args={
                "temperature": 0.7,
                "top_p": 0.8,
                "repetition_penalty": 1.0,
            },
            **kwargs,
        )

        if content == "无":
            kwargs["intentCode"] = "aigc_functions_diagnosis_result"
            content: str = await self.aaigc_functions_general(
                _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
            )
        else:
            if model_args.get("stream") is True:
                content: AsyncGenerator = construct_naive_response_generator(content)
        return content

    @param_check(check_params=["messages"])
    async def aigc_functions_drug_recommendation(self, **kwargs) -> List[Dict]:
        """用药建议"""
        _event = "用药建议"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0, top_p=0.9, repetition_penalty=1.0
        )
        response: Union[str, AsyncGenerator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        if isinstance(response, openai.AsyncStream):
            return response
        try:
            content = json5.loads(response)
        except Exception as e:
            try:
                content = re.findall("```json(.*?)```", response, re.DOTALL)[0]
                content = dumpJS(json5.loads(content))
            except Exception as e:
                logger.error(f"AIGC Functions {_event} json5.loads error: {e}")
                content = dumpJS([])
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_food_principle(self, **kwargs) -> str:
        """饮食原则"""
        _event = "饮食原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sport_principle(self, **kwargs) -> str:
        """运动原则"""
        _event = "运动原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_mental_principle(self, **kwargs) -> str:
        """情志原则"""

        _event = "情志原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_chinese_therapy(self, **kwargs) -> str:
        """中医调理"""
        _event = "中医调理"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(kwargs)
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return content

    async def aigc_functions_food_principle_new(self, **kwargs) -> str:
        """饮食原则"""
        _event = "饮食原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "symptom": kwargs.get("symptom", "无"),

        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     key, values = line.split('：', 1)
        #     if values=='无':
        #         data[key]=[]
        #     else:
        #         data[key] = values
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_sport_principle_new(self, **kwargs) -> str:
        """运动原则"""
        _event = "运动原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "symptom": kwargs.get("symptom", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     if '：' in line:
        #         key, values = line.split('：', 1)
        #         if values=='无':
        #             data[key]=[]
        #         else:
        #             data[key] = values
        #     else:
        #         data['运动课程']=line
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_mental_principle_new(self, **kwargs) -> str:
        """情志原则"""

        _event = "情志原则"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "symptom": kwargs.get("symptom", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # res = json5.loads(content)
        # _content = "\n".join([i["title"]+":" + i["content"] for i in res])
        # filtered_string = content.replace('1.','').replace('2.','').replace('3.','')
        return content

    # @param_check(check_params=["messages"])
    async def aigc_functions_chinese_therapy_new(self, **kwargs) -> str:
        """中医调理"""
        _event = "中医调理"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "symptom": kwargs.get("symptom", "无"),
        }
        model_args = await self.__update_model_args__(kwargs)
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        # data = {}
        # lines = content.split('\n')
        # for line in lines:
        #     if len(line)>0:
        #         key, values = line.split('：', 1)
        #         if values=='无':
        #             data[key]=[]
        #         else:
        #             data[key] = values
        return content

    async def aigc_functions_auxiliary_history_talking(self, **kwargs: object):
        """医生端 - 生成问题"""
        _event = "生成医生问题"

        def __fmt_history(_messages):
            _role_map = {"user": "用户", "assistant": "医生"}
            _tmp_lst = []
            for item in _messages:
                role = _role_map.get(item["role"], "用户")
                content = item["content"]
                _tmp_lst.append(f"{role}: {content}")
            return "\n".join(_tmp_lst)

        duplicate_content = {}
        _messages = []
        for item in kwargs["messages"]:
            _content = item["content"]
            if not duplicate_content.get(_content):
                _messages.append(item)
                duplicate_content[_content] = 1
        history_str = __fmt_history(_messages)
        prompt = (
            f"# 用户与医生的对话记录\n{history_str}\n"
            "# 角色定位\n"
            "请你扮演一个经验丰富的医生,协助我为患者的疾病进行问诊\n"
            "# 任务描述\n"
            "1. 在多轮的对话中会提供患者的个人信息和感受,请你根据自身经验分析,针对个人情况提出相应的 问题\n"
            "2. 问题关键点可以包括:持续时间、发生时机、诱因或症状发生部位等\n"
            "3. 不要重复询问同一个问题,问题尽可能简洁,每次最多提出两个问题\n"
            "4. 纯净模式，只输出要询问患者的问题，不同的问题用｜｜隔开\n"
            "5. 输出示例如下\n"
            "5. 您的发热情况如何，如否有测量体温？｜｜除了嗓子痛和痒，是否有咳嗽或者喉咙有异物感？\n"
        )

        # 打印拼接后的 prompt
        logger.info(f"AIGC Functions {_event} Assembled Prompt: \n{prompt}")

        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )

        content: Union[str, Generator] = await acallLLM(
            model="Qwen1.5-72B-Chat",
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: \n{content}")
        # 使用正则表达式找到所有的句子边界（句号或问号）
        try:
            sentences = content.split("｜｜", 1)
        except Exception as err:
            logger.error(err)
            sentences = [content]
        return sentences

    async def aigc_functions_auxiliary_diagnosis(self, **kwargs):
        prompt_template = "# 患者与医生历史会话信息\n{history_str}\n\n"
        user_input = (
            "# 任务描述\n"
            "请你扮演一个经验丰富的医生,协助我进行疾病的诊断,"
            "根据患者与医生的历史会话信息,输出若干个患者最多5个可能的诊断以及对应的概率值\n"
            "格式参考: 疾病-概率,疾病-概率, 以`,`分隔\n"
            "只输出`疾病`-`概率`,避免输入任何其他内容，一定按照下面例子输出\n"
            "例子如下\n"
            "头痛-30%, 咽喉炎-25%, 鼻窦炎-20%, 胃炎-15%, 咳嗽-10%\n"
        )

        # 去重过滤输入内容
        duplicate_messages, _messages = {}, []
        for item in kwargs["messages"]:
            if not duplicate_messages.get(item["content"]):
                _messages.append(item)
                duplicate_messages[item["content"]] = 1

        history_str = "\n".join([f"{item['content']}" for item in _messages])
        prompt = prompt_template.format(history_str=history_str)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        # 日志记录输入
        logger.debug(f"Formatted Prompt: {repr(prompt)}")
        logger.debug(f"AIGC Functions Auxiliary Diagnosis LLM Input: {repr(messages)}")

        # 获取模型参数
        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )

        try:
            # 调用模型
            d_p_pair_str = await acallLLM(
                model="Qwen1.5-72B-Chat",
                query=messages,
                **model_args,
            )
            logger.info(f"AIGC Functions Auxiliary Diagnosis LLM Output: {repr(d_p_pair_str)}")

            # 数据解析，清理多余字符
            d_p_pair = []
            for item in d_p_pair_str.split(","):
                try:
                    if "-" in item:
                        disease, probability = item.split("-", 1)
                        disease = disease.strip()
                        probability = probability.strip().replace("%", "")

                        # 过滤多余的文字和换行符
                        if "\n" in disease:
                            disease = disease.split("\n")[-1].strip()
                        if "。" in probability:
                            probability = probability.split("。")[0].strip()

                        # 验证并格式化
                        if disease and probability.isdigit():
                            d_p_pair.append({"name": disease, "prob": probability + "%"})
                except Exception as parse_error:
                    logger.warning(f"Error parsing diagnosis item: {repr(item)}, Error: {repr(parse_error)}")

        except Exception as err:
            logger.error(f"Error during LLM processing: {repr(err)}")
            d_p_pair = []

        # 返回标准化的结果
        response = d_p_pair
        logger.info(f"Final Response: {repr(response)}")
        return response

    async def aigc_functions_relevant_inspection(self, **kwargs):
        import logging

        # 初始化日志器
        logger = logging.getLogger("aigc_inspection")
        logging.basicConfig(level=logging.INFO)

        prompt_template = (
            "# 患者与医生历史会话信息\n{history_str}\n\n"
            "# 任务描述\n"
            "你是一个经验丰富的医生,请你协助我进行疾病的鉴别诊断,输出建议我做的临床辅助检查项目\n"
            "1. 请你根据历史会话信息、初步诊断的结果、鉴别诊断的知识、分析我的疾病，进一步输出能够让我确诊的临床检查项目\n"
            "2. 只输出检查项目的名称，不要其他的内容\n"
            "3. 不同检查项目名称之间用`,`隔开,检查项目不要重复\n\n"
            "# 初步诊断结果\n{diagnosis_str}\n\n"
            "Begins!"
        )

        duplicate_content = {}
        _messages = []
        for item in kwargs["messages"]:
            _content = item["content"]
            if not duplicate_content.get(_content):
                _messages.append(item)
                duplicate_content[_content] = 1

        rolemap: Dict[str, str] = {"user": "患者", "assistant": "医生"}
        history_str = "\n".join(
            [f"{rolemap[item['role']]}: {item['content']}" for item in _messages]
        )
        diagnosis_str = ",".join([i["name"] for i in kwargs["diagnosis"]])
        prompt = prompt_template.format(
            history_str=history_str, diagnosis_str=diagnosis_str
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_str},
        ]

        # 记录输入日志
        logger.info(f"Generated Prompt: {prompt}")
        model_args = await self.__update_model_args__(kwargs)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        content = await acallLLM(
            model="Qwen1.5-72B-Chat",
            query=messages,
            **model_args,
        )

        # 记录模型输出
        logger.info(f"Model Raw Output: {content}")

        # 去除前缀和多余内容
        clean_output = [
            i.strip().removeprefix("医生: ").strip() for i in content.split(",")
        ]
        logger.info(f"Cleaned Output: {clean_output}")

        return clean_output

    @param_check(check_params=["messages"])
    async def aigc_functions_reason_for_care_plan(self, **kwargs) -> str:
        """康养方案推荐原因"""
        _event = "康养方案推荐原因"
        user_profile = self.__compose_user_msg__(
            "user_profile",
            user_profile=kwargs["user_profile"],
        )
        messages = self.__compose_user_msg__("messages", messages=kwargs["messages"])
        drug_plan = self.__compose_user_msg__(
            "drug_plan", drug_plan=kwargs.get("drug_plan", "")
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "drug_plan": drug_plan,
            "diagnosis": kwargs.get("diagnosis", "无"),
            "food_principle": kwargs.get("food_principle", ""),
            "sport_principle": kwargs.get("sport_principle", ""),
            "mental_principle": kwargs["mental_principle"],
            "chinese_therapy": kwargs["chinese_therapy"],
        }
        model_args = await self.__update_model_args__(kwargs, temperature=0.7, top_p=1)
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        return response

    async def sanji_questions_generate(self, **kwargs) -> str:
        _event = "猜你想问"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )

        prompt_vars = {
            "user_profile": user_profile,
            "content_all": kwargs["content_all"],
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        result = content.split('\n')
        result_ = []
        for i in result:
            i = i.replace('-', '')
            question_marks = i.count('？')
            if question_marks > 1:
                last_question_mark_index = i.rfind('？')
                j = i[:last_question_mark_index].replace('？', '。') + i[last_question_mark_index:]
            else:
                j = i
            result_.append(j)
        return {'ques': result_}

    async def sanji_assess_3d_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_3d_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        content = content.replace("：\n", "：")
        content = content.replace("：\n\n", "\n")
        data = {}
        lines = content.split("\n")
        for line in lines:
            if "：" in line:
                key, values = line.split("：", 1)
                if values == "无":
                    data[key] = []
                else:
                    data[key] = values.split("|")
        return data

    async def sanji_assess_keyword_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_keyword_cl"
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "messages": messages,
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0, max_tokens=8192
        )
        content: str = await self.sanji_general(
            # process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        # content = content.replace("：\n", "：")
        # lines = content.split("\n")
        # data = {}
        # for line in lines:
        #     if "：" in line:
        #         key, values = line.split("：", 1)
        #         if values == "无":
        #             data[key] = []
        #         else:
        #             data[key] = values.split(", ")
        content = content.split("||")
        data = {}
        data["one"] = content
        return data

    async def sanji_assess_3health_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_3health_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "symptom": kwargs.get("symptom", "")
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )
        content = content.replace("：\n", "：")
        lines = content.split("\n")
        data = {}
        for line in lines:
            if "：" in line:
                key, values = line.split("：", 1)
                if values == "无;" or values == ";":
                    data[key] = []
                else:
                    data[key] = [values]
        return data

    async def sanji_assess_literature_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_liter_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "diagnosis": kwargs.get("diagnosis", "无"),
        }
        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            _event=_event, prompt_vars=prompt_vars, model_args=model_args, **kwargs
        )
        if "\n" in content:
            lines = content.split("\n")
        else:
            lines = content.split("||")
        result = {"one": lines}

        return result

    async def sanji_intervene_goal_classification(self, **kwargs) -> str:
        """"""

        _event = "sanji_intervene_cl"
        user_profile: str = self.__compose_user_msg__(
            "user_profile", user_profile=kwargs["user_profile"]
        )
        messages = (
            self.__compose_user_msg__("messages", messages=kwargs["messages"])
            if kwargs.get("messages")
            else ""
        )
        prompt_vars = {
            "user_profile": user_profile,
            "messages": messages,
            "symptom": kwargs.get("symptom", "")
        }

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=0.3, repetition_penalty=1.0
        )
        content: str = await self.sanji_general(
            process=0,
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            **kwargs,
        )

        data = {}
        data["goal"] = {}
        data["literature"] = {}

        content = content.replace("：\n", "：")
        # if '-' in content:
        #     lines = content.split("\n\n")
        # else:
        lines = content.split("\n")
        for line in lines:
            if ":" in line or "：" in line:
                key, values = line.split("：", 1)
                if values == "无":
                    data["goal"][key] = []
                else:
                    my_list = values.split("||")
                    filtered_list = [item for item in my_list if item]
                    data["goal"][key] = filtered_list

        return data

    @param_check(check_params=["plan_ai", "plan_human"])
    async def aigc_functions_plan_difference_finder(
            self, **kwargs
    ) -> Union[str, Generator]:
        """差异点发现"""
        _event = "差异点发现"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_plan_difference_analysis(
            self, **kwargs
    ) -> Union[str, Generator]:
        """差异能力分析"""
        _event = "差异能力分析"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_plan_adjustment_suggestion(
            self, **kwargs
    ) -> Union[str, Generator]:
        """方案调整建议生成"""
        _event = "方案调整建议生成"
        prompt_template = (
            "You are a helpful assistant.\n"
            "# ai方案\n{plan_ai}\n"
            "# 人工方案\n{plan_human}\n"
            "# 任务描述\n"
            "请你分析以上两个方案的差异点"
        )
        prompt_vars = {"plan_ai": kwargs["plan_ai"], "plan_human": kwargs["plan_human"]}
        model_args = {"temperature": 1, "top_p": 0.8}
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            model_args=model_args,
            prompt_template=prompt_template,
            **kwargs,
        )
        return response

    async def aigc_functions_doctor_recommend(self, **kwargs) -> Union[str, Generator]:
        async def download_and_load_doctor_messages() -> List[Dict]:
            doctor_example_path = Path(".cache/doctor_examples.json")
            if not doctor_example_path.exists():
                download_from_oss(
                    "ai-algorithm-nlp/intelligent-health-manager/data/docter_examples.json",
                    doctor_example_path,
                )
            doctor_examples = json.load(open(".cache/doctor_examples.json", "r"))
            return doctor_examples

        if kwargs.get("model_args") and kwargs["model_args"].get("stream") is True:
            raise ValidationException("医生推荐 model_args.stream can't be True")

        user_demands = self.__compose_user_msg__(
            "messages",
            messages=kwargs["messages"],
            role_map={"assistant": "助手", "user": "用户"},
        )

        # 先判断用户是否需要医生推荐
        prompt_template_assert = f"请你帮我判断用户是否需要推荐医生,需要:`Yes`, 不需要:`No`\n用户:{user_demands}"
        if_need = self.aigc_functions_single_choice(
            prompt_template_assert, options=["Yes", "No"]
        )
        if if_need.lower() != "yes":
            return []

        _event = "医生推荐"

        prompt_template = (
            "# 已知信息\n"
            "1.问诊结果：{diagnosis_result}\n"
            "2.我的诉求：{user_demands}\n"
            "# 医生信息\n"
            "{doctor_message}\n\n"
            "# 任务描述\n"
            "你是一位经验丰富的智能健康助手，请你根据输出要求、我的诉求、已知信息，为我推荐符合我病情的医生。\n"
            "# 输出要求\n"
            "1.根据已知信息、我对医生的诉求、医生信息列表，帮我推荐最符合我情况的备选5个医生信息\n"
            "2.你推荐我的医生，第一需求应该符合我的疾病诊断或者检查检验报告结论\n"
            "3.其他需求你要考虑我对医生擅长领域的需求，我对医生性别的需求等\n"
            "4.推荐医生的顺序按照符合我条件的优先级前后展示，输出格式参考：医生名称1，医生名称2，以`,`隔开\n"
            "5.记住只输出医生名字，别输出多余的其他描述内容就行"
            "Begins~"
        )

        # TODO 从外部加载医生数据
        if not hasattr(self, "docter_message"):
            doctor_examples = await download_and_load_doctor_messages()
            self.docter_message = "\n\n".join(
                [DoctorInfo(**i).__str__() for i in doctor_examples]
            )

        prompt_vars = {
            "doctor_message": self.docter_message,
            "diagnosis_result": kwargs.get("prompt", ""),
            "user_demands": user_demands,
        }
        model_args = await self.__update_model_args__(kwargs, temperature=1, top_p=0.8)
        # kwargs["messages"] = [
        #     {
        #         "role": "user",
        #         "content": "请你综合考虑医生的专业匹配度、医生职称、医生工作年限等信息来帮我推荐医生,,
        #     }
        # ]
        response: Union[str, Generator] = await self.aaigc_functions_general(
            _event=_event,
            prompt_vars=prompt_vars,
            prompt_template=prompt_template,
            model_args=model_args,
            **kwargs,
        )
        try:
            # raise AssertionError("未定义err")
            # result = [i.strip() for i in response.split(",")]
            result = [i.strip() for i in re.split(r"[,|，]", response)]
        except Exception as err:
            logger.error(repr(err))
            result = err
        return result

    def aigc_functions_general(
            self,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: dict = {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            },
            **kwargs,
    ) -> str:
        """通用生成"""
        event = kwargs.get("intentCode")
        model = self.gsr.get_model(event)
        prompt_template: str = self.gsr.get_event_item(event)["description"]
        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")
        content: str = callLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def aaigc_functions_general(
            self,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: Dict = {},
            prompt_template: str = "",
            **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""
        event = kwargs.get("intentCode")
        model = self.gsr.get_model(event)
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 1,
                "repetition_penalty": 1.0,
            }
            if not model_args
            else model_args
        )
        prompt_template: str = (
            prompt_template
            if prompt_template
            else self.gsr.get_event_item(event)["description"]
        )
        logger.debug(f"Prompt Vars Before Formatting: {repr(prompt_vars)}")

        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")

        content: Union[str, Generator] = await acallLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def sanji_general(
            self,
            process: int = 1,
            _event: str = "",
            prompt_vars: dict = {},
            model_args: Dict = {},
            prompt_template: str = "",
            **kwargs,
    ) -> Union[str, Generator]:
        """通用生成"""
        event = kwargs.get("intentCode")
        model = "Qwen1.5-32B-Chat"
        model_args: dict = (
            {
                "temperature": 0,
                "top_p": 0.3,
                "repetition_penalty": 1.0,
                "max_tokens": 12000
            }
            if not model_args
            else model_args
        )
        des = self.gsr.get_event_item(event)["description"]
        if process == 2:
            des = (
                    self.gsr.get_event_item(event)["process"]
                    + self.gsr.get_event_item(event)["constraint"]
            )
        if process == 0:
            des += self.gsr.get_event_item(event)["constraint"]
        prompt_template: str = prompt_template if prompt_template else des
        prompt = prompt_template.format(**prompt_vars)
        logger.debug(f"AIGC Functions {_event} LLM Input: {repr(prompt)}")
        content: Union[str, Generator] = await acallLLM(
            model=model,
            query=prompt,
            **model_args,
        )
        if isinstance(content, str):
            logger.info(f"AIGC Functions {_event} LLM Output: {repr(content)}")
        return content

    async def __preprocess_function_args__(self, kwargs) -> dict:
        """处理aigc functions入参"""
        if not kwargs.get("model_args"):
            kwargs["model_args"] = {}
        if not kwargs.get("user_profile"):
            kwargs["user_profile"] = {}
        if not kwargs.get("diagnosis"):
            kwargs["diagnosis"] = ""
        return kwargs

    async def call_function(self, **kwargs) -> Union[str, Generator]:
        """调用函数
        - Args:
            intentCode (str): 意图代码
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            str: 答案
        """
        intent_code = kwargs.get("intentCode")
        # TODO intentCode -> funcCode
        intent_code = (
            self.gsr.intent_aigcfunc_map.get(intent_code)
            if self.gsr.intent_aigcfunc_map.get(intent_code)
            else intent_code
        )
        if not self.funcmap.get(intent_code):
            logger.error(f"intentCode {intent_code} not found in funcmap")
            raise RuntimeError(f"Code not supported.")

        # # 检查是否为特定的并行化意图代码
        # if intent_code in ["aigc_functions_meal_plan_generation"]:
        #     return await self.handle_parallel_intent(**kwargs)

        # kwargs = await self.__preprocess_function_args__(kwargs)
        try:
            func = self.funcmap.get(intent_code)
            if asyncio.iscoroutinefunction(func):
                content = await func(**kwargs)
            else:
                content = func(**kwargs)
        except Exception as e:
            logger.exception(f"call_function {intent_code} error: {e}")
            raise e
        return content

    def aigc_functions_judge_question(self, **kwargs):
        """
        判断输入的句子是否为疑问句

        - Args:
            prompt (str): 输入的句子

        - Returns:
            int: 是否为疑问句，1表示是疑问句，0表示非疑问句
        """
        options = ["0", "1"]
        prompt = kwargs.get("prompt")
        new_prompt = f"请你帮我判断用户输入的句子是否为疑问句：'{prompt}'，疑问句:'1',非疑问句:'0'"
        answer = self.aigc_functions_single_choice(new_prompt, options)
        return answer

    async def aigc_functions_customer_conversion(self, **kwargs) -> dict:
        """
        客户转化推荐分析

        需求文档:
            https://alidocs.dingtalk.com/i/nodes/amweZ92PV6BD4ZlzHEol2Pw0VxEKBD6p?utm_medium=wiki_feed_notification&utm_source=im_SYSTEM

        能力说明:
            根据客户的不同类型（体检（门诊）客户、已出组客户、未出组客户），动态生成已知信息和输出要求，
            调用大模型完成转化分析并将输出解析为结构化字段，返回转化评估情况、沟通话术及方式方法。

        参数:
            kwargs (dict): 包含以下键的参数字典：
                - intentCode (str): 意图编码，用于动态获取模板。
                - customer_type (str): 客户类型，支持 "体检（门诊）客户"、"已出组客户"、"未出组客户"。
                - user_profile (dict, optional): 用户画像，包含客户的基本信息（如年龄、性别、身高、体重等）。
                - report_summary (str, optional): 客户体检报告总结部分，仅体检（门诊）客户需要。
                - diagnosis (str, optional): 门诊病历内容，仅体检（门诊）客户需要。
                - group_report (dict, optional): 出组报告，仅已出组客户需要，包含数据充分性、血糖达标分析和波动情况。
                - interaction_data (dict, optional): 群聊互动数据，仅未出组客户需要，包含发言次数、图片发送次数等。

        返回:
            dict: 包含解析后模型输出的字段：
                - conversion_evaluation (str): 转化评估情况，包括短期和长期转化可能性。
                - communication_script (str): 沟通话术，用于与客户交流时的推荐话术。
                - methods (str): 方式方法，包含具体的转化方案建议。

        示例:
            请求参数:
            {
                "intentCode": "aigc_functions_customer_conversion",
                "customer_type": "体检（门诊）客户",
                "user_profile": {
                    "age": 45,
                    "gender": "男",
                    "height": "170cm",
                    "weight": "70kg",
                    "current_diseases": "高血糖",
                    "history": "无"
                },
                "report_summary": "客户空腹血糖偏高，推荐进一步检查糖耐量",
                "diagnosis": "糖尿病前期，需监测"
            }

            响应结果:
            {
                "conversion_evaluation": "短期转化可能性：中等\n长期转化可能性：高",
                "communication_script": "尊敬的客户，你好！根据你的体检报告和门诊记录，你的空腹血糖已经偏高，并且被诊断为糖尿病前期...",
                "methods": "1. 建议客户使用动态血糖仪进行实时监测。\n2. 提供五师团队服务，包含饮食、运动、心理支持等。\n3. 鼓励客户加入健康管理群组，与其他客户共享经验。"
            }
        """
        _event = "客户转化推荐分析"
        kwargs = deepcopy(kwargs)
        intent_code = kwargs.get("intentCode")

        customer_type = kwargs.get("customer_type")
        if not customer_type:
            raise ValueError("客户类型不能为空")

        # Step 1: 从 MySQL 动态获取模板
        prompt_template = self.gsr.get_event_item(intent_code).get("description")

        # Step 2: 动态生成两部分内容
        known_info = await self._generate_known_info(customer_type, kwargs)
        output_requirements = await self._generate_output_requirements(customer_type)

        # Step 3: 合并所有部分
        prompt = prompt_template.format(
            known_info=known_info,
            output_requirements=output_requirements
        )

        model_args = await self.__update_model_args__(
            kwargs, temperature=0.7, top_p=1, repetition_penalty=1.0
        )
        content: str = await self.aaigc_functions_general(
            _event=_event, prompt_template=prompt, model_args=model_args, **kwargs
        )

        result = await self._parse_model_output(content)
        return result

    async def _parse_model_output(self, content: str) -> dict:
        """
        解析大模型返回的内容，将结果分解为结构化字段：转化评估情况、沟通话术、方式方法。

        参数:
            content (str): 大模型返回的完整字符串。

        返回:
            dict: 包含解析后字段的字典。
                - conversion_evaluation: 转化评估情况
                - communication_script: 沟通话术
                - methods: 方式方法
        """
        # 初始化返回结构
        parsed_result = {
            "conversion_evaluation": None,
            "communication_script": None,
            "methods": None
        }

        # 根据分隔符解析内容
        sections = content.split("\n\n")

        # 遍历每一段内容
        for section in sections:
            if section.startswith("转化评估情况："):
                # 提取转化评估情况
                parsed_result["conversion_evaluation"] = section.replace("转化评估情况：", "").strip()
            elif section.startswith("沟通话术："):
                # 提取沟通话术
                parsed_result["communication_script"] = section.replace("沟通话术：", "").strip()
            elif section.startswith("方式方法："):
                # 提取方式方法
                parsed_result["methods"] = section.replace("方式方法：", "").strip()

        # 返回解析后的内容
        return parsed_result

    async def _generate_known_info(self, customer_type: Optional[str], params: dict) -> str:
        """
        动态生成 # 已知信息，根据客户类型选择性生成内容。
        """
        # 提取字段值
        user_profile = params.get("user_profile", {})
        report_summary = params.get("report_summary")
        diagnosis = params.get("diagnosis")
        group_report = params.get("group_report", {})
        interaction_data = params.get("interaction_data", {})

        # 使用通用组装方法生成用户画像内容
        user_profile_str = self.__compose_user_msg__("user_profile",
                                                           user_profile=user_profile) if user_profile else None

        # 处理出组报告内容
        group_report_str = self.__compose_user_msg__(
            mode="group_report",
            user_profile=group_report
        ) if group_report and customer_type == "已出组客户" else None

        # 处理群聊互动数据内容
        interaction_data_str =  self.__compose_user_msg__(
            mode="interaction_data",
            user_profile=interaction_data
        ) if interaction_data and customer_type == "未出组客户" else None

        # 根据客户类型动态构造已知信息段落
        known_info_parts = [
            f"# 已知信息\n## 客户来源类型：{customer_type}" if customer_type else None,
            f"## 用户画像\n{user_profile_str}".strip() if user_profile_str else None,
        ]

        # 针对体检（门诊）客户，拼接体检报告或门诊病历
        if customer_type == "体检（门诊）客户":
            if report_summary:
                known_info_parts.append(f"## 客户体检报告\n- {report_summary}")
            if diagnosis:
                known_info_parts.append(f"## 门诊病历\n- {diagnosis}")

        # 针对已出组客户，拼接出组报告
        if customer_type == "已出组客户":
            if group_report_str:
                known_info_parts.append(f"## 出组报告\n{group_report_str}")

        # 针对未出组客户，拼接群聊互动数据
        if customer_type == "未出组客户":
            if interaction_data_str:
                known_info_parts.append(f"## 群聊互动数据\n{interaction_data_str}")

        # 过滤掉 None 的部分并拼接
        return "\n\n".join(filter(None, known_info_parts))

    async def _generate_output_requirements(self, customer_type: str) -> str:
        """
        动态生成 # 输出要求
        """
        requirements = {
            "体检（门诊）客户": (
                "- 分析待转化客户是否已诊断糖尿病，血糖是否管理不佳（血糖是否异常，糖化血红蛋白是否异常）。\n"
                "- 期望结论：无糖尿病诊断&血糖异常用户，转化重心：暂无转化可能性，转化方向：来院就诊，明确糖尿病诊断；\n"
                "  有糖尿病诊断&血糖异常用户，转化重心：转化可能性中，转化方向：培养血糖管理意识。"
            ),
            "已出组客户": (
                "- 分析管理效果，包括依从性和管理效果（血糖波动、达标情况）。\n"
                "- 期望结论：根据依从性和管理效果判定短期和长期转化可能性。\n"
                "- 提供优化方案建议。"
            ),
            "未出组客户": (
                "- 分析当前管理数据，判断是否存在改进空间。\n"
                "- 提供当前阶段的转化可能性分析和改进方向。"
            ),
        }

        return f"# 输出要求\n{requirements.get(customer_type, '未知客户类型')}\n"


if __name__ == "__main__":
    gsr = InitAllResource()
    agents = Agents(gsr)
    param = testParam.param_dev_report_interpretation
    agents.call_function(**param)