import asyncio
from copy import deepcopy
import json
import random
import re
import sys
import time
from os.path import basename
from pathlib import Path

import json5
import openai
from fastapi.exceptions import ValidationException
from requests import Session

import datetime
from datetime import datetime, timedelta
from string import Template
from typing import AsyncGenerator, Dict, Generator, List, Literal, Optional, Union

from langchain.prompts.prompt import PromptTemplate
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

from chat.qwen_chat import Chat
from data.constrant import *
from data.constrant import DEFAULT_RESTAURANT_MESSAGE, HOSPITAL_MESSAGE
from data.jiahe_prompt import *
from data.jiahe_util import *
from data.test_param.test import testParam
from src.pkgs.models.utils import ParamTools
from src.prompt.model_init import ChatMessage, acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.module import (
    InitAllResource,
    accept_stream_response,
    clock,
    compute_blood_pressure_level,
    construct_naive_response_generator,
    download_from_oss,
    dumpJS,
    param_check,
    parse_examination_plan,
    calculate_bmr,
    parse_measurement,
    parse_historical_diets,
    async_clock,
    convert_meal_plan_to_text,
)

class JiaheExpertModel:
    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr = gsr
        self.client = openai.OpenAI()
h
    @staticmethod
    async def long_term_nutritional_management(userInfo={}, history=[]):
        """长期营养管理意图识别"""
        info, his_prompt = get_userInfo_history(userInfo, history[-1:])
        #
        messages = [
            {
                "role": "user",
                "content": jiahe_nutritious_manage_prompt.format(his_prompt),
            }
        ]
        logger.debug("长期营养管理识别模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("长期营养管理识别模型输出： " + json.dumps(messages, ensure_ascii=False))
        generate_text = generate_text[generate_text.find('Output')+6:].split('\n')[0].strip()
        if '否' in generate_text:
            yield {'underlying_intent': False, 'message': '', 'end': True}
        else:
            messages = [
                {
                    "role": "user",
                    "content": jiahe_nutritious_manage_prompt.format(userInfo, his_prompt)
                }
            ]
            logger.debug("长期营养管理话术模型输入： " + json.dumps(messages, ensure_ascii=False))
            generate_text = callLLM(
                history=messages,
                max_tokens=2048,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen1.5-32B-Chat",
            )
            logger.debug("长期营养管理话术模型输出： " + json.dumps(messages, ensure_ascii=False))
            yield {"message": generate_text, "underlying_intent": True, "end": True}