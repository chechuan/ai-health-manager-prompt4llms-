# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-30 11:15:48
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import openai
import yaml
from pydantic import BaseModel, Field

from src.prompt.qwen_generation_utils import make_context

api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
openai.api_base = api_config['llm'] + "/v1"
openai.api_key = "EMPTY"
model="Qwen-14B-Chat"

def chat_qwen(query: str = "", history: List[Dict] = []):
    if isinstance(query, str) and query:
        history += [{"role": "user", "content": query}]
    completion = openai.ChatCompletion.create(
        model=model,
        messages=history,
        top_k=0, 
        top_p=0, 
        repetition_penalty=1.1,
        temperature=0.6
    )
    ret = completion['choices'][0]['message']['content'].strip()
    return ret

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None
    schedule: Optional[List] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
