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

from src.utils.Logger import logger

api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
openai.api_base = api_config['llm'] + "/v1"
openai.api_key = "EMPTY"

def chat_qwen(query: str = "", 
              history: List[Dict] = [], 
              temperature=0.5,
              top_k=-1,
              top_p=0.5,
              repetition_penalty=1.1,
              max_tokens=512,
              model="Qwen-14B-Chat",
              do_sample=True,
              **kwargs):
    """chat with qwen api which is serve at http://10.228.67.99:26921
    
    List options

    Args:
        query (string or null, Required):
            Can be None or emtpy string.
            If query, history will append {{"role":"user", "content": query}}
        history (`array[Dict]`, [], Required):
            A list of messages comprising the conversation so far.
        top_k (`float`):
            Adding some randomness helps make output text more natural. 
            In top-k decoding, we first shortlist three tokens then sample one of them considering their likelihood scores.
        top_p (`number` or `null` Optional Defaults to 0.5):
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both.
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty.
        temperature (number or null Optional Defaults to 0.7): 
            What sampling temperature to use, between 0 and 2. 
            Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        do_sample (bool, optional, defaults to True)
            Whether or not to use sampling ; use greedy decoding otherwise.
    """
    t_st = time.time()
    if not history:
        completion = openai.Completion.create(
            model=model,
            prompt=query,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            do_sample=do_sample
        )
        ret = completion['choices'][0]['text']
    else:
        if query and not isinstance(query, object):
            history += [{"role": "user", "content": query}]
        msg = ''
        for i, n in enumerate(list(reversed(history))):
            msg += n['content']
            if len(msg) > 1200:
                h = history[-i:]
                break
            else:
                h = history
        completion = openai.ChatCompletion.create(
            model=model,
            messages=h,
            top_k=top_k, 
            top_p=top_p, 
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            do_sample=do_sample
        )
        ret = completion['choices'][0]['message']['content'].strip()
    time_cost = round(time.time() - t_st, 1)
    logger.debug(f"Gen counts," + 
                f"prompt_tokens:{completion['usage']['prompt_tokens']}," + 
                f"completion_tokens:{completion['usage']['completion_tokens']}," + 
                f"total_tokens:{completion['usage']['total_tokens']},"
                f"cost: {time_cost}s")
    return ret

#def truncat_history(history):

    #for cnt in range(len(history)-1, -1, -1):
        

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
