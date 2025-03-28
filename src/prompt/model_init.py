# -*- encoding: utf-8 -*-
"""
@Time    :   2023-10-30 11:15:48
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
from ast import dump
import time
from typing import Dict, List, Literal, Optional, Tuple, Union

import openai
from pydantic import BaseModel, Field

from src.utils.Logger import logger
from src.utils.module import apply_chat_template, dumpJS

from src.utils.Logger import logger
from data.constrant import DEFAULT_MODEL
import asyncio
from datetime import datetime

default_model = "Qwen1.5-32B-Chat"

retry_times = 3


def pre_process_model_args(**kwargs) -> Dict:
    keywords = ["repetition_penalty", "do_sample", "top_k", "verbose"]
    for k in keywords:
        if k in kwargs:
            logger.warning(
                f"Keyword {k} is not supported in callLLM, please remove it."
            )
            del kwargs[k]
    return kwargs


def callLLM(
        query: str = "",
        history: List[Dict] = [],
        temperature=0.5,
        top_p=0.5,
        max_tokens=512,
        model: str = DEFAULT_MODEL,
        stop=[],
        stream=False,
        is_vl=False,
        extra_params: dict = {},
        **kwargs,
):
    """chat with qwen api which is serve at http://10.228.67.99:26921

    List options

    Args:
        query (string or null, Required):
            Can be None or emtpy string.
            If query, history will append {{"role":"user", "content": query}}
        history (`array[Dict]`, [], Required):
            A list of messages comprising the conversation so far.
        top_p (`number` or `null` Optional Defaults to 0.5):
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both.
        temperature (number or null Optional Defaults to 0.7):
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        stop (List[str], optional, defaults to []):
            A list of stop words to stop the chat.
        stream (bool, optional, defaults to False):
            Whether to stream the response or return the full response at once.
    """
    kwargs = pre_process_model_args(**kwargs)
    client = openai.OpenAI()
    # logger.info(f"base_url: {client.base_url}, api_key: {client.api_key}")
    # if model != default_model:
    #     logger.warning(
    #         f"There will change Model: {model} to {default_model}."
    #         + "Please manually check your code use config file to manage which model to use."
    #     )
    for msg in history:
        msg_copy = msg.copy()
        msg_copy.pop('intentCode', None)

    if stream and stop:
        logger.warning(
            "Stop is not supported in stream mode, please remove stop parameter or set stream to False. Otherwise, stop won't be work fine."
        )
    # model = default_model
    t_st = time.time()
    kwds = {
        "model": model,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
        "timeout": 120,
        **kwargs,
    }
    logger.trace(f"callLLM with {dumpJS(kwds)}")

    if not history:
        if "qwen1.5" in model.lower():
            query = apply_chat_template(query)
        kwds["prompt"] = query
        retry = 0
        while retry <= retry_times:
            try:
                completion = client.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"request llm model error, retry to request")
                continue

        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = client.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"request llm model error, retry to request")
                continue

        ret = completion.choices[0].text
    else:
        if is_vl:
            kwds["messages"] = history
        else:
            if query and not isinstance(query, object):
                history += [{"role": "user", "content": query}]
            # 定义允许的字段集合
            allowed_keys = {'role', 'content'}

            # 过滤消息中的非允许字段
            clean_history = []
            for msg in history:
                msg_copy = {k: v for k, v in msg.items() if k in allowed_keys}
                clean_history.append(msg_copy)

            msg = ""
            for i, n in enumerate(list(reversed(clean_history))):
                msg += n["content"]
                if len(msg) > 1800:
                    h = clean_history[-i:]
                    break
                else:
                    h = clean_history
            kwds["messages"] = h
        # logger.debug("LLM输入：" + json.dumps(kwds, ensure_ascii=False))
        retry = 0
        while retry <= retry_times:
            try:
                completion = client.chat.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = client.chat.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")
        ret = completion.choices[0].message.content.strip()
    time_cost = round(time.time() - t_st, 1)
    logger.info(
        f"Model {model} generate costs summary: "
        + f"prompt_tokens:{completion.usage.prompt_tokens}, "
        + f"completion_tokens:{completion.usage.completion_tokens}, "
        + f"total_tokens:{completion.usage.total_tokens}, "
          f"cost: {time_cost}s"
    )
    return ret


async def acallLLM(
        query: str = "",
        history: List[Dict] = [],
        temperature=0.5,
        top_p=0.5,
        max_tokens=1024,
        model: str = DEFAULT_MODEL,
        stop=[],
        stream=False,
        is_vl=False,
        **kwargs,
):
    """chat with qwen api which is serve at http://10.228.67.99:26921

    List options

    Args:
        query (string or null, Required):
            Can be None or emtpy string.
            If query, history will append {{"role":"user", "content": query}}
        history (`array[Dict]`, [], Required):
            A list of messages comprising the conversation so far.
        top_p (`number` or `null` Optional Defaults to 0.5):
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both.
        temperature (number or null Optional Defaults to 0.7):
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        stop (List[str], optional, defaults to []):
            A list of stop words to stop the chat.
        stream (bool, optional, defaults to False):
            Whether to stream the response or return the full response at once.
    """
    kwargs = pre_process_model_args(**kwargs)
    aclient = openai.AsyncOpenAI()
    # logger.info(f"base_url: {client.base_url}, api_key: {client.api_key}")
    # if model != default_model:
    #     logger.warning(
    #         f"There will change Model: {model} to {default_model}."
    #         + "Please manually check your code use config file to manage which model to use."
    #     )
    if stream and stop:
        logger.warning(
            "Stop is not supported in stream mode, please remove stop parameter or set stream to False. Otherwise, stop won't be work fine."
        )
    # model = default_model
    t_st = time.time()
    kwds = {
        "model": model,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
        "timeout": 240,
        **kwargs,
    }
    logger.trace(f"callLLM with {dumpJS(kwds)}")

    if not history:
        if "qwen" in model.lower():
            query = apply_chat_template(query)
        kwds["prompt"] = query
        retry = 0
        while retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue

        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].text
    else:
        if is_vl:
            kwds["messages"] = history
        else:
            if query and not isinstance(query, object):
                history += [{"role": "user", "content": query}]
            msg = ""
            for i, n in enumerate(list(reversed(history))):
                msg += n["content"]
                if len(msg) > 1800:
                    h = history[-i:]
                    break
                else:
                    h = history
            kwds["messages"] = h
        retry = 0 if not is_vl else 2 # vl相对较慢，不太好直接修改全局变量retry_times，所以增加起始计数，相当于减少vl重试次数
        while retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
        logger.info(f"Model generate completion:{repr(completion)}")
        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].message.content.strip()
    time_cost = round(time.time() - t_st, 1)
    logger.info(
        f"Model {model} generate costs summary: "
        + f"prompt_tokens:{completion.usage.prompt_tokens}, "
        + f"completion_tokens:{completion.usage.completion_tokens}, "
        + f"total_tokens:{completion.usage.total_tokens}, "
          f"cost: {time_cost}s"
    )
    return ret


async def acallLLtrace(
        query: str = "",
        history: List[Dict] = [],
        temperature=0.5,
        top_p=0.5,
        max_tokens=1024,
        model: str = DEFAULT_MODEL,
        stop=[],
        stream=False,
        is_vl=False,
        extra_params = {},
        **kwargs,
):
    """chat with qwen api which is serve at http://10.228.67.99:26921

    List options

    Args:
        query (string or null, Required):
            Can be None or emtpy string.
            If query, history will append {{"role":"user", "content": query}}
        history (`array[Dict]`, [], Required):
            A list of messages comprising the conversation so far.
        top_p (`number` or `null` Optional Defaults to 0.5):
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both.
        temperature (number or null Optional Defaults to 0.7):
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        stop (List[str], optional, defaults to []):
            A list of stop words to stop the chat.
        stream (bool, optional, defaults to False):
            Whether to stream the response or return the full response at once.
    """
    kwargs = pre_process_model_args(**kwargs)
    aclient = openai.AsyncOpenAI()
    # 从 extra_params 提取追踪相关信息
    langfuse = extra_params.get("langfuse")
    user_id = extra_params.get("user_id")
    session_id = extra_params.get("session_id")
    # 如果存在 user_id 和 session_id，创建 trace
    trace = None
    generation = None
    if user_id and session_id:
        trace = langfuse.trace(
            name=extra_params.get("name"),
            user_id=user_id,
            session_id=session_id,
            release=extra_params.get("release"),
            tags=extra_params.get("tags"),
            metadata=extra_params.get("metadata")
        )

        # 创建 Generation
        generation = trace.generation(
            name="llm-generation",
            model=model,
            model_parameters={
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            metadata={"streaming": stream},
        )

    # 确保 trace 不为 None
    if trace is not None:
        trace.update(input=query or history)
    else:
        logger.error("Trace object is None, skipping trace update.")
    if generation is not None:
        generation.update(
            input=query or history
        )
    # logger.info(f"Starting model invocation with query: {query}")

    if stream and stop:
        logger.warning(
            "Stop is not supported in stream mode, please remove stop parameter or set stream to False. Otherwise, stop won't be work fine."
        )
    # model = default_model
    t_st = time.time()
    kwds = {
        "model": model,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
        "timeout": 240,
        **kwargs,
    }
    logger.trace(f"callLLM with {dumpJS(kwds)}")

    if not history:
        if "qwen" in model.lower():
            query = apply_chat_template(query)
        kwds["prompt"] = query
        retry = 0
        while retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                if completion.choices:
                    generation.update(
                        completion_start_time=datetime.now()
                    )
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue

        if stream:
            if completion:
                # 记录流的开始时间
                generation.update(completion_start_time=datetime.now())
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].text
    else:
        if is_vl:
            kwds["messages"] = history
        else:
            if query and not isinstance(query, object):
                history += [{"role": "user", "content": query}]
            msg = ""
            for i, n in enumerate(list(reversed(history))):
                msg += n["content"]
                if len(msg) > 1800:
                    h = history[-i:]
                    break
                else:
                    h = history
            kwds["messages"] = h
        retry = 0 if not is_vl else 2 # vl相对较慢，不太好直接修改全局变量retry_times，所以增加起始计数，相当于减少vl重试次数
        while retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                if completion.choices:
                    generation.update(
                        completion_start_time=datetime.now()
                    )
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
        logger.info(f"Model generate completion:{repr(completion)}")
        if stream:
            if completion:
                # 记录流的开始时间
                generation.update(completion_start_time=datetime.now())
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].message.content.strip()
    time_cost = round(time.time() - t_st, 1)

    # 如果存在 trace 和 generation，则记录追踪信息
    if trace and generation:
        # 提取模型使用情况
        usage_details = {
            "input": completion.usage.prompt_tokens,
            "output": completion.usage.completion_tokens,
            "total": completion.usage.total_tokens
        }
        cost_details = {
            "input": usage_details["input"] * 0.00001,
            "output": usage_details["output"] * 0.00002,
        }

        # logger.info(f"Output value being sent to Langfuse: {ret}")
        generation.end(
            usage=usage_details,
            total_cost=cost_details,
        )

        # 确保 Flush 成功
        try:
            langfuse.flush()
            logger.info("Langfuse flush executed successfully.")
        except Exception as e:
            logger.error(f"Langfuse flush failed: {e}")

    if trace is not None:
        trace.update(output=ret)

    logger.info(
        f"Model {model} generate costs summary: "
        + f"prompt_tokens:{completion.usage.prompt_tokens}, "
        + f"completion_tokens:{completion.usage.completion_tokens}, "
        + f"total_tokens:{completion.usage.total_tokens}, "
          f"cost: {time_cost}s"
    )

    return ret


async def callLikangLLM(
        query: str = "",
        history: List[Dict] = [],
        temperature=0.5,
        top_p=0.5,
        max_tokens=1024,
        model: str = DEFAULT_MODEL,
        stop=[],
        stream=False,
        **kwargs,
):
    """chat with qwen api which is serve at http://10.228.67.99:26921

    List options

    Args:
        query (string or null, Required):
            Can be None or emtpy string.
            If query, history will append {{"role":"user", "content": query}}
        history (`array[Dict]`, [], Required):
            A list of messages comprising the conversation so far.
        top_p (`number` or `null` Optional Defaults to 0.5):
            An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are considered.
            We generally recommend altering this or temperature but not both.
        temperature (number or null Optional Defaults to 0.7):
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        stop (List[str], optional, defaults to []):
            A list of stop words to stop the chat.
        stream (bool, optional, defaults to False):
            Whether to stream the response or return the full response at once.
    """
    kwargs = pre_process_model_args(**kwargs)
    aclient = openai.AsyncOpenAI()
    # logger.info(f"base_url: {client.base_url}, api_key: {client.api_key}")
    # if model != default_model:
    #     logger.warning(
    #         f"There will change Model: {model} to {default_model}."
    #         + "Please manually check your code use config file to manage which model to use."
    #     )
    if stream and stop:
        logger.warning(
            "Stop is not supported in stream mode, please remove stop parameter or set stream to False. Otherwise, stop won't be work fine."
        )
    # model = default_model
    t_st = time.time()
    kwds = {
        "model": model,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "stream": stream,
        "timeout": 240,
        **kwargs,
    }
    logger.trace(f"callLLM with {dumpJS(kwds)}")

    if not history:
        if "qwen" in model.lower():
            query = apply_chat_template(query)
        kwds["prompt"] = query
        retry = 0
        while retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue

        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].text
    else:
        if query and not isinstance(query, object):
            history += [{"role": "user", "content": query}]
        msg = ""
        for i, n in enumerate(list(reversed(history))):
            msg += n["content"]
            if len(msg) > 25000:
                h = history[-i:]
                break
            else:
                h = history
        kwds["messages"] = h
        retry = 0
        while retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                break
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
        logger.info(f"Model generate completion:{repr(completion)}")
        if stream:
            return completion
        retry = 0
        while not completion.choices and retry <= retry_times:
            try:
                completion = await aclient.chat.completions.create(**kwds)
                retry += 1
            except Exception as e:
                retry += 1
                logger.info(f"call llm error:{repr(e)}")
                logger.info(f"request llm model error, retry to request")
                continue
            logger.info(f"Model generate completion:{repr(completion)}")

        ret = completion.choices[0].message.content.strip()
    time_cost = round(time.time() - t_st, 1)
    logger.info(
        f"Model {model} generate costs summary: "
        + f"prompt_tokens:{completion.usage.prompt_tokens}, "
        + f"completion_tokens:{completion.usage.completion_tokens}, "
        + f"total_tokens:{completion.usage.total_tokens}, "
          f"cost: {time_cost}s"
    )
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
