# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-25 01:05:35
@desc    :   simple chatbot
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import copy
import json
import os
import sys
from pathlib import Path

sys.path.append(Path("./").parent.as_posix())

import streamlit as st
from loguru import logger
from openai import OpenAI

from src.test.exp.data.prompts import Sysprompt

logger.remove()
logger.add(sink=sys.stderr, level="TRACE", backtrace=True, diagnose=True)
logger.remove()
logger.add(sink=sys.stderr, level="TRACE", backtrace=True, diagnose=True)
logger.add(
    Path("logs", "chatbot.log"),
    level="TRACE",
    encoding="utf-8",
    rotation="100 MB",
    retention="5 days",
    compression="gz",
    backtrace=True,
    diagnose=True,
)

client = OpenAI()
default_system_prompt = Sysprompt.system_prompt


class Args: ...


args = Args()


def dumpJS(obj):
    return json.dumps(obj, ensure_ascii=False)


def place_sidebar():
    with st.sidebar:
        client.base_url = st.text_input(
            "api base",
            key="openai_api_base",
            value=os.environ.get("OPENAI_API_BASE", ""),
        )
        api_key = st.text_input("api key", key="openai_api_key", value=None)
        client.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")

        model_list = [i.id for i in client.models.list().data]
        args.model = st.selectbox("Choose your model", model_list, index=2)
        st.text_area(
            "system prompt",
            default_system_prompt,
            height=200,
            key="system_prompt",
            on_change=initlize_system_prompt,
        )
        prepare_parameters()
        # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


def prepare_parameters():
    """Initialize the parameters for the llm"""
    global args
    args.max_tokens = st.sidebar.slider("Max tokens", min_value=1, max_value=32000, value=4096, step=1)
    args.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="模型在生成文本时的随机性。这个参数的值在0到2之间，越高表示模型越倾向于选择不太可能的单词，越低表示模型越倾向于选择最可能的单词。一般来说，这个值越高，模型生成的文本越有创意，但也可能会出现更多的语法错误或不合理的内容。",
    )
    args.top_p = st.sidebar.slider(
        "Top p",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="模型在生成文本时的概率阈值。这个参数的值在0到1之间，表示模型只会从概率之和大于等于这个值的单词中选择一个。这个参数可以帮助模型过滤掉一些极端的或不相关的单词，提高文本的质量和一致性。",
    )
    # args.top_k = st.sidebar.slider(
    #     "Top k",
    #     min_value=-1,
    #     max_value=100,
    #     value=-1,
    #     step=1,
    #     help="模型在生成文本时的候选单词的数量。这个参数的值是一个整数，表示模型只会从概率最高的这么多个单词中选择一个。这个参数和top_p类似，也可以帮助模型过滤掉一些不合适的单词，但是它不考虑单词的概率，只考虑排名。这个参数在您的代码中被注释掉了，表示不使用它。",
    # )
    args.n = st.sidebar.slider(
        "N",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        help="模型生成的文本的数量。这个参数的值是一个整数，表示模型会根据同一个提示生成多少个不同的文本。这个参数可以帮助您比较模型的多样性和稳定性，或者从多个选项中选择最合适的一个。",
    )
    args.presence_penalty = st.sidebar.slider(
        "Presence penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="此参数用于阻止模型在生成的文本中过于频繁地重复相同的单词或短语。它是每次在生成的文本中出现时都会添加到标记的对数概率中的值。较高的Frequency_penalty值将导致模型在使用重复标记时更加保守。",
    )
    args.frequency_penalty = st.sidebar.slider(
        "Frequency penalty",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="用来控制模型生成文本时避免重复相同的单词或短语。它是一个值，每次生成的单词出现在文本中时，就会加到该单词的对数概率上。这个值越高（接近1），模型就越不倾向于重复单词或短语；这个值越低（接近0），模型就越允许重复。您可以根据您的需求和期望的输出来调整这个参数的值",
    )
    args.repetition_penalty = st.sidebar.slider(
        "Repetition penalty",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="用来控制模型生成文本时避免重复相同的单词或短语。它是一个值，每次生成的单词出现在文本中时，就会乘以该单词的对数概率上。这个值越高（接近1），模型就越不倾向于重复单词或短语；这个值越低（接近0），模型就越允许重复。您可以根据您的需求和期望的输出来调整这个参数的值",
    )
    args.stop = ['\nObservation']


def initlize_system_prompt():
    """Initialize the system prompt"""
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": st.session_state.system_prompt})

    # logger.debug(f"Update system_prompt:\n{st.session_state.system_prompt}")


def parse_response(text):
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


place_sidebar()


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenSource LLM")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

    if st.session_state.system_prompt:
        st.session_state.messages.append({"role": "system", "content": st.session_state.system_prompt})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        logger.debug(f"{dumpJS(args.__dict__)}")
        logger.debug(f"Input: \n{prompt}")
        for response in client.chat.completions.create(
            **args.__dict__, messages=st.session_state.messages, stream=True
        ):
            if not response.choices[0].delta.content:
                continue
            full_response += response.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        logger.debug(f"Full response:\n{full_response}")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    messages = copy.deepcopy(st.session_state.messages)
    logger.debug(f"Messages: {dumpJS(messages)}")
    logger.info("".center(80, "="))


# pip install openai --upgrade
# streamlit run Chatbot.py --server.port
