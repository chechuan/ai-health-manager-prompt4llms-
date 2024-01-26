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

sys.path.append(Path(__file__).parent.as_posix())

import streamlit as st
from loguru import logger
from openai import OpenAI

from data.prompts import AuxiliaryDiagnosisPrompt

logger.add(
    Path("logs", "chatbot.log"),
    encoding="utf-8",
    rotation="00:00",
    retention="10 days",
    compression="gz",
    backtrace=True,
    diagnose=True,
)

client = OpenAI()
default_system_prompt = AuxiliaryDiagnosisPrompt.system_prompt_v2


class Args:
    ...


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
        args.model = st.selectbox("Choose your model", model_list, index=1)

        st.text_area(
            "system prompt",
            default_system_prompt,
            height=400,
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
    args.max_tokens = st.sidebar.slider(
        "Max tokens", min_value=1, max_value=32000, value=4096, step=1
    )
    args.temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1
    )
    args.top_p = st.sidebar.slider(
        "Top p", min_value=0.0, max_value=1.0, value=0.8, step=0.1
    )
    # args.top_k = st.sidebar.slider("Top k", min_value=-1, max_value=100, value=-1, step=1)
    args.n = st.sidebar.slider("N", min_value=1, max_value=50, value=1, step=1)
    args.presence_penalty = st.sidebar.slider(
        "Presence penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1
    )
    args.frequency_penalty = st.sidebar.slider(
        "Frequency penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1
    )
    args.stop = ["\nObservation", "\nFinally"]


def initlize_system_prompt():
    """Initialize the system prompt"""
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "system", "content": st.session_state.system_prompt}
    )
    logger.debug(f"Update system_prompt:\n{st.session_state.system_prompt}")


def parse_response(text):
    # text = """Thought: 我对问题的回复\nDoctor: 这里是医生的问题或者给出最终的结论"""
    thought_index = text.find("Thought:")
    doctor_index = text.find("\nDoctor:")
    if thought_index == -1 or doctor_index == -1:
        return None, None
    thought = text[thought_index + 8 : doctor_index].strip()
    doctor = text[doctor_index + 8 :].strip()
    return thought, doctor


place_sidebar()


st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenSource LLM")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

    if st.session_state.system_prompt:
        st.session_state.messages.append(
            {"role": "system", "content": st.session_state.system_prompt}
        )
        logger.debug("update system_prompt to messages")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    prompt = f"Observation: {prompt}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"{prompt}")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        logger.debug(f"{dumpJS(args.__dict__)}")
        for response in client.chat.completions.create(
            **args.__dict__, messages=st.session_state.messages, stream=True
        ):
            if not response.choices[0].delta.content:
                continue
            full_response += response.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        logger.debug(f"full_response:\n{full_response}")
        thought, doctor_output = parse_response(full_response)
        message_placeholder.markdown(
            f"~~Thought: {thought}~~ \nDoctor: {doctor_output}"
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": f"Doctor: {doctor_output}"}
    )
    messages = copy.deepcopy(st.session_state.messages)
    logger.debug(f"Messages:\n{[dumpJS(i) for i in messages]}")


# pip install openai --upgrade
# streamlit run Chatbot.py --server.port
