# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-25 01:05:35
@desc    :   simple chatbot
@Author  :   ticoAg
@Contact :   1627635056@qq.com
'''

import os
import time

import streamlit as st
from loguru import logger
from openai import OpenAI

client = OpenAI()

class Args:
    ...
args = Args()
def prepare_parameters():
    """Initialize the parameters for the llm"""
    global args
    args.max_tokens = st.sidebar.slider("Max tokens", min_value=1, max_value=32000, value=4096, step=1)
    args.temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    args.top_p = st.sidebar.slider("Top p", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    # args.top_k = st.sidebar.slider("Top k", min_value=-1, max_value=100, value=-1, step=1)
    args.n = st.sidebar.slider("N", min_value=1, max_value=50, value=1, step=1)
    args.stop = st.sidebar.text_input("Stop words(split with `,`)", value="")
    args.presence_penalty = st.sidebar.slider("Presence penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    args.frequency_penalty = st.sidebar.slider("Frequency penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

with st.sidebar:
    client.base_url = st.text_input("api base", key="openai_api_base", value=os.environ.get("OPENAI_API_BASE", ""))
    api_key = st.text_input("api key", key="openai_api_key", value=None)
    client.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")
    
    model_list = [i.id for i in client.models.list().data]
    args.model = st.selectbox("Choose your model", model_list,index=1)

    system_prompt = st.text_area(
        "system prompt", 
"""请你模拟医生问诊的过程，逐步分析用户的病情，给出初步诊断，请遵循一下流程:
1. 明确患者主诉信息
2. 逐步询问患者持续时间、发生时机、诱因或症状发生部位等信息，每次只问一个问题
3. 明确必要的患者信息后，推断用户可能患有的疾病，逐步询问患者疾病初诊、鉴别诊断、确诊需要的其他信息, 如家族史、既往史、检查结果等信息
4. 经过多轮问诊后给出初步诊断结果，列出可能性最高的几种疾病并解释原因""",
    height=200
    )
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

prepare_parameters()
st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenSource LLM")
if "messages" not in st.session_state:
    st.session_state['messages'] = []
    if system_prompt:
        st.session_state.messages.append({"role": "system", "content": system_prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("AI is thinking..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(**args.__dict__, 
                                                            messages=st.session_state.messages, 
                                                            stream=True):
                if not response.choices[0].delta.content:
                    continue
                full_response += response.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        logger.info(f"User: {prompt}\nAssistant: {full_response}\nHistory: {st.session_state.messages}")

# pip install openai --upgrade
# streamlit run Chatbot.py --server.port