# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-25 01:05:35
@desc    :   simple chatbot
@Author  :   ticoAg
@Contact :   1627635056@qq.com
'''

import os
from platform import system

import streamlit as st
from loguru import logger
from openai import OpenAI

client = OpenAI()

default_system_prompt = """你是一个经验丰富的医生，同时又是一个营养运动学专家，请你协助我进行疾病的诊断，下面是对诊断流程的描述
1. 在多轮的对话中我会提供我的个人信息和感受，请你根据自身经验分析，针对我的个人情况提出相应的问题，但是每次只能问一个问题
2. 问题关键点可以包括：持续时间、发生时机、诱因或症状发生部位等, 注意同类问题可以总结在一起问
3. 最后请你结合获取到的信息给出我的诊断结果，可以是某种疾病，或者符合描述的中医症状，并解释给出这个诊断结果的原因，以及对应的处理方案
"""
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
    args.presence_penalty = st.sidebar.slider("Presence penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    args.frequency_penalty = st.sidebar.slider("Frequency penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    args.stop = st.sidebar.text_input("Stop words(split with `,`)", value="")

def initlize_system_prompt():
    """Initialize the system prompt"""
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": st.session_state.system_prompt})
    logger.debug("update system_prompt to messages")

with st.sidebar:
    client.base_url = st.text_input("api base", key="openai_api_base", value=os.environ.get("OPENAI_API_BASE", ""))
    api_key = st.text_input("api key", key="openai_api_key", value=None)
    client.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", "")
    
    model_list = [i.id for i in client.models.list().data]
    args.model = st.selectbox("Choose your model", model_list,index=1)

    st.text_area(
        "system prompt", 
        default_system_prompt,
        height=400,
        key="system_prompt",
        on_change=initlize_system_prompt
    )
    prepare_parameters()
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"





st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenSource LLM")

if "messages" not in st.session_state:
    st.session_state['messages'] = []

    if st.session_state.system_prompt:
        st.session_state.messages.append({"role": "system", "content": st.session_state.system_prompt})
        logger.debug("update system_prompt to messages")

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
        for response in client.chat.completions.create(**args.__dict__, messages=st.session_state.messages, stream=True):
            if not response.choices[0].delta.content:
                continue
            full_response += response.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# pip install openai --upgrade
# streamlit run Chatbot.py --server.port