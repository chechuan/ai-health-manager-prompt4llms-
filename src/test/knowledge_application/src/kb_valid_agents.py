# -*- encoding: utf-8 -*-
"""
@Time    :   2024-07-18 17:36:29
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import json5
import jsonlines
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.log import setup_logger
from qwen_agent.tools.base import BaseTool, register_tool
from requests import Session

sys.path.append(Path(__file__).parents[1].as_posix())

from src.protocal import SearchDocsRequest

load_dotenv(dotenv_path="src/test/knowledge_application/.env")
setup_logger("INFO")


@register_tool("query_judge")
class queryJudge(BaseTool):
    description = "query_judge可以帮助判断给出的问题是否需要查询知识库的知识"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "要判断的问题",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)["query"]
        system = (
            "请你帮我判断给出的问题是否需要查询知识库的知识,"
            "如果需要严谨的知识支撑才能回答,或者你本身不懂/不确定相关知识,"
            "则输出`Yes`,要求后续步骤查询相关知识,不需要相关知识就可以回答则输出`No`"
        )
        msg = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result: ChatCompletion = asyncio.run(
            aclient.chat.completions.create(messages=msg, model="Qwen2-7B-Instruct")
        )
        return result.choices[0].message.content


@register_tool("search_knowledge")
class searchKnowledgeTool(BaseTool):
    description = (
        "我们提供了一个丰富的健康领域知识库，"
        "知识来源于各种健康书籍，文献等，但是比较分散且存在内容的重复，"
        "可以通过语义匹配的方式查询知识库"
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "要查询的问题",
            "required": True,
        }
    ]

    def get_session(
        self,
    ):
        return Session()

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)["query"]
        # prompt = urllib.parse.quote(prompt)

        # TODO 增加Query改写环节 并发查询

        data = SearchDocsRequest(
            knowledge_base_name="valid",
            query=prompt,
            top_k=3,
            score_threshold=0.7,
            use_reranker=False,
            # rerank_top_k=3,
            # rerank_threshold=0.35,
        ).model_dump()

        with self.get_session() as session:
            response = session.post(
                os.getenv("BASE_KB_URL") + "/knowledge_base/search_docs",
                json=data,
                headers={"Content-Type": "application/json"},
            ).json()
        if not response:
            return "未查询到相关知识"
        else:
            return "\n\n".join(
                [re.sub(r"\n+", "\n", i["page_content"]) for i in response]
            )


def init_agent_service():
    llm_cfg = {
        "model": "Qwen2-7B-Instruct",
        "model_server": "http://10.39.91.251:40024/v1",
        "api_key": "sk-UuG2Rssx35xS7RHdF5E1E4Ad3e04435f844c7a4e5a4f6bF4",
    }
    system = (
        "请遵循以下流程回答问题\n"
        "1. 先使用query_judge工具判断是否需要查询知识库\n"
        "2. 不需要查询知识库的问题直接回答\n"
        "3. 需要查询知识库的在获取到知识后结合知识回答问题"
    )

    tools = [
        "query_judge",
        "search_knowledge",
    ]
    bot = Assistant(
        llm=llm_cfg,
        name="查克拉",
        description="查不查知识? 这是个问题",
        system_message=system,
        function_list=tools,
    )
    return bot


async def send_a_req(id, query: str) -> List[Dict]:
    print()
    bot = init_agent_service()
    messages = [{"role": "user", "content": query}]

    rsps = bot.run_nonstream(messages=messages, stream=False)

    processed_query.append(query)
    with jsonlines.open(rsps_cache_path, mode="a") as f:
        f.write(messages + rsps)

async def control_concurrency(task, semaphore):
    async with semaphore:
        await task
async def main(query: str = "成人糖尿病夏天吃什么比较合适?"):
    # Define the agent
    global rsps_cache_path
    global processed_query

    samaphore = asyncio.Semaphore(10)

    root_path = Path(__file__).parents[1]
    query_file_path = root_path.joinpath(".cache/kb_valid_query.json")
    rsps_cache_path = root_path.joinpath(".cache", "valid_ver1.1.jsonl")
    if not rsps_cache_path.exists():
        rsps_cache_path.touch()

    all_query = json5.load(open(query_file_path, "r"))
    processed_query = [i[0]["content"] for i in jsonlines.open(rsps_cache_path)]
    tasks = []
    for id, query in enumerate(all_query):
        if query in processed_query:
            continue
        tasks.append(control_concurrency(send_a_req(id, query), samaphore))

    await asyncio.gather(*tasks)

def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        query = input("user question: ")
        messages.append({"role": "user", "content": query})
        response = []
        for response in bot.run(messages=messages):
            print("bot response:", response)
        messages.extend(response)


def app_gui():
    # Define the agent
    bot = init_agent_service()
    chatbot_config = {
        "prompt.suggestions": [
            "画一只猫的图片",
            "画一只可爱的小腊肠狗",
            "画一幅风景画，有湖有山有树",
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == "__main__":
    aclient = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL"),
    )
    asyncio.run(main())
    # app_tui()
    # app_gui()
