# -*- encoding: utf-8 -*-
"""
@Time    :   2024-07-18 17:36:29
@desc    :   
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import argparse
import asyncio
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest import result

from tqdm import tqdm

sys.path.append(Path(__file__).parents[1].as_posix())

from typing import Dict, List, Union

import json5
import jsonlines
import openai
import pandas as pd
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from qwen_agent.agents import Assistant
from qwen_agent.log import setup_logger
from qwen_agent.tools.base import BaseTool, register_tool
from requests import Session

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
        system = """请你帮我判断给出的问题是否需要查询知识库的知识
        如果你本身无法解读,需要严谨的知识支撑才能回答,或者你本身不懂/不确定相关知识,则输出`Yes`,要求后续步骤查询相关知识,不需要相关知识就可以回答则输出`No`"""
        msg = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
        result: ChatCompletion = client.chat.completions.create(
            messages=msg, model="Qwen2-7B-Instruct"
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
        "model": args.model,
        "model_server": args.base_url,
        "api_key": args.api_key,
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


def send_a_req(id, query: str) -> List[Dict]:
    print(f"start to process task: {id}")
    # 异步调用init_agent_service
    bot = init_agent_service()
    messages = [{"role": "user", "content": query}]

    # 异步调用bot.run_nonstream
    rsps = bot.run_nonstream(messages=messages, stream=False)

    processed_query.append(query)
    with jsonlines.open(rsps_cache_path, mode="a") as f:
        f.write(messages + rsps)
    print(f"finish processing task: {id}")
    return rsps


class ResultItem:
    init_query: str = ""
    is_query_knowledge: str = "No"
    knowledge_query: str = ""
    knowledge_result: str = ""
    answer: str = ""

    def repr(self):
        return [
            self.init_query,
            self.is_query_knowledge,
            self.knowledge_query,
            self.knowledge_result,
            self.answer,
        ]


def parse_result(rsps_cache_path: Union[Path, str]):
    """解析结果记录"""
    result_save_path = rsps_cache_path.parent.joinpath(
        f"{rsps_cache_path.stem}_result_{args.model}.xlsx"
    )
    rsps_result = [i for i in jsonlines.open(rsps_cache_path)]
    df = pd.DataFrame(
        columns=["原始问题", "是否查知识", "知识查询query", "引用知识", "最终回复"],
        index=range(len(rsps_result)),
    )

    for idx, rsps in enumerate(rsps_result):
        result_item = ResultItem()
        result_item.init_query = rsps[0]["content"]
        result_item.answer = rsps[-1]["content"]
        for item in rsps:
            if item["role"] == "function" and item["name"] == "query_judge":
                assert_result = item["content"].split("\n")[0]
                if not assert_result.lower() == "yes":
                    result_item.is_query_knowledge = "No"
                else:
                    result_item.is_query_knowledge = "Yes"
            if (
                item["role"] == "assistant"
                and item.get("function_call")
                and item["function_call"]["name"] == "search_knowledge"
            ):
                try:
                    result_item.knowledge_query = json5.loads(
                        item["function_call"]["arguments"]
                    )["query"]
                except Exception as err:
                    print(repr(err))
                    result_item.knowledge_query = item["function_call"]["arguments"]
            if item["role"] == "function" and item["name"] == "search_knowledge":
                result_item.knowledge_result = item["content"]
        df.loc[idx] = result_item.repr()
    df.to_excel(result_save_path)
    return result_save_path


def extract(input_text):
    """使用extract抽取模获得抽取的结果"""
    ans_pattern = re.compile(r"评价结果: (.)", re.S)

    problems = ans_pattern.findall(input_text)
    if not problems:
        print(input_text)
        return "0"
    return problems[0]


def run_validate_releated_binary(
    query: str = "原始问题", knoeledge: str = "检索出的知识"
):
    """验证知识和问题是否相关"""

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
# 角色定义
你是一个经验丰富的医生,你可以深刻得理解我给出的信息和知识。

# 任务描述
1. 请你根据对问题与知识的理解,及丰富的诊疗经验,评估出已知信息能否可以回答问题,且知识的正确性
2. 需要注意只参考已知信息中明确提及的信息，不能参考超出该文本的信息判断
3. 请逐步分析问题需要什么知识,判断提供的知识是否满足要求,在最后一行输出最终的结果,最后一行的格式为: `\n评价结果: 0`

# 最终输出结果的判定标准
1. 知识和问题完全无关或知识错误，输出0
2. 知识和问题有关，对回答问题有帮助且没有事实错误, 输出1
#问题
{query}
#知识
{knoeledge}<|im_end|>
<|im_start|>assistant
"""
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.completions.create(
        prompt=prompt, model=args.model, temperature=0, top_p=0.01, max_tokens=1024
    )
    content = response.choices[0].text
    return extract(content), content


def run_validate_integrity_binary(
    query: str = "原始问题", knoeledge: str = "检索出的知识"
):
    """验证相关程度"""

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
# 角色定义
你是一个能力强大的信息鉴别助手,你可以深刻得理解我给出的信息和知识。

# 任务描述
1. 请你根据对问题与知识的理解,评估出只根据已知信息能否完整回答问题
2. 需要注意只参考已知信息中明确提及的信息，不能参考超出该文本的信息判断
3. 请逐步分析问题需要什么知识,判断提供的知识是否满足要求,在最后一行输出最终的结果,最后一行的格式为: `评价结果: 0`

# 最终输出结果的判定标准
1. 知识和问题相关但不能完全用来支持解决问题, 输出0
2. 知识完全能支撑回答问题, 输出1

#问题
{query}
#知识
{knoeledge}<|im_end|>
<|im_start|>assistant
"""
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.completions.create(
        prompt=prompt, model=args.model, temperature=0, top_p=0.01, max_tokens=1024
    )
    content = response.choices[0].text
    return extract(content), content


def run_auto_validate_result(result_path: Path):
    """测试结果自动化验证
    1. 相关性
    2. 完整度
    """
    df = pd.read_excel(result_path)
    df[
        [
            "相关性(二值)",
            "相关性(二值)-推理过程",
            "完整度(二值)",
            "完整度(二值)-推理过程",
        ]
    ] = ""
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        length, future_data = 0, {}
        for idx, row in df.iterrows():
            if row.是否查知识 == "Yes" and df.loc[idx, "相关性(二值)"] not in [
                "0",
                "1",
            ]:
                future = pool.submit(
                    run_validate_releated_binary,
                    row.原始问题,
                    row.引用知识,
                )
                length += 1
                future_data[future] = idx
        for future in tqdm(
            as_completed(future_data), total=length, desc="执行相关性验证-ing"
        ):
            idx = future_data[future]
            is_related, content = future.result()
            df.loc[idx, "相关性(二值)"] = is_related
            df.loc[idx, "相关性(二值)-推理过程"] = content

        # 对存在相关性的数据执行完整度推理验证
        length, future_data = 0, {}
        for idx, row in df.iterrows():
            if row["相关性(二值)"] == "1" and df.loc[idx, "完整度(二值)"] not in [
                "0",
                "1",
            ]:
                future = pool.submit(
                    run_validate_integrity_binary,
                    row.原始问题,
                    row.引用知识,
                )
                length += 1
                future_data[future] = idx
        for future in tqdm(
            as_completed(future_data), total=length, desc="执行完整度验证-ing"
        ):
            idx = future_data[future]
            intergrity_result, content = future.result()
            df.loc[idx, "完整度(二值)"] = intergrity_result
            df.loc[idx, "完整度(二值)-推理过程"] = content
    result_valid_save_path = result_path.parent.joinpath(
        f"{rsps_cache_path.stem}_result_{args.model}_valid.xlsx"
    )
    df.to_excel(result_valid_save_path)


async def main(query: str = "成人糖尿病夏天吃什么比较合适?"):
    # Define the agent
    global rsps_cache_path
    global processed_query

    root_path = Path(__file__).parents[1]
    query_file_path = root_path.joinpath(".cache/kb_valid_query.json")
    rsps_cache_path = root_path.joinpath(".cache", "valid_ver1.1.jsonl")
    if not rsps_cache_path.exists():
        rsps_cache_path.touch()

    all_query = json5.load(open(query_file_path, "r"))
    processed_query = [i[0]["content"] for i in jsonlines.open(rsps_cache_path)]
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        features = [
            pool.submit(send_a_req, id, query)
            for id, query in enumerate(all_query)
            if query not in processed_query
        ]
        for feature in as_completed(features):
            rsp = feature.result()
    result_save_path = parse_result(rsps_cache_path)
    run_auto_validate_result(result_save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1, help="线程池并发量")
    parser.add_argument(
        "--base_url",
        type=str,
        # default="http://10.39.91.251:40024/v1",
        default="http://10.228.67.99:26932/v1",
        help="openai-like base_url",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-UuG2Rssx35xS7RHdF5E1E4Ad3e04435f844c7a4e5a4f6bF4",
        help="api key",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen2-72B-Instruct", help="模型名称"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main())
