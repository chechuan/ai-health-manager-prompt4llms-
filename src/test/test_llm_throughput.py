# -*- encoding: utf-8 -*-
"""
@Time    :   2024-07-02 15:18:22
@desc    :   llm并发压测
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import argparse
import asyncio
import time
from typing import List

import openai
from prettytable import PrettyTable
from loguru import logger

short_prompt = """请给我20个保持健康的建议吧."""

long_prompt = """# 已知信息
## 用户标签
年龄：未知
性别：未知
身高：未知
体重：未知
现患疾病：未知
管理目标：未知
食物过敏：大豆
特殊饮食习惯：未知
口味偏好：未知
是否特殊生理期：未知
中医体质：湿热质
## 今日日期
2024年6月5日
## 所处地域
北京
## 会话历史记录
assistant: 请问你是否有其他饮食偏好？
user:我喜欢吃甜的
assistant: 你是否有运动习惯？
user:没有
## 饮食调理原则
夏季已至，北京的天气逐渐炎热，此时养心为重。夏季阳气旺盛，应以清淡、易消化的食物为主，帮助调理肠胃。推荐多吃蔬果，如西瓜、苦瓜，以清热解暑，同时补充水分。避免过于油腻和辛辣，以防加重肠胃负担。如有疾病在身，特别是心脏疾病，更应注意避免冷饮，以免刺激心脏。保持饮食规律，助于养心护胃，保持身体健康。
## 历史食谱
2024年5月5日：
早餐：豆腐脑,鸡蛋,凉拌芹菜
午餐：大米饭,清炒油麦菜,红烧鸡翅,芹菜汁
晚餐：玉米,鸡腿,牛奶
2024年5月6日：
早餐：豆腐脑,鸡蛋,凉拌芹菜
午餐：大米饭,清炒油麦菜,红烧鸡翅,芹菜汁
晚餐：玉米,鸡腿,牛奶

## 今日食谱
早餐：豆浆,鸡蛋三明治
# 输出餐次
午餐

# 任务描述
你是一位经验丰富的智能营养师，请你根据输出要求、我的已知信息、我的饮食调理原则，为我输出要求的餐次食谱。
# 输出要求
## 内容输出要求
- 每餐应该尽量包含谷薯类、蔬菜类、肉蛋奶豆类，每天都要有水果，保证膳食结构均衡
- 你需要重点关注节气、地域信息，推荐适合当前节气以及地域性食物的食谱
- 你需要注意我的特殊饮食习惯，比如素食者，避免输出肉类食物
- 你需要考虑我的口味偏好，在健康的基础上适当推荐
- 你需要考虑我是否在特殊生理期，根据不同情况合理推荐
- 输出的食谱请参考前2天的食谱，注意输出的相邻3天的菜品应有所差异，符合食物多样性的原则，提高用户体验度，并且要符合我的饮食调理原则
- 每餐的食物数量请尽量不超过4个
- 请避推荐包含食物过敏的食材，如大豆过敏，不要推荐豆浆、豆腐等大豆类食物
- 请避免今天三餐的菜品重复
- 每餐的营养价值说明请重点从中医养生、药食同源、营养学角度说明整餐食物的功效以及搭配原理
- 每餐的营养价值说明输出的内容应该通俗易懂
- 每餐的营养价值说明可以说明每个食物重点提供的营养价值
- 请避免重复说明我的要求，直接输出营养价值内容

## 格式输出要求
- 请按markdown格式输出，二级标题内容为`餐次和食物名称`，格式上`餐次和食物名称`为一个整体，不同食物名称之间用`、`隔开；正文内容为`营养价值`，严格按照'输出样例'markdown格式输出。
- 纯净模式，请只输出食谱结果
- 请注意每餐中推荐一个饮品，如奶类、茶饮、水果或蔬菜汁等，放在正餐中
- 请避免推荐食物质量或单位
- 请避免输出关于`注意`类的内容
- 每餐的营养价值说明整体输出字符尽量不超过48个

# 输出样例：
## 早餐：全麦面包、黑芝麻糊、煎鸡蛋、番茄
黑芝麻糊补肝肾，滋养乌发；煎鸡蛋提供优质蛋白质，益智健脑；番茄清热利尿，维生素C丰富。此搭配养心护眼，提供早晨所需的活力与营养。

Begins!"""


async def send_a_request(id, completion_tokens_list):
    if args.prompt_type == "long":
        prompt = long_prompt
    elif args.prompt_type == "short":
        prompt = short_prompt
    time_st = time.time()
    # 使用 OpenAI 的 Completion API
    logger.debug(f"Send request {id}/{args.num_requests}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = await aclient.chat.completions.create(
        model=args.model,  # 请根据你的订阅选择合适的模型
        messages=messages,
        max_tokens=4096,
        temperature=1,
        top_p=0.8,
        stream=False,
    )
    response_data = response.model_dump()
    response_data["usage"]["cost"] = round(time.time() - time_st, 2)
    completion_tokens_list.append(response_data["usage"])
    logger.debug(
        {
            "id": id,
            "conversation_id": response_data["id"],
            "usage": response_data["usage"],
        }
    )


async def control_concurrency(task, semaphore):
    # 控制并发量
    async with semaphore:
        await task


async def perform_load_test() -> List:
    """
    模拟并发请求
    """
    global aclient

    aclient = openai.AsyncClient(
        base_url=args.base_url, api_key=args.api_key, max_retries=3
    )

    completion_tokens_list = []
    # 创建信号量，限制并发量
    semaphore = asyncio.Semaphore(args.concurrency)
    # 创建并发任务
    tasks = [
        control_concurrency(send_a_request(id + 1, completion_tokens_list), semaphore)
        for id in range(args.num_requests)
    ]

    # 等待所有任务完成
    await asyncio.gather(*tasks)

    # 删除首尾的几个请求，因为这些请求是warmup请求
    completion_tokens_list = completion_tokens_list[
        args.concurrency * 2 : -2 * args.concurrency
    ]
    # 计算平均吞吐量
    if len(completion_tokens_list) > 0:
        average_tokens = sum(
            [i["completion_tokens"] for i in completion_tokens_list]
        ) / sum([i["cost"] for i in completion_tokens_list])
        average_completion_tokens = sum(
            [i["completion_tokens"] for i in completion_tokens_list]
        ) / len(completion_tokens_list)
        average_prompt_tokens = sum(
            [i["prompt_tokens"] for i in completion_tokens_list]
        ) / len(completion_tokens_list)
        return [
            args.concurrency,
            args.num_requests,
            f"{average_prompt_tokens:.2f}",
            f"{average_completion_tokens:.2f}",
            f"{average_tokens:.2f}",
            f"{average_tokens*concurrency:.2f}",
        ]


def init_args():
    """初始化参数"""
    parser = argparse.ArgumentParser(description="LLM Throughput Test")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://10.228.67.99:26930/v1",
        help="openai-like api 请求地址",
    )
    parser.add_argument(
        "--api-key", type=str, default="empty", help="openai-like api key"
    )
    parser.add_argument(
        "--concurrency-lst",
        type=str,
        default="1,5,10,20,50",
        # default="1,2",
        help="请求并发量",
    )
    parser.add_argument(
        "--alpha-times",
        type=int,
        default=10,
        help="总请求数=倍率*并发数, 注: 开始的2*并发和结束的2*并发数会被忽略，保证warmup和warmdown",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="long",
        choices=["long", "short"],
        help="测试类型，长提示 or 短提示",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2-7B-Instruct",
        help="模型名称",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = init_args()
    table = PrettyTable()
    table.field_names = [
        "Concurrency",
        "Num of Requests",
        "Average Prompt Tokens",
        "Average Completion Tokens",
        "Generation Average Throughput",
        "All Generation Thoughput",
    ]
    args.concurrency_lst = [int(i) for i in args.concurrency_lst.split(",")]
    for concurrency in args.concurrency_lst:
        args.concurrency = concurrency
        args.num_requests = args.alpha_times * concurrency
        table.add_row(asyncio.run(perform_load_test()))
    print(table)
