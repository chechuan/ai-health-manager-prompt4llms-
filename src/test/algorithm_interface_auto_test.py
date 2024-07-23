# -*- encoding: utf-8 -*-
"""
@Time    :   2024-07-23 17:01:44
@desc    :   算法接口自动化测试
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

import asyncio
import aiohttp
import pandas as pd
import sys
import json
import time
import os
from loguru import logger

# 配置日志格式
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# 添加配置到logger
logger.remove()  # 移除默认的日志配置
logger.add(sys.stdout, format=LOG_FORMAT, level="DEBUG")

# 获取 extract_logs.py 文件所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# 配置参数
CONFIG = {
    "env": "dev",  # 环境: 'local' 或 'dev'
    "url_prefix": {
        'local': 'http://0.0.0.0:6500',
        'dev': 'http://ai-health-manager-prompt4llms.data-engine-dev.laikang.enn.cn'
    },
    "log_file": os.path.join(root_dir, '.cache', 'logs.jsonl'),  # 日志文件路径
    "timeout": 1200,  # 请求超时时间，单位为秒
    "max_tests": 2,  # 每个端点的最大测试次数
    "output_basic_test": os.path.join(root_dir, '.cache', 'basic_test_results.xlsx'),  # 基本测试结果输出文件
    "output_final_test": os.path.join(root_dir, '.cache', 'final_test_results.xlsx'),  # 最终测试结果输出文件
    "special_params": {  # 特异化参数配置
        "default": [
            # {"user_profile": None},
            # {"user_profile": ""},
            # {"key_indicators": None},
            # {"key_indicators": ""}
        ],
        # "/aigc/functions/endpoint1": [
        #     {"user_profile": {"age": None, "gender": None, "height": None, "weight": None}},
        # ],
        # "/aigc/functions/endpoint2": [
        #     {"key_indicators": {"indicator1": None, "indicator2": None}}
        # ],
        # 为其他端点添加特异化参数
    },
    # 特定EndPoint(如/aigc/functions) 根据指定字段区分测试数量
    "field_test_limits": {
        # "/aigc/functions": {"intentCode": 5}
    },
    "concurrent_requests": 5  # 并发请求数量
}

# 获取当前环境的URL前缀
URL_PREFIX = CONFIG["url_prefix"][CONFIG["env"]]


# 将字典转换为JSON字符串
def dict_to_json_str(data):
    return json.dumps(data, ensure_ascii=False, indent=4)


# 从日志文件中加载日志数据
def load_log_data(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        logs = [json.loads(line) for line in f]
    return logs


# 验证响应结果的类型
def validate_result_type(response, expected_type):
    try:
        assert isinstance(response, expected_type), f"Expected {expected_type}, but got {type(response)}"
    except AssertionError as e:
        return str(e)
    return None


# 测试单个端点
async def test_endpoint(session, log, i, semaphore):
    async with semaphore:
        url = URL_PREFIX + log["endpoint"]
        logger.info(f"(line {i + 1}) Sending request to {url} with data: {log['input_param']}")
        result = {
            "endpoint": log["endpoint"],
            "data": dict_to_json_str(log["input_param"]),
            "result": None,
            "error": None
        }
        try:
            # 发送POST请求
            async with session.post(url, json=log["input_param"], timeout=CONFIG["timeout"]) as response:
                content_type = response.headers.get('Content-Type', '')

                # 检查Content-Type是否表明这是一个SSE流式响应
                if 'text/event-stream' in content_type:
                    sse_events = []

                    async for line in response.content:
                        line = line.decode('utf-8').strip()

                        if line.startswith('data:'):
                            data_line = line[5:]  # 去掉'data:'前缀
                            if data_line:  # 确保不是空数据行
                                try:
                                    event_data = json.loads(data_line)
                                    sse_events.append(event_data)
                                except json.JSONDecodeError:
                                    # 如果JSON解析失败，记录错误
                                    logger.error(f"Failed to parse JSON from SSE event: {data_line}")
                                    result["error"] = "Failed to parse JSON from SSE event"
                                    break

                    result["result"] = sse_events
                else:
                    try:
                        # 解析JSON响应
                        response_json = await response.json()
                        error = validate_result_type(response_json, dict)
                        result["result"] = response_json
                        result["error"] = error
                    except aiohttp.ContentTypeError:
                        # 如果Content-Type不是JSON，解析为文本
                        response_text = await response.text()
                        result["result"] = response_text
                        result["error"] = "Received text/plain response instead of JSON"
                logger.info(f"Request to {url} succeeded with response: {result['result']}")
        except aiohttp.ClientError as e:
            # 捕获客户端错误
            logger.error(f"Request to {url} failed: {e}")
            result["error"] = str(e)
        except json.JSONDecodeError as e:
            # 捕获JSON解析错误
            logger.error(f"JSON decode failed for {url}: {e}")
            result["error"] = "JSON decode error"
        except asyncio.TimeoutError as e:
            # 捕获请求超时错误
            logger.error(f"Request to {url} timed out: {e}")
            result["error"] = "Timeout error"

        return result


# 测试所有端点
async def test_endpoints(logs):
    results = []
    semaphore = asyncio.Semaphore(CONFIG["concurrent_requests"])
    async with aiohttp.ClientSession() as session:
        tasks = [test_endpoint(session, log, i, semaphore) for i, log in enumerate(logs)]
        responses = await asyncio.gather(*tasks)
        results.extend(responses)
    return results


# 将测试结果保存到Excel文件
def save_results_to_excel(results, output_file):
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


# 测试单个端点的请求数量限制
async def test_endpoint_limit(session, log, max_tests, semaphore):
    async with semaphore:
        endpoint = log["endpoint"]
        url = URL_PREFIX + endpoint
        unique_results = set()
        logger.info(f"Testing endpoint limits for {url}")
        try:
            async with session.post(url, json=log["input_param"], timeout=CONFIG["timeout"]) as response:
                try:
                    response_json = await response.json()
                    unique_results.add(str(response_json))
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    unique_results.add(response_text)
                if len(unique_results) >= max_tests:
                    logger.info(f"Reached max test limit for {endpoint}")
                    return {endpoint: len(unique_results)}
        except aiohttp.ClientError as e:
            logger.error(f"Request to {url} failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for {url}: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Request to {url} timed out: {e}")
        return {endpoint: len(unique_results)}


# 测试所有端点的请求数量限制
async def test_endpoint_limits(logs, max_tests=None):
    results = []
    semaphore = asyncio.Semaphore(CONFIG["concurrent_requests"])
    async with aiohttp.ClientSession() as session:
        tasks = [test_endpoint_limit(session, log, max_tests, semaphore) for log in logs]
        responses = await asyncio.gather(*tasks)
        for res in responses:
            results.append(res)
    return results


# 测试端点的特异化参数
async def test_special_parameter(session, log, param, semaphore):
    async with semaphore:
        data = log["input_param"].copy()
        data.update(param)
        endpoint = log["endpoint"]
        url = URL_PREFIX + endpoint
        result = {
            "endpoint": endpoint,
            "data": dict_to_json_str(data),
            "result": None,
            "error": None
        }
        try:
            # 发送带特异化参数的请求
            async with session.post(url, json=data, timeout=CONFIG["timeout"]) as response:
                try:
                    response_json = await response.json()
                    result["result"] = response_json
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    result["result"] = response_text
                    result["error"] = "Received text/plain response instead of JSON"
                logger.info(f"Request to {url} succeeded with response: {result['result']}")
        except aiohttp.ClientError as e:
            logger.error(f"Request to {url} failed: {e}")
            result["error"] = str(e)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for {url}: {e}")
            result["error"] = "JSON decode error"
        except asyncio.TimeoutError as e:
            logger.error(f"Request to {url} timed out: {e}")
            result["error"] = "Timeout error"
        return result


# 测试所有端点的特异化参数
async def test_special_parameters(log):
    special_params = CONFIG["special_params"].get(log["endpoint"], CONFIG["special_params"]["default"])
    results = []
    semaphore = asyncio.Semaphore(CONFIG["concurrent_requests"])
    async with aiohttp.ClientSession() as session:
        tasks = [test_special_parameter(session, log, param, semaphore) for param in special_params]
        responses = await asyncio.gather(*tasks)
        results.extend(responses)
    return results


# 运行所有测试
async def run_tests(logs):
    test_results = []
    semaphore = asyncio.Semaphore(CONFIG["concurrent_requests"])
    async with aiohttp.ClientSession() as session:
        limit_tasks = []
        special_param_tasks = []
        for log in logs:
            field_limits = CONFIG.get("field_test_limits", {}).get(log["endpoint"])
            if field_limits:
                field = list(field_limits.keys())[0]
                max_tests = field_limits[field]
                field_value = log["input_param"].get(field)
                if field_value:
                    limit_tasks.append(test_endpoint_limits(
                        [log for log in logs if log["input_param"].get(field) == field_value], max_tests))
                else:
                    limit_tasks.append(test_endpoint_limits([log], CONFIG["max_tests"]))
            else:
                limit_tasks.append(test_endpoint_limits([log], CONFIG["max_tests"]))

            special_param_tasks.append(test_special_parameters(log))

        limit_responses = await asyncio.gather(*limit_tasks)
        special_param_responses = await asyncio.gather(*special_param_tasks)

        for i, log in enumerate(logs):
            limit_test_result = limit_responses[i]
            special_param_results = special_param_responses[i]
            test_results.append({
                "endpoint": log["endpoint"],
                "limit_test_result": limit_test_result,
                "special_param_results": special_param_results
            })
    return test_results

def main():
    start_time = time.time()
    log_file = CONFIG["log_file"]
    logs = load_log_data(log_file)  # 读取日志数据
    logger.info(f"Loaded {len(logs)} logs from {log_file}")

    # 运行基本测试并保存结果到Excel
    loop = asyncio.new_event_loop()
    basic_test_results = loop.run_until_complete(test_endpoints(logs))
    save_results_to_excel(basic_test_results, CONFIG["output_basic_test"])

    # 运行数量限制和特异化参数测试，并保存结果到Excel
    final_results = loop.run_until_complete(run_tests(logs))
    save_results_to_excel(final_results, CONFIG["output_final_test"])

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()