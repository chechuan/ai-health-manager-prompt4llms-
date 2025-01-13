import os
import requests
import json
import time
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe



langfuse = Langfuse(
    secret_key="sk-lf-dd72d122-597f-4816-be79-605ecdabc575",  # 替换为你的实际密钥
    public_key="pk-lf-006f983d-9437-4d7b-8b01-b17b03a5ab91",  # 替换为你的实际公钥
    host="https://ai-health-manager-langfuse-web.op.laikang.com"  # 替换为你的实际主机地址
)


# @observe()
def call_qwen_model(prompt: str):
    url = "http://10.228.67.99:26928/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525'
    }

    payload = {
        "model": "Qwen2-72B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    # 元数据 需要参考微信doc
    langfuse_context.update_current_observation(
        metadata={
            "url": url,
            "model": "Qwen2-72B-Instruct"
        },
        input=prompt
    )

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()

    # 更新观测
    langfuse_context.update_current_observation(
        output=response_data,
        metadata={
            "status_code": response.status_code,
            "latency": response.elapsed.total_seconds()
        }
    )

    return response_data


@observe(as_type="generation")
def generate_completion(prompt_name: str = "aigc_functions_test1230", age: str = "18"):
    # 编译 prompt
    prompt = langfuse.get_prompt(prompt_name)
    compiled_prompt = prompt.compile(age=age)

    # 更新 generation
    langfuse_context.update_current_observation(
        prompt=prompt,
        model="Qwen2-72B-Instruct",
        model_parameters={
            "temperature": 0.7,
            "max_tokens": 1000
        },
        input=compiled_prompt
    )

    response = call_qwen_model(compiled_prompt)
    output_text = response['choices'][0]['message']['content']
    usage = response.get('usage', {})

    # 更新使用统计
    langfuse_context.update_current_observation(
        output=output_text,
        usage_details={
            "input": usage.get('prompt_tokens', 0),
            "output": usage.get('completion_tokens', 0),
            "total": usage.get('total_tokens', 0)
        },
        cost_details={
            "input": usage.get('prompt_tokens', 0) * 0.00001,
            "output": usage.get('completion_tokens', 0) * 0.00002,
        }
    )

    # 记录响应长度分数
    langfuse_context.score_current_observation(
        name="response_length",
        value=len(output_text)
    )

    return {
        "prompt": compiled_prompt,
        "output": output_text,
        "usage": usage
    }


# @observe()
def process_user_request(user_id: str):
    result = generate_completion()
    return result


def main():
    try:
        # 使用低级 SDK 创建 trace
        trace = langfuse.trace(
            name="qwen-test",
            user_id="user-123",  # 设置用户 ID
            tags=["test", "qwen"],
            metadata={
                "environment": "testing",
                "version": "1.0"
            }
        )

        # 创建 generation
        generation = trace.generation(
            name="qwen-completion",
            model="Qwen2-72B-Instruct",
        )

        # 获取并编译 prompt
        prompt = langfuse.get_prompt("test_1226")
        compiled_prompt = prompt.compile(age="18")

        # 调用模型
        response = call_qwen_model(compiled_prompt)
        output_text = response['choices'][0]['message']['content']
        usage = response.get('usage', {})
        print("compiled_prompt", compiled_prompt)        # 更新 generation
        print("output_text", output_text)        # 更新 generation
        generation.update(
            input=compiled_prompt,
            output=output_text,
            usage_details={
                "input": usage.get('prompt_tokens', 0),
                "output": usage.get('completion_tokens', 0),
                "total": usage.get('total_tokens', 0)
            },
            cost_details={
                "input": usage.get('prompt_tokens', 0) * 0.00001,
                "output": usage.get('completion_tokens', 0) * 0.00002,
            }
        )

        print("编译后的 prompt:", compiled_prompt)
        print("模型响应:", output_text)
        print("使用统计:", usage)

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        langfuse.flush()


if __name__ == "__main__":
    # 示例1：使用装饰器方式
    # print("使用装饰器方式:")
    # result = process_user_request("user-456")
    # print(result)
    # print("\n")

    # 示例2：使用低级 SDK 方式
    print("使用低级 SDK 方式:")
    main()