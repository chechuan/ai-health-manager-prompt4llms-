import os
from mem0 import Memory
import logging

logging.basicConfig(level=logging.DEBUG)

os.environ["OPENAI_API_KEY"] = "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525"

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "cyh2",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768
        }
    },
    "llm": {
        "provider": "openai_structured",
        "config": {
            "model": "Qwen2.5-7B-Instruct",
            "api_key": "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525",
            "openai_base_url": "http://10.228.67.99:26945/v1",
            "top_p": 0.5
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "bce-embedding-base-v1",
            "openai_base_url": "http://10.228.67.99:26928/v1",
            "api_key": "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525",
            # "embedding_dims": 768
        }
    }
}
uid="yaya"
print("正在初始化 Memory...")
m = Memory.from_config(config)
print("Memory 初始化成功")
print("\n开始添加记忆...")
message = "喜欢游泳"
print(f"输入消息: {message}")
result = m.add(message, user_id=uid)
print("添加完毕")

history = m.history(memory_id=uid)
print(history)

print("\n尝试搜索添加的记忆...")
search_result = m.search(
    query="风筝",
    user_id=uid,
    limit=5
)
print("搜索结果:", search_result)