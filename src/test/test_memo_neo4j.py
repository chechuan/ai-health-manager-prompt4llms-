import os
from mem0 import Memory
import logging

# logging.basicConfig(level=logging.DEBUG)

# os.environ["OPENAI_API_KEY"] = "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525"

# config = {
#     "vector_store": {
#         "provider": "qdrant",
#         "config": {
#             "collection_name": "cyh2",
#             "host": "localhost",
#             "port": 6333,
#             "embedding_model_dims": 768
#         }
#     },
#     "graph_store": {
#         "provider": "neo4j",
#         "config": {
#             "url": "neo4j://localhost:7687",
#             "username": "neo4j",
#             "password": "Cc304589~"
#         }
#     },
#     "version": "v1.1",
#     "llm": {
#         "provider": "openai_structured",
#         "config": {
#             "model": "Qwen2.5-7B-Instruct",
#             "api_key": "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525",
#             "openai_base_url": "http://10.228.67.99:26945/v1",
#             "top_p": 0.5
#         }
#     },
#     "embedder": {
#         "provider": "openai",
#         "config": {
#             "model": "bce-embedding-base-v1",
#             "openai_base_url": "http://10.228.67.99:26928/v1",
#             "api_key": "sk-fOQllxzXJ7SEHGVO2b426e829d2f496792Ac902b11Cd0525",
#             "embedding_dims": 768
#         }
#     }
# }
# uid="yaya"
# print("正在初始化 Memory...")
# m = Memory.from_config(config)
# print("Memory 初始化成功")

# message = "喜欢吃泡面"
# print(f"输入消息: {message}")
# result = m.add(message, user_id=uid)
# print("添加完毕")


# history = m.history(memory_id=uid)
# print(history)

# search_result = m.search(
#     query="风筝",
#     user_id=uid,
#     limit=5
# )
# print("搜索结果:", search_result)


from datetime import datetime, timedelta

# 示例数据
glucose_data = [
    {"item_name": "随机血糖", "item_value": "8.5", "create_time": "2025-01-15 11:23:47"},
    {"item_name": "随机血糖", "item_value": "8.5", "create_time": "2024-01-15 16:23:47"},
    {"item_name": "随机血糖", "item_value": "8.5", "create_time": "2025-01-15 16:23:47"},
    {"item_name": "随机血糖", "item_value": "8.6", "create_time": "2025-01-15 16:22:05"},
    {"item_name": "随机血糖", "item_value": "8.6", "create_time": "2025-01-15 16:17:05"},
    {"item_name": "随机血糖", "item_value": "8.6", "create_time": "2025-01-15 15:22:05"},
    {"item_name": "随机血糖", "item_value": "8.8", "create_time": "2025-01-15 16:21:19"},
    {"item_name": "随机血糖", "item_value": "9.0", "create_time": "2025-01-15 16:20:26"},
    {"item_name": "随机血糖", "item_value": "9.1", "create_time": "2025-01-15 16:19:10"},
    {"item_name": "随机血糖", "item_value": "9.1", "create_time": "2025-01-15 16:24:10"},
    {"item_name": "随机血糖", "item_value": "7.2", "create_time": "2025-01-15 13:27:10"},
    {"item_name": "随机血糖", "item_value": "8.3", "create_time": "2025-01-15 13:37:10"},
    {"item_name": "随机血糖", "item_value": "6.1", "create_time": "2025-01-15 13:42:10"},
    {"item_name": "随机血糖", "item_value": "9.1", "create_time": "2025-01-15 16:25:10"}
]

# 假设当前时间
rencent_time = "2025-01-15 16:27:10"  # 示例时间，可以替换为实际时间
rencent_time = datetime.strptime(rencent_time, "%Y-%m-%d %H:%M:%S")

# 将时间字符串转换为datetime对象
for entry in glucose_data:
    entry["create_time"] = datetime.strptime(entry["create_time"], "%Y-%m-%d %H:%M:%S")

# 按时间排序
glucose_data.sort(key=lambda x: x["create_time"])

# 过滤出在rencent_time前3小时内的数据
recent_3h_data = [entry for entry in glucose_data if entry["create_time"] >= rencent_time - timedelta(hours=3) and entry["create_time"] <= rencent_time]

# 采样逻辑
sampled_data = []
time_threshold_2h = rencent_time.replace(second=0, microsecond=0) - timedelta(hours=2)
time_threshold_3h = rencent_time.replace(second=0, microsecond=0) - timedelta(hours=3)

# 近2h内，每5分钟取一个
last_2h_samples = []
current_time = time_threshold_2h
while current_time < time_threshold_2h + timedelta(hours=2):
    for entry in recent_3h_data:
        if entry["create_time"].replace(second=0, microsecond=0) == current_time:
            last_2h_samples.append(entry)
            break
    current_time += timedelta(minutes=5)

# 近3h-2h内，每10分钟取一个
last_3h_2h_samples = []
current_time = time_threshold_3h
while current_time < time_threshold_2h:
    for entry in recent_3h_data:
        if entry["create_time"].replace(second=0, microsecond=0) == current_time:
            last_3h_2h_samples.append(entry)
            break
    current_time += timedelta(minutes=10)

# 合并采样数据
sampled_data.extend(last_2h_samples)
sampled_data.extend(last_3h_2h_samples)

# 找出波峰波谷
def find_peaks_and_valleys(data):
    peaks = []
    valleys = []
    for i in range(1, len(data)-1):
        if data[i]["item_value"] > data[i-1]["item_value"] and data[i]["item_value"] > data[i+1]["item_value"]:
            peaks.append((data[i]["create_time"], data[i]["item_value"]))
        if data[i]["item_value"] < data[i-1]["item_value"] and data[i]["item_value"] < data[i+1]["item_value"]:
            valleys.append((data[i]["create_time"], data[i]["item_value"]))
    return peaks, valleys

peaks, valleys = find_peaks_and_valleys(recent_3h_data)

# 输出结果
print("Sampled Data:")
for entry in sampled_data:
    print(entry)

print("\nPeaks:")
for peak in peaks:
    print(peak)

print("\nValleys:")
for valley in valleys:
    print(valley)
