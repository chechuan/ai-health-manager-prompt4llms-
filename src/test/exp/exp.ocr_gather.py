import json
import re
import sys
import time
from pathlib import Path
from turtle import right

import openai
from rapidocr_onnxruntime import RapidOCR

sys.path.append(str(Path.cwd()))
from src.utils.Logger import logger

openai.api_key = "sk-proxy-laikangjiankangdamoxing-vr-shuziren"
openai.api_base = "http://10.228.67.99:26921/v1"


def dumpJS(obj):
    return json.dumps(obj, ensure_ascii=False)


def accept_stream_response(response, verbose=True):
    content = ""
    tst = time.time()
    for chunk in response:
        if chunk.get("object") == "text_completion":
            if hasattr(chunk["choices"][0], "text"):
                chunk_text = chunk["choices"][0]["text"]
                content += chunk_text
                if verbose:
                    print(chunk_text, end="", flush=True)
        else:
            if hasattr(chunk["choices"][0]["delta"], "content"):
                chunk_text = chunk["choices"][0]["delta"]["content"]
                content += chunk_text
                if verbose:
                    print(chunk_text, end="", flush=True)
    t_cost = round(time.time() - tst, 2)
    logger.debug(
        f"Model {chunk['model']}, Generate {len(content)} words, Cost {t_cost}s"
    )
    return content


ocr = RapidOCR(lang="zh")

result, _ = ocr(
    "/home/tico/workspace/ai-health-manager-prompt4llms/.tmp/images/肺炎.jpeg"
)


sysprompt = """You are a helpful assistant.
# 任务描述
1. 下面我将给你报告OCR提取的内容，它是有序的，优先从上到下从左到右
2. 请你参考给出的内容的前后信息，对报告的内容进行归类，类别不少于4个
3. 只给出分类开始和结尾内容对应的index, 相邻分类的index应当是相连的
4. 给出的index不应超出给定的内容
5. 输出格式参考:```json\n{"分类1": [start_idx, end_idx]}\n```
"""


content_index = {idx: text for idx, text in enumerate([i[1] for i in result])}
messages = [
    {"role": "system", "content": sysprompt},
    {"role": "user", "content": str(content_index)},
]

print(json.dumps(messages, ensure_ascii=False))
client = openai.OpenAI()
# response = openai.chat.completions.create(
#     model="Qwen-72B-Chat",
#     messages=messages,
#     temperature=0.7,
#     n=1,
#     top_p=0.8,
#     top_k=-1,
#     presence_penalty=0,
#     frequency_penalty=0.5,
#     stream=True,
# )
# content = accept_stream_response(response, verbose=True)
content = """```json
{"基本信息": [0, 8], "影像信息": [9, 16], "检查结果": [18, 22], "其他信息": [23, 23]}
```"""
content = re.findall("```json(.*?)```", content, re.S)[0].strip()

try:
    loc = json.loads(content)
except:
    loc = {}

rectangles_with_text = []
for topic, index_range in loc.items():
    start_idx, end_idx = index_range
    start_msg, end_msg = result[start_idx], result[end_idx]
    left_top, right_bottom = start_msg[0][0], end_msg[0][2]
    rectangles_with_text.append((tuple(left_top + right_bottom), topic))

print(rectangles_with_text)
