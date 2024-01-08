"""
pip install openai==0.28.1 -i https://mirrors.bfsu.edu.cn/pypi/web/simple
"""
import time

import openai

openai.api_key = "sk-proxy-laikangjiankangdamoxing-vr-shuziren"
openai.api_base = "http://aimp-algo-openai-proxy.en.laikang.com/v1"

messages = [
    {"role":"user", "content":"帮我写一首悲凉壮阔的边塞诗"}
]
model_list = [model.id for model in openai.Model.list().data]
print(f"Current Support Model: {model_list}")

start_time = time.time()
model = model_list[0]

print(f"Currnet use model: {model}")

response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    temperature=0.9,
    top_p=0.8,
    top_k=-1,
    repetition_penalty=1.2,
    stream=True
)

response_time = time.time()
print(f'latency {response_time - start_time:.2f} s -> response')

content = ""
printed = False
for i in response:
    t = time.time()
    msg = i.choices[0].delta.to_dict()
    text_stream = msg.get('content')
    if text_stream:
        if not printed:
            print(f'latency first token {t - start_time:.2f} s')
            printed = True
        content += text_stream
        print(text_stream, flush=True, end="")