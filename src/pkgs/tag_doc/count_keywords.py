# path = "src/format_tranfer/data/三高共管规范化诊疗中国专家共识.pdf"
# path = "src/format_tranfer/data/H型高血压诊断与治疗专家共识.pdf"
# from langchain.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader(path, mode="elements")   # Title and NarrativeText
# print(loader.load())

# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader(path)
# pages = loader.load()
# content = "\n\n".join([i.page_content.replace("\n", "").replace(" ", "") for i in pages])

# from pdf2docx import Converter

# pdf_file = "src/format_tranfer/data/H型高血压诊断与治疗专家共识.pdf"
# docx_file = 'src/format_tranfer/data/H型高血压诊断与治疗专家共识.docx'

# # convert pdf to docx
# cv = Converter(pdf_file)
# cv.convert(docx_file) # 默认参数start=0, end=None
# cv.close()

import json
import time
import requests
import threading
import jieba.analyse
from pathlib import Path
from tqdm import tqdm
jieba.load_userdict("vocab.txt")

file_dir = Path("data")
txt_files = file_dir.glob("*.txt")
keyword_per_file = {}
keyword_per_file_textrank = {}


# 从文本文件中读取文本内容，并作简单清洗
for txt_path in tqdm(txt_files):
    tst = time.time()
    with open(txt_path, 'r', encoding='utf-8') as file:
        # 去除换行符，使文本连续
        text = file.read().replace('\n', '').replace(' ', '')
    filename = txt_path.stem
    print(f"process file {filename} - words {len(text)}")

    # seg_out = pku_seg.cut(TEXT)
    # seg_out0 = pku_seg0.cut(TEXT)
    # keywords = Rake().apply(text)

    # # 使用 jieba 进行 TF-IDF 算法提取文本关键词
    keywords = jieba.analyse.extract_tags(
        sentence=text,    # 文本内容
        topK=50,          # 提取的关键词数量
        allowPOS=['n','nz','v', 'vd', 'vn', 'ns', 'nr'],  # 允许的关键词的词性
        withWeight=True,  # 是否附带词语权重
        withFlag=True,    # 是否附带词语词性
    )
    keywords_textrank = jieba.analyse.textrank(
        sentence=text,    # 文本内容
        topK=50,          # 提取的关键词数量
        allowPOS=['n', 'nz', 'v', 'vd', 'vn', 'ns', 'nr'],  # 允许的关键词的词性
        withWeight=True,  # 是否附带词语权重
        withFlag=True,    # 是否附带词语词性
    )
    keyword = {item[0].word: {"attr": item[0].flag, "weight": item[1]}
               for item in keywords}
    keywords_textrank = {item[0].word: {"attr": item[0].flag, "weight": item[1]}
                         for item in keywords_textrank}
    keyword_per_file[filename] = keyword
    keyword_per_file_textrank[filename] = keywords_textrank
    tcost = round(time.time() - tst, 2)
    print(f"{filename} cost - {tcost}s")
# 输出提取到的关键词
save_path = file_dir.parent.joinpath("keyword_per_file.json")
save_path_textrank = file_dir.parent.joinpath("keyword_per_file_textrank.json")
json.dump(keyword_per_file, 
          open(save_path, "w", encoding="utf-8"),
          ensure_ascii=False, 
          indent=4)
json.dump(keyword_per_file_textrank, 
          open(save_path_textrank, "w", encoding="utf-8"),
          ensure_ascii=False, 
          indent=4)

# 输出txt
pth = Path("keyword_per_file_textrank.json")
pth1 = pth = Path("keyword_per_file.json")
dt = json.load(open(pth, "r", encoding="utf-8"))
dt1 = json.load(open(pth1, "r", encoding="utf-8"))
text = ""
for k, item in dt.items():
    item1 = dt1[k]
    dict0 = {k: v['weight'] for k, v in item.items()}
    dict1 = {k: v['weight'] for k, v in item1.items()}

    merged = {**dict0, **dict1}

    for ik, iitem in merged.items():
        text += f"{k}\t{ik}\t{iitem}\n"
open("result.txt", "w").write(text)









# url = "http://10.228.67.99:26921/v1/chat/completions"
# payload = {
#     # "model": "Qwen1.5-14B-Chat",
#     "model": "Baichuan2-13B-Chat",
#     "messages": [
#         {
#             "role": "user",
#             "content": "写一个不短于300字的长篇玄幻小说"
#         }
#     ],
#     "max_tokens": 200
# }
# headers = {"content-type": "application/json"}

# def script(group_id):
#     semaphore.acquire()
#     st = time.time()
#     response = requests.request("POST", url, json=payload, headers=headers)
#     print(f"cost {time.time() - st}s")
#     semaphore.release()

# def one_round(num, func):
#     ts = []
#     for n in range(num):
#         t = threading.Thread(target=func, args=(n,))
#         ts.append(t)
#     for t in ts:
#         t.daemon = True
#         t.start()
#     for t in ts:
#         t.join()

# def extract_with_llm(text, num=20, func=script):
#     global semaphore
#     semaphore = threading.Semaphore(num)
#     trs = []
#     for r in range(round):
#         tr = threading.Thread(target=one_round, args=(num, func))
#         trs.append(tr)
#     for tr in trs:
#         tr.daemon = True
#         tr.start()
#     for tr in trs:
#         tr.join()
#         print(f"测试{r}结束")
