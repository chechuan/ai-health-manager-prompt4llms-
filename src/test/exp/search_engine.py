# -*- encoding: utf-8 -*-
'''
@Time    :   2024-01-16 11:37:04
@desc    :   XXX
@Author  :   ticoAg
@Contact :   1627635056@qq.com
'''
import re
import timeit

from duckduckgo_search import DDGS

query = "高血压应该怎么治疗"
def search_engine():
    with DDGS(proxies="http://127.0.0.1:7890") as ddgs:
        for i in ddgs.text(query,
                            region="cn-zh",
                            safesearch="moderate",
                            timelimit="y",
                            max_results=1):
            print(i)

def search_engine_1():
    for i in ddgs.text(query, region="cn-zh", safesearch="moderate", timelimit="y", max_results=1):
        print(i)



cost = timeit.timeit(search_engine, number=10)
print(f"type 0 cost {cost:.5f}s")

ddgs = DDGS(proxies="http://127.0.0.1:7890")
cost = timeit.timeit(search_engine_1, number=10)
print(f"type 1 cost {cost:.5f}s")