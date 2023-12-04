# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-04 15:10:41
@desc    :   https://blog.csdn.net/qq_29654777/article/details/104051567 搜索api爬虫
@Author  :   宋昊阳, 员庆阳
@Contact :   1627635056@qq.com
'''
import asyncio
import json
import re
import sys
from pathlib import Path

from numpy import tile

sys.path.append(str(Path.cwd().absolute()))
from datetime import datetime
from typing import Dict, List

import requests
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from lxml import etree

from src.pkgs.knowledge.config.prompt_config import PROMPT_TEMPLATES
from src.utils.Logger import logger


def get_template(key: str) -> str:
    """根据关键字取templates
    """
    assert PROMPT_TEMPLATES.get(key), f"{key} not in PROMPT_TEMPLATES"
    return PROMPT_TEMPLATES[key]

async def parse_detail_page(d_url):
    try:
        async with ClientSession() as session:
            async with session.get(d_url) as response:
                res_text = await response.text()
    except Exception as e:
        return None, None
    json_data = re.findall('type="application/json">(.*?)</script>', res_text, re.S)
    data = json.loads(json_data[0])
    tags_str = data['props']['pageProps']['contentData']['content'][0]
    title = data['props']['pageProps']['commonData']['title']
    soup = BeautifulSoup(tags_str, 'html.parser')
    text = ""
    for p in soup.find_all('p'):
        text += p.text
    return title, text

async def search_engine_chat(query: str,
                             top_k: int = 3,
                             session = requests.Session(),
                             **kwargs) -> str:
    url = f"http://www.baidu.com/s?wd={query}&cl=3&pn=1&ie=utf-8&rn={top_k+5}&tn=baidurt"
    response = session.get(url)
    res = etree.HTML(response.text)
    detail_urls = res.xpath('//h3[@class="t"]/a/@href')
    
    task_list = []
    for d_url in detail_urls:
        task_list.append(asyncio.create_task(parse_detail_page(d_url)))
    done, pending = await asyncio.wait(task_list, timeout=None)
    
    content = ""
    i = 0
    for done_task in done:
        title, text = done_task.result()
        if title and i < 3:
            content += f"title: {title}\n{text}\n"
            i += 1
    return content


if __name__ == '__main__':
    query = '早上起来头疼怎么回事'
    text = asyncio.run(search_engine_chat(query))
    print(text)
