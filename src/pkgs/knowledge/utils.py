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

sys.path.append(str(Path.cwd().absolute()))
from typing import Any, AnyStr, Dict, List, Optional

import requests
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from lxml import etree

from src.pkgs.knowledge.config.prompt_config import PROMPT_TEMPLATES
from src.utils.Logger import logger


def check_task(task: Any) -> str or None:
    if type(task) is str:
        try:
            task = json.loads(task)['task']   
        except Exception as err:
            logger.exception(err)
            task = None
    elif type(task) is dict:
        task = task['task']
    else:
        task = None
        raise Exception(f"Unknown task type {task}")
    return task
            
def get_template(key: str) -> str:
    """根据关键字取templates
    """
    assert PROMPT_TEMPLATES.get(key), f"{key} not in PROMPT_TEMPLATES"
    return PROMPT_TEMPLATES[key]

def parse_diff_pages(url: str, _html: str) -> (str, str):
    """解析不同的域名
    """
    if 'www.zhihu.com' in url:
        soup = BeautifulSoup(_html, 'html.parser')
        title = soup.find('h1', class_='QuestionHeader-title').text
        for span in soup.select('span'):
            if span.get('class') and "RichText" in span['class']:
                text = "\n".join([p.text for p in span.find_all('p')])
    elif 'zhuanlan.zhihu.com' in url:
        soup = BeautifulSoup(_html, 'html.parser')
        title = soup.find('h1', class_='Post-Title').text
        for div in soup.select('div'):
            if div.get('class') and "RichText" in div['class']:
                text = "\n".join([p.text for p in div.find_all('p')])
    elif 'health.baidu.com' in url:
        html = etree.HTML(_html)
        title = html.xpath("//p[contains(@class, 'hc-line-clamp2')]/text()")[0]
        p_str_list = html.xpath("//div[contains(@class, 'index_textContent ')]//p//text()")
        text = "\n".join(p_str_list)
    elif 'm.baidu.com' in url:
        html = etree.HTML(_html)
        _data = html.xpath("//script[@id='__NEXT_DATA__']//text()")[0]
        data = json.loads(_data)
        title = data['props']['pageProps']['commonData']['title']
        text = data['props']['pageProps']['contentData']['content'][0]
    elif 'youlai.cn' in url:
        html = etree.HTML(_html)
        title = html.xpath('//*[@class="v_title"]/text()')[0]
        text = html.xpath('//*[@class="text"]/text()')[0].strip()
    else:
        title, text = None, None
    return title, text

async def parse_detail_page(d_url):
    try:
        async with ClientSession() as session:
            headers = {
                "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            }
            async with session.get(d_url, headers=headers) as response:
                _html = await response.text()
    except Exception as e:
        return None, None
    
    try:
        title, text = parse_diff_pages(d_url, _html)
        return title, text
    except TypeError as terr:
        logger.debug(f"error to parse {d_url}")
        return None, None
    except Exception as err:
        logger.error(f"解析网页出错: {err}")
        logger.error(f"parse err url {d_url}")
        return None, None 

async def search_engine_chat(query: str,
                             top_k: int = 3,
                             session = requests.Session(),
                             backup_nums = 5,
                             max_length = 500,
                             **kwargs) -> str:
    url = f"http://www.baidu.com/s?wd={query}&cl=3&pn=1&ie=utf-8&rn={top_k + backup_nums}&tn=baidurt"
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
    content = content.replace("<img(.*?)>", "")
    content = content.replace("</p><p>", "\n").replace("<br>", "\n").replace("<br/>", "\n")
    content = content.replace("<p>", "").replace("</p>", "").replace("<span >", "").replace("</span>", "")
    content = content[:max_length]
    return content
    
if __name__ == '__main__':
    query = '人为什么会陷入虚无主义'
    # query = '早起头疼是什么原因'
    text = asyncio.run(search_engine_chat(query))
    print(text)
