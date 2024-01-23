# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-04 15:10:41
@desc    :   https://blog.csdn.net/qq_29654777/article/details/104051567 搜索api爬虫
@Author  :   宋昊阳, 员庆阳
@Contact :   1627635056@qq.com
'''
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from unittest import result

from pydantic import ValidationError
from torch import topk

sys.path.append(str(Path.cwd().absolute()))
from typing import Any, AnyStr, Dict, List, Optional, Union

import requests
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.llms import openai
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from lxml import etree

# from langchain.chains.base import Chain
from src.pkgs._langchain.chains.base import Chain
from src.pkgs.knowledge.config.prompt_config import (PROMPT_TEMPLATES, SEARCH_QA_HISTORY_PROMPT,
                                                     SEARCH_QA_PROMPT)
from src.utils.Logger import logger

# headers = {
# 	"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
# 	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
# 	"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
# 	"Connection": "keep-alive",
# 	"Accept-Encoding": "gzip, deflate, br",
#     "Content-Type": "text/html; charset=utf-8",
# 	# "Host": "www.baidu.com",
# 	# 需要更换Cookie
# 	"Cookie": """BAIDUID=24F27377269B91FC7119DDADA7C1DC24:FG=1; BAIDUID_BFESS=24F27377269B91FC7119DDADA7C1DC24:FG=1; channel=bing; ab_sr=1.0.1_NGI0MTBkMjc0Yzg4Yjc5MTAwZDY1NTY0Y2QwMmEyZjM3NTczYzY1NjJjMWU3N2FlMTZhYTc1YWU3NDJhZDM5OTA0ZThmOTNhMjRjMzNiN2QzYzU1ODQ2NDY4OGI5OGUzZDViM2Q0ZTYyZDI2MTlmNTgwMDMxM2Q1NTE5YmU1MDZkZDMxYzRmYzVkZmM5MDYwY2M4MDIzYTQwOWI5ZGI4Mw=="""
# }

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Cookie": "sm_diu=d8b432a1faac5bb75333878e8aab9966%7C%7C11eef1ee74be5e53b9%7C1705987906; XSRF-TOKEN=42cb2151-94a0-4960-bb81-a24defa0b371; sm_uuid=49ae3193ba8848e89163950d2a16e2c1|||1705989325; cna=8HgxHsrNU2ACAS/sHe96a39i; __itrace_wid=35857b25-eba3-4be0-874b-ab0a7d40bf2f; lsmap2=2f03M94U07S1AA0CE2Dq0Go1Nn0Ps0Tx0Y20Yw0hi0hj0qy2qz1su3sv2sw4sx3ti1tj1; sm_ruid=b838f22edd52d937a954a350d2d11ddf%7C%7C%7C1705989607; phid=44e82806fad9454ead850c51fd534f15; isg=BPDwLxbtYVzz4D1YzuUfWxtowb5COdSDKOGkkepFI8sOpZJPlEhoEuSH--1gLoxb; sm_sid=5f2293b1e93d4da4bb0341b6ad8d3441",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding":"gzip, deflate, br",
    "Accept-Language":"zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Cache-Control": "max-age=0"
}

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
    html = etree.HTML(_html)
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
        title = html.xpath("//p[contains(@class, 'hc-line-clamp2')]/text()")[0]
        p_str_list = html.xpath("//div[contains(@class, 'index_textContent ')]//p//text()")
        text = "\n".join(p_str_list)
    elif 'm.baidu.com' in url:
        _data = html.xpath("//script[@id='__NEXT_DATA__']//text()")[0]
        data = json.loads(_data)
        title = data['props']['pageProps']['commonData']['title']
        text = data['props']['pageProps']['contentData']['content'][0]
    elif 'youlai.cn' in url:
        if "video/article/" in url:
            # title = html.xpath('//h1[@class="title"]/text()')[0]
            title = html.xpath('//h3[@class="v_title"]/text()')[0]
            text = html.xpath('//div[@class="p_text_box hidden_auto"]//p//text()')
        else:
            title = html.xpath('//*[@class="v_title"]/text()')[0]
            text = html.xpath('//div[@class="text"]//text()')
    elif 'baike.baidu.com' in url:
        title = html.xpath('//*[@class="lemmaTitle_pjifB J-lemma-title"]/text()')[0]
        text = html.xpath('//*[@class="lemmaSummary_VQxNY J-summary"]//text()')
    elif 'lemon.baidu.com' in url:
        if '/ec/article/' in url:
            title = html.xpath('//*[@class="ArticleDetail_detailTitle__2ti6K undefined ArticleDetail_biggerTitle__3RZK1"]/text()')[0]
            text = html.xpath('//*[@class="article-detail"]//text()')
        elif '/ec/question' in url:
            title = html.xpath('//*[@class="Title_detailTitle__3GfU7"]/text()')[0]
            text = html.xpath('//*[@class="QuestionDetail_questionDetailWrapper__3QH5b"]//text()')
    elif 'songbai.cn' in url:
        if 'pcs/qa/answers' in url:
            title = html.xpath('//section[@class="qa-summary"]/h1[@class="title"]//text()')[0]
            text = html.xpath('//div[@class="answer-item"]//text()')
    elif 'www.kedaifu.com' in url:
        if '/ask/audio' in url:
            title = html.xpath('//p[@class="audioTitle"]/text()')[0]
            text = html.xpath('//div[@class="audioData"]//div[@class="frame"]//p//text()')
    else:
        title, text = None, None
    if isinstance(text, list):
        text = "".join([i.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "") for i in text])
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
        return None, None, None
    
    try:
        title, text = parse_diff_pages(d_url, _html)
        if title and text:
            logger.debug(f"url: {d_url}\ntitle: {title}\ntext: {text}")
        else:
            logger.debug(f"url: fail to parse {d_url}")
        return title, text, d_url
    except Exception as err:
        logger.error(f"解析网页出错: {err}")
        logger.error(f"parse err url {d_url}")
        return None, None, None

async def search_engine_chat(query: str,
                             top_k: int = 3,
                             session = requests.Session(),
                             backup_nums = 5,
                             max_length = 800,
                             **kwargs) -> str:
    # url = f"http://www.baidu.com/s?wd={query}&cl=3&pn=1&ie=utf-8&rn={top_k + backup_nums}&tn=baidurt"
    # url = f"https://duckduckgo.com/?t=h_&q={query}"
    # url = f"https://www.startpage.com/do/search?cmd=process_search&query={query}"
    url = f"https://www.sogou.com/web?query={query}"
    # url = f"https://quark.sm.cn/s?q={query}"
    response = None

    if url.startswith("https://www.startpage.com/do/search?"):
        response = session.get(url)
        res = etree.HTML(response.text)
        detail_urls = res.xpath('//div[@class="w-gl__result-title result-link"]/@href')
    elif url.startswith("https://www.baidu.com/s?"):
        response = session.get(url, headers=headers)
        res = etree.HTML(response.text)
        detail_urls = res.xpath('//h3[@class="t"]/a/@href')
    elif url.startswith("https://www.sogou.com/web?"):
        def fetch_url_list(url):
            response = session.get(url, headers=headers)
            res = etree.HTML(response.text)
            detail_urls = res.xpath('//h3[@class="vr-title  "]/a/@href')
            return detail_urls
        detail_urls = [i for j in [fetch_url_list(url + f"&page={page}&ie=utf8") for page in range(1,3)] for i in j]
        detail_urls = [(i if i.startswith("http") else "https://www.sogou.com" + i) for i in detail_urls]

    if isinstance(response, requests.Response) and response.status_code != 200:
        logger.error(f"Err to get url: {url}, status_code: {response.status_code}")
        return "对不起, 网络连接异常, 未检索到相关内容"
    if not detail_urls:
        logger.error(f"Err to parse url: {url}\n")
        return "对不起, searchEngine未查询到相关内容"
    task_list = []
    for d_url in detail_urls:
        task_list.append(asyncio.create_task(parse_detail_page(d_url)))
    done, pending = await asyncio.wait(task_list, timeout=None)
    
    content = ""
    i = 0
    result_list = []
    for done_task in done:
        title, text, d_url = done_task.result()
        if not title or not text:
            continue
        result_list.append({"title": title, "body": text, "url": d_url})
        if title and i < 3 and len(content) < max_length:
            content += f"{title}\n{text}\n\n"
            i += 1
    content = content.replace("<img(.*?)>", "")
    content = content.replace("</p><p>", "\n").replace("<br>", "\n").replace("<br/>", "\n")
    content = content.replace("<p>", "").replace("</p>", "").replace("<span >", "").replace("</span>", "")
    if len(content) > max_length + 200:
        content = content[:max_length+200]
    if kwargs.get("return_list", False):
        return result_list
    else:
        return content

class DDGSearchChain():
    proxies: Union[dict, str] = "http://127.0.0.1:7890"
    input_key: str = "query"
    output_key: str = "answer"
    
    def __init__(self, proxies) -> None:
        """duckduckgo search 封装
        
        Args:
            proxies (Union[dict, str], optional): Proxies for the HTTP client (can be dict or str). Defaults to None.
        """
        self.proxies = proxies
        self.engine = DDGS(proxies=proxies)
    
    def call(
        self, 
        keywords: str = "",
        region: str = "us-en",      # 语言区域, 使用zh-cn 报错DuckDuckGoSearchException: Ratelimit
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        DuckDuckGo text search generator. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            backend: api, html, lite. Defaults to api.
                api - collect data from https://duckduckgo.com,
                html - collect data from https://html.duckduckgo.com,
                lite - collect data from https://lite.duckduckgo.com.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List[Dict[str, Any]]: list of search results.
        """
        if not keywords and not kwargs.get('keywords'):
            raise ValidationError("keywords is required")
        result = []
        try:
            for i in self.engine.text(keywords, 
                                      region=region, 
                                      safesearch=safesearch, 
                                      timelimit=timelimit, 
                                      max_results=max_results
                                      ):
                result.append(i)
        except Exception as e:
            logger.exception(f"DDGSearch error: {e}")
            self.engine = DDGS(proxies=self.proxies)
            for i in self.engine.text(keywords, region=region, safesearch=safesearch, timelimit=timelimit, max_results=max_results):
                result.append(i)
        return result

class SearchQAChain(Chain):
    """Chain for search engine and QA system.
    """
    qa_chain: LLMChain
    ddg_search_chain: DDGSearchChain
    input_key: str = "query"
    output_key: str = "answer"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @classmethod
    def from_llm(
        cls, 
        llm: BaseLanguageModel, 
        qa_prompt: BasePromptTemplate = SEARCH_QA_PROMPT,
        proxies: Union[dict, str] = "socks://127.0.0.1:7891",
        **kwargs: Any
    ) -> "SearchQAChain":
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt, return_final_only=False, verbose=kwargs.get("verbose", False))
        logger.info(f"Initialize DDGSearchChain with proxies: {proxies}")
        ddg_search_chain = DDGSearchChain(proxies=proxies)
        return cls(
            qa_chain=qa_chain,
            ddg_search_chain=ddg_search_chain,
            **kwargs
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        max_results: int = 3,
        **kwargs: Any
    ) -> Dict[str, str]:
        """search query with web and answer question.
        
        Args:
            inputs (Dict[str, Any]): input data.
            max_results (int, optional): top k search results. Defaults to 3.

        Returns:
            Dict[str, str]: output data.
        """
        return_vars = {}
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        return_vars["query"] = question
        # search engine
        search_results = self.ddg_search_chain.call(question, max_results=max_results)
        return_vars['search_results'] = search_results
        if not search_results:
            _run_manager.on_text(
                f"no search results: {question}", 
                color="red", 
                end="\n", 
                verbose=self.verbose
            )
            return_vars[self.output_key] = "对不起, 未查询到相关内容"
            return return_vars
        _run_manager.on_text(f"search results num: {len(search_results)}", end="\n", color="green", verbose=self.verbose)
        # qa system
        content = "\n\n".join([f"{i['title']}\n{i['body']}" for i in search_results])
        qa_inputs = {"question": question,"context": content}

        qa_outputs = self.qa_chain(qa_inputs, callbacks=_run_manager.get_child())
        return_vars['qa_content'] = content
        # qa_result = re.findall("<回答>\n(.*?)\n</回答>", "<回答>\n" + qa_outputs['text'])
        return_vars[self.output_key] = qa_outputs['text']
        _run_manager.on_text(f"search qa answer: {qa_outputs['text']}", end="\n", color="green", verbose=self.verbose)
        return return_vars

class LangChainDDGSResults:
    def __init__(self) -> None:
        from langchain.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=2, )
        self.search = DuckDuckGoSearchResults(wrapper=self.wrapper)

    def call(self, query: str):
        results = self.search.search(query)
        return results

if __name__ == '__main__':
    # query = '糖尿病可以吃哪些食物？'
    query = '今天北京天气怎么样?'
    # text = asyncio.run(search_engine_chat(query))
    # print(text)
    llm = openai.OpenAI(
        model_name = "Baichuan2-7B-Chat",
        openai_api_base=os.getenv("OPENAI_API_BASE") + "/v1", 
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    qa_chain = SearchQAChain.from_llm(llm=llm, return_final_only=False, verbose=True)
    qa_outputs = qa_chain.run(query=query, max_results=3)
