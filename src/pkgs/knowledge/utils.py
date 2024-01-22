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
from tabnanny import verbose
from uu import Error
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

headers = {
	"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
	"Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
	"Connection": "keep-alive",
	"Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "text/html; charset=utf-8",
	"Host": "www.baidu.com",
	# 需要更换Cookie
	# "Cookie": """BIDUPSID=439EE448EACEE1F6F7990558471F74E8; PSTM=1691026709; H_WISE_SIDS=234020_131861_213350_214789_110085_244714_236312_262914_256419_265881_266361_265615_267072_268592_268031_259642_269236_256154_269731_269832_269328_269904_267066_256739_270460_270548_271172_263618_256957_267659_271322_271268_257179_271477_266028_270102_271812_271935_271254_234296_234208_269297_272279_272465_253022_271688_272840_260335_272989_272556_273062_267559_273164_273139_273235_273300_273374_273399_273380_271158_273451_270055_273519_272225_271562_271146_273671_273705_273318_264170_270186_273902_274080_273931_273966_274140_269609_274209_273917_273786_273043_273594_274300_256223_272806_274279_272319_272685_274441_272331_274256_197096_274765_274760_270142_274844_274854_274856_274847_270158_274871_275070_272801_274450_275097_272324_267806_267547; H_WISE_SIDS_BFESS=234020_131861_213350_214789_110085_244714_236312_262914_256419_265881_266361_265615_267072_268592_268031_259642_269236_256154_269731_269832_269328_269904_267066_256739_270460_270548_271172_263618_256957_267659_271322_271268_257179_271477_266028_270102_271812_271935_271254_234296_234208_269297_272279_272465_253022_271688_272840_260335_272989_272556_273062_267559_273164_273139_273235_273300_273374_273399_273380_271158_273451_270055_273519_272225_271562_271146_273671_273705_273318_264170_270186_273902_274080_273931_273966_274140_269609_274209_273917_273786_273043_273594_274300_256223_272806_274279_272319_272685_274441_272331_274256_197096_274765_274760_270142_274844_274854_274856_274847_270158_274871_275070_272801_274450_275097_272324_267806_267547; Hm_lvt_aec699bb6442ba076c8981c6dc490771=1694682757; Hm_lpvt_aec699bb6442ba076c8981c6dc490771=1694682757; COOKIE_SESSION=109027_0_1_0_2_1_1_0_1_1_0_0_69_0_73_0_1694682760_0_1694682687%7C2%230_0_1694682687%7C1; delPer=0; BD_CK_SAM=1; ZFY=xscdHGcRiKqtiy:Bzq0CMjcEqySzF3nh9R:Ao025lMpTk:C; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; PSINO=1; __bid_n=189a9c64a3f74561fe3291; BAIDUID=24F27377269B91FC7119DDADA7C1DC24:FG=1; BAIDUID_BFESS=24F27377269B91FC7119DDADA7C1DC24:FG=1; BDRCVFR[BIMQ49Drrdf]=mk3SLVN4HKm; BD_UPN=12314753; BDUSS=FhLXQzLWF-NVVDUmhsYS1iNTZFbkNuczlzSVJVVUtEQTlCaGdPZkIxWHF-cFJsRVFBQUFBJCQAAAAAAAAAAAEAAACy1YNOdGljb0FnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOpxbWXqcW1lVW; BDUSS_BFESS=FhLXQzLWF-NVVDUmhsYS1iNTZFbkNuczlzSVJVVUtEQTlCaGdPZkIxWHF-cFJsRVFBQUFBJCQAAAAAAAAAAAEAAACy1YNOdGljb0FnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOpxbWXqcW1lVW; ZD_ENTRY=bing; RT="z=1&dm=baidu.com&si=c1e335f4-496b-4aea-9fff-7ecbb197ce4d&ss=lpvzk3ax&sl=z&tt=i2s&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=9yot&ul=a1lv&hd=a1lz"; H_PS_PSSID=39713_39730_39779_39703_39686_39679_39783_39842_39904_39819_39909_39935_39937_39933_39945_39940_39939_39931; BA_HECTOR=ak8521a1a0ah818l200ga58n1inarmn1r; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598"""
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
                             max_length = 800,
                             **kwargs) -> str:
    url = f"http://www.baidu.com/s?wd={query}&cl=3&pn=1&ie=utf-8&rn={top_k + backup_nums}&tn=baidurt"
    response = session.get(url, headers=headers, )
    res = etree.HTML(response.text)
    detail_urls = res.xpath('//h3[@class="t"]/a/@href')
    
    if not detail_urls:
        logger.error(f"Err to parse url: {url}\n")
        return "对不起, searchEngine未查询到相关内容"
    task_list = []
    for d_url in detail_urls:
        task_list.append(asyncio.create_task(parse_detail_page(d_url)))
    done, pending = await asyncio.wait(task_list, timeout=None)
    
    content = ""
    i = 0
    for done_task in done:
        title, text = done_task.result()
        if title and i < 3:
            content += f"{title}\n{text}\n\n"
            if len(content) > max_length:
                break
            i += 1
    content = content.replace("<img(.*?)>", "")
    content = content.replace("</p><p>", "\n").replace("<br>", "\n").replace("<br/>", "\n")
    content = content.replace("<p>", "").replace("</p>", "").replace("<span >", "").replace("</span>", "")
    if len(content) > max_length + 200:
        content = content[:max_length+200]
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
