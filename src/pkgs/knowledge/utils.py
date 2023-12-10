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
    
if __name__ == '__main__':
    query = '高血压可能是什么原因造成的'
    # query = '早起头疼是什么原因'
    text = asyncio.run(search_engine_chat(query))
    print(text)
