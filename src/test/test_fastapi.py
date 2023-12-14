# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-14 11:32:13
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import asyncio
import sys
from pathlib import Path

import uvicorn as u
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.utils.Logger import logger
from src.utils.module import clock, curr_time, dumpJS

app = FastAPI(cross_origin=True)

async def async_sleep():
    await asyncio.sleep(1)
    return "async"

 #   声明参数类
class Request_data(BaseModel):
    province_index: str  #province-index形式
    addrs: str      #上面对应的中文地址

@app.post('/test/async')
async def _test_async(request_data: Request_data):
    """获取意图代码
    """
    t1 = curr_time()
    await async_sleep()
    ret = {"start":t1, "end": curr_time()}
    logger.debug(ret)
    return ret


 #   声明参数类
class Item(BaseModel):
    province_index: str  #province-index形式
    addrs: str      #上面对应的中文地址

@app.post("/country_psm_pred")
async def psm_predict(request_data: Item):
    province_index = request_data.province_index
    addrs =  request_data.addrs
    res = {'province_index':province_index,'addrs':addrs}
    return res
# POST 请求
class Item(BaseModel):
    province_index: str  #province-index形式
    addrs: str      #上面对应的中文地址
@app.post("/items")
async def create_item(item: Item):
    return {"message": "Item created successfully: {}".format(str(item))}

if __name__ == '__main__':
    u.run(app,host="127.0.0.1",port=6500)  