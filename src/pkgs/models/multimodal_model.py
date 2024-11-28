# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-21 17:49:45
@Author  :   李力
@desc    :   多模态大模型相关功能
鉴于图像需要部分预处理，避免图片质量问题导致大模型处理错误
且多模态大模型需本地处理，故参考之前OCR项目的实现
本端处理参数和返回结果，服务端处理图片相关逻辑
"""

# 将项目根目录加入sys.path，方便单元测试
# import sys
# import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import asyncio
import aiohttp
import os
import json
from src.utils.resources import InitAllResource
from src.utils.Logger import logger

class MultiModalModel:
    def __init__(self, gsr):
        self.gsr = gsr
        self.endpoint = self.gsr.multimodal_config["endpoint"]
    
    def _get_result(self, head: int, items: dict, msg: str = "") -> dict:
        """封装返回结果"""
        return {
            "head": head,
            "items": items,
            "msg": msg
        }
    
    async def image_type_recog(self, **kwargs):
        """图片分类，包含：饮食、运动、报告、其他"""
        image_url = kwargs.get("image_url")
        diet_recog = kwargs.get("diet_recog", False)
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")
        if not diet_recog in [True, False]:
            return self._get_result(400, {}, "diet_recog must be boolean.")
        payload = {"image_url": image_url, "diet_recog": diet_recog}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/dishes/ana_image_type", data=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"image_type_recog success: {data}")
                    return self._get_result(resp.status, data, "")
                else:
                    error_content = await resp.text()
                    error_msg = f"image_type_recog failed with status {resp.status}: {error_content}"
                    logger.error(error_msg)
                    return self._get_result(resp.status, {}, error_msg)
    
    async def diet_recog(self, **kwargs):
        """菜品识别，提取菜品名称、数量、单位信息"""
        image_url = kwargs.get("image_url")
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")
        payload = {"image_url": image_url}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/dishes/ana_food_info", data=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"diet_recog success: {data}")
                    return self._get_result(resp.status, data, "")
                else:
                    error_content = await resp.text()
                    error_msg = f"diet_recog failed with status {resp.status}: {error_content}"
                    logger.error(error_msg)
                    return self._get_result(resp.status, {}, error_msg)
    
    async def diet_eval(self, **kwargs):
        """饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        user_info = kwargs.get("user_info", {})
        diet_info = kwargs.get("diet_info", [])
        management_tag = kwargs.get("management_tag", "")
        diet_period = kwargs.get("diet_period", "")
        payload = {"user_info": user_info, "diet_info": diet_info, "management_tag": management_tag, "diet_period": diet_period}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/dishes/ana_food_eval", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"diet_eval success: {data}")
                    return self._get_result(resp.status, data, "")
                else:
                    error_content = await resp.text()
                    error_msg = f"diet_eval failed with status {resp.status}: {error_content}"
                    logger.error(error_msg)
                    return self._get_result(resp.status, {}, error_msg)
                
if __name__ == '__main__':

    # 初始化资源
    gsr = InitAllResource()
    multimodal_model = MultiModalModel(gsr)

    # 测试图片
    image_data = [
        {},
        {"image_url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/temp/dishes/0.jpg"},
        {"image_url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/temp/dishes/1.jpg"},
        {"image_url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/temp/dishes/2.png"},
        {"image_url": "https://lk-shuzhizhongtai-common.oss-cn-beijing.aliyuncs.com/temp/dishes/3.jpg"},
    ]

    # 图片分类
    for data in image_data:
        result = asyncio.run(multimodal_model.image_type_recog(**data))
    # 菜品识别
    result = asyncio.run(multimodal_model.diet_recog(**image_data[2]))
    # 饮食一句话点评
    result = asyncio.run(multimodal_model.diet_eval(**{}))
