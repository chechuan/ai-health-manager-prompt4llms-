# -*- encoding: utf-8 -*-
"""
@Time    :   2024-11-21 17:49:45
@Author  :   李力
@desc    :   多模态大模型相关功能
"""

import asyncio
import aiohttp
import os
import re
import sys
import json

# 将项目根目录加入sys.path，方便单元测试
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from src.utils.resources import InitAllResource
from src.utils.Logger import logger
from src.prompt.model_init import acallLLM, callLLM


class MultiModalModel:
    def __init__(self, gsr):
        self.gsr = gsr
        # self.endpoint = self.gsr.multimodal_config["endpoint"]
        self._init_prompts()

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
        logger.debug(f"image_type_recog image_url: {image_url} diet_recog: {diet_recog}")
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")
        if not diet_recog in [True, False]:
            return self._get_result(400, {}, "diet_recog must be boolean.")

        # 检查图片 URL 是否可访问
        async with aiohttp.ClientSession() as session:
            async with session.head(image_url) as response:
                if not (response.status == 200):
                    return self._get_result(400, {}, "image_url is not accessible.")

        # 先调用多模态大模型识别
        messages = [{
            "role": "system",
            "content": self.prompts["图片内容"]
        }, {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_url}}]
        }]
        image_caption = await acallLLM(
            history=messages, max_tokens=768, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=20
        )

        # 图片分类
        messages = [{
            "role": "user",
            "content": f"{self.prompts['图片分类']} {image_caption}",
        }]
        classification_text = await acallLLM(
            history=messages, max_tokens=64, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=20
        )

        # 类别定义，大类分四种，子类别index对应types的编号
        types = {"其他": 0, "饮食": 1, "运动": 2, "报告": 3}
        subtypes = [
            ["其他"],
            ["其他", "餐食", "食材"],
            ["其他", "运动图片", "运动后的报告"],
            ["其他", "体检报告", "检查报告", "检验报告", "体重报告", "血压报告", "血糖报告"],
        ]

        # 分类结果处理
        json_data = {"status": 0, "type": 0, "desc": "其他", "subtype": 0, "subdesc": "其他"}
        if classification_text in types:
            json_data["type"] = types[classification_text]
            json_data["desc"] = classification_text
            if classification_text == "饮食":
                json_data["subtype"] = 1
                json_data["subdesc"] = "餐食"
            elif classification_text == "运动":
                json_data["subtype"] = 2
                json_data["subdesc"] = "运动后的报告"
            elif classification_text == "报告":
                json_data["subtype"] = 4
                json_data["subdesc"] = "体重报告"

            # 后端建议，判断是否需要额外做饮食识别
            if diet_recog is True and json_data["desc"] == "饮食":
                # 原本想直接用图片描述做菜品信息格式化来提高效率
                # 但后来更换QWen模型，prompt略有不同，还是采用两步策略
                messages = [{
                    "role": "system",
                    "content": self.prompts["菜品描述"]
                }, {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_url}}]
                }]
                diet_text = await acallLLM(
                    history=messages, max_tokens=768, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1",
                    timeout=20
                )

                messages = [{
                    "role": "user",
                    "content": f"{self.prompts['菜品格式化']} {diet_text}",
                }]
                generate_text = await acallLLM(
                    history=messages, max_tokens=1024, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=20
                )
                # 处理结果
                diet_info = None
                try:  # 去掉json串中多余的内容
                    diet_info = self._get_food_info_json(generate_text)
                except Exception as error:
                    logger.error("image_type_recog error check json {0}".format(image_url))
                if diet_info:
                    json_data["foods"] = diet_info["foods"]
                else:
                    json_data["foods"] = []
                    json_data["status"] = -1

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"image_type_recog success: {result}")
        return result

    async def diet_recog(self, **kwargs):
        """菜品识别，提取菜品名称、数量、单位信息"""
        image_url = kwargs.get("image_url")
        logger.debug(f"diet_recog image_url: {image_url}")
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")

        # 检查图片 URL 是否可访问
        async with aiohttp.ClientSession() as session:
            async with session.head(image_url) as response:
                if not (response.status == 200):
                    return self._get_result(400, {}, "image_url is not accessible.")

        # 先调用多模态大模型识别菜品名称以及数量
        messages = [{
            "role": "system",
            "content": self.prompts["菜品描述"]
        }, {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image_url}}]
        }]
        diet_text = await acallLLM(
            history=messages, max_tokens=768, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=20
        )

        # 格式化菜品信息
        messages = [{
            "role": "user",
            "content": f"{self.prompts['菜品格式化']} {diet_text}",
        }]
        generate_text = await acallLLM(
            history=messages, max_tokens=1024, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=20
        )

        # 处理结果
        json_result = None
        try:  # 去掉json串中多余的内容
            json_result = self._get_food_info_json(generate_text)
        except Exception as error:
            logger.error("diet_recog error check json {0}".format(image_url))
        if json_result:
            json_data = {"status": 0, "foods": json_result["foods"]}
        else:
            json_data = {"status": -1, "foods": []}

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"diet_recog success: {result}")
        return result

    async def diet_eval(self, **kwargs):
        """饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        user_info = kwargs.get("user_info", {}) or {}
        diet_info = kwargs.get("diet_info", []) or []
        management_tag = kwargs.get("management_tag", "") or ""
        diet_period = kwargs.get("diet_period", "") or ""
        logger.debug(
            f"diet_eval user_info: {user_info} diet_info: {diet_info} management_tag: {management_tag} diet_period: {diet_period}")

        # 初始化返回内容
        json_data = {
            "status": 0,
            "content": "无论您选择何种餐食，都请记得关注食物的多样性和营养均衡，细嚼慢咽享受美味的一餐吧~"
        }

        # 如果没有饮食信息也就不用出其他内容了
        if len(diet_info) <= 0:
            result = self._get_result(200, json_data, "")
            logger.debug(f"diet_eval success: {result}")
            return result

        # 拼接用户信息
        query = ""
        if len(user_info.keys()) > 0:
            query += f"用户基础信息：{json.dumps(user_info, ensure_ascii=False)}\n"
        if diet_period in ["早餐", "午餐", "晚餐"]:
            query += f"本餐餐段：{diet_period}\n"
        query += f"本餐餐食信息：{json.dumps(diet_info, ensure_ascii=False)}\n"
        if management_tag in ["血糖管理", "血压管理", "减脂减重管理"]:
            query += f"用户目标：{management_tag}\n"

        # 请求LLM
        messages = [{
            'role': 'user',
            'content': f"{self.prompts['饮食一句话建议']} {query}",
        }]
        generate_text = await acallLLM(
            history=messages, max_tokens=512, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=20
        )
        json_data["content"] = generate_text

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"diet_eval success: {result}")
        return result

    def _init_prompts(self):
        self.prompts = {
            "菜品描述": "请分析图片中所有菜品以及数量",
            "菜品格式化": """请根据图片描述，将菜品信息进行格式化。
以下是返回示例：
如果图片中没有食物，则返回
``` json
{"foods":[]}
```
如果图片中包含一份黄焖鸡和一碗米饭，则返回
``` json
{"foods":[{"foodname": "黄焖鸡", "unit": "份", "count": "1"},
{"foodname": "米饭", "unit": "碗", "count": "1"}]}
```
注意不要拆分成食材返回
``` json
{"foods":[{"foodname": "鸡肉", "unit": "块", "count": "4"},
{"foodname": "青椒", "unit": "根", "count": "2"},
{"foodname": "香菇", "unit": "朵", "count": "2"},
{"foodname": "大米", "unit": "份", "count": "1"}]}
```
请严格遵循以下要求：
过滤掉无法食用的内容。
严格按照json的结构返回结果，必须包含foodname、unit、count字段。
其中foodname字段尽量返回菜名，用中文回答。
以下是图片描述：
""",
            "图片内容": """简述图片中的场景和元素""",
            "图片分类": """请根据图片描述，判断内容属于"饮食"、"运动"、"报告"、"其他"中的哪一类。
如果图片中主要是食物，则返回"饮食"。
如果图片中是运动器材、体能、健身的内容，则返回"运动"。
如果图片中是包含体重、体脂的截图内容，则返回"报告"。
如果图片中是药物、血压计血糖仪等医疗器械，或其他内容，则返回"其他"。
不要返回其他内容，仅返回类别名称。
以下是图片描述：
""",
            # 这个识别是给多模态大模型直接用的，但效果一般，后来还是改成多模态+llm配合的方式了
            "菜品识别": """请你扮演一位餐厅服务员，分析顾客图片中的菜品名称以及数量。
以下是返回示例：
如果图片中没有食物，则返回
``` json
{"foods":[]}
```
如果图片中包含一份黄焖鸡和一碗米饭，则返回
``` json
{"foods":[{"foodname": "黄焖鸡", "unit": "份", "count": "1"},
{"foodname": "米饭", "unit": "碗", "count": "1"}]}
```
请严格遵循以下要求：
尤其关注一下有没有主食，并过滤掉无法食用的内容。
严格按照json的结构返回结果。
""",
            "饮食一句话建议": """请你扮演一位营养师，告知用户提交的信息内的食物是否可以吃完？需要添加什么食物，或者不需要吃图片中的什么食物。根据211法则判断本餐搭配是否合理，即蔬菜：蛋白质：主食=2：1：1。生成一句话建议，只返回建议不要其他内容，不超过50个字。
以下是专业营养师饮食建议样例：
如果用户提交食物为南瓜1个、黄瓜1根、虾4只，示例建议：本餐的蔬菜选择的是黄瓜，这里整体的分量不多，饱腹感不够的话可以再加1根黄瓜，蛋白稍微不够，虾再多加3-4只。
如果用户提交食物为糙米饭1份、黄瓜1根、牛腩1份，示例建议：主食糙米饭团很不错，黄瓜可以放心吃完。牛腩只吃瘦肉部分，建议吃到一掌心的量。
如果用户提交食物为清炒油菜1份、玉米1根、冬瓜排骨肉1份，示例建议：肉我们只吃瘦肉部分，分量上再加两块没问题，蔬菜和玉米分量可以。
如果用户提交食物为清炒菜心1份、蒸鱼3块、白米饭1份，示例建议：鱼肉我们可以再加一小块，蔬菜全部吃完没问题，白米饭最后可以剩下来一两口，下次掺杂一点小米一起煮会更好哟。
如果用户提交食物为豆腐脑1碗、鸡蛋2个，示例建议：蛋白质的量是可以的哈，豆腐脑卤有勾芡，把卤剩下一些哈。另外缺少蔬菜，早餐可以增加一份蔬菜，选择黄瓜或西红柿是可以的。
以下是当前用户提交的信息：
""",
        }

    def _get_food_info_json(self, result):
        # 定向纠错，去掉json串前面的分析内容
        if result[:-3].rfind("```") > 0:
            result = result[result[:-3].rfind("```"):]
        if result[:-3].rfind("json") > 0:
            result = result[result[:-3].rfind("json"):]
        if result.rfind("返回") > 0:
            result = result[result.rfind("返回"):]
        if result.rfind("//") > 0:  # 去掉多余的注释内容
            result = re.sub(r"//.*?\n", "", result)
        # 去掉json串中多余的markdown内容
        info = json.loads(result.replace("```", "").replace("json", "").replace("返回", "").replace("。", ""))

        if "foods" not in info or not isinstance(info["foods"], list):
            raise Exception
        for i, food in enumerate(info["foods"]):
            if "foodname" not in food or "unit" not in food or "count" not in food:
                raise Exception
            try:  # 检查数量是否为数字，不是则默认为1
                count = int(food["count"])
            except Exception as _:
                food["count"] = "1"

        return info


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
    input_data = {
        "user_info": None,
        "management_tag": "",
        "diet_period": "早餐",
        "diet_info": [{"foodname": "鸡蛋", "count": "1", "unit": "个"}]
    }

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 图片分类
        for i, data in enumerate(image_data):
            result = loop.run_until_complete(multimodal_model.image_type_recog(**data))
            print("#########", result)
        # 菜品识别
        result = loop.run_until_complete(multimodal_model.diet_recog(**image_data[2]))
        print("#########", result)
        # 饮食一句话点评
        result = loop.run_until_complete(multimodal_model.diet_eval(**input_data))
        print("#########", result)
    except Exception as e:
        print("发生错误：", e)
    finally:
        loop.close()
