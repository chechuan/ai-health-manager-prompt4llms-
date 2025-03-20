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
import time

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
        try:
            image_caption = await acallLLM(
                history=messages, max_tokens=768, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=45
            )
        except Exception as error:
            logger.error("image_type_recog error recog image {0}".format(image_url))
            image_caption = "图片识别失败"

        # 图片分类
        messages = [{
            "role": "user",
            "content": f"{self.prompts['图片分类']} {image_caption}",
        }]
        classification_text = await acallLLM(
            history=messages, max_tokens=64, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
        )

        # 类别定义，大类分四种，子类别index对应types的编号
        types = {"其他": 0, "饮食": 1, "运动": 2, "报告": 3}
        subtypes = [
            ["其他"],
            ["其他", "餐食", "食材"],
            ["其他", "运动图片", "运动后的报告"],
            ["其他报告", "体检报告", "检查报告", "检验报告", "体重报告", "血压报告", "血糖报告", "饮食报告"],
        ]

        # 分类结果处理
        json_data = {"status": 0, "type": 0, "desc": "其他", "subtype": 0, "subdesc": "其他"}
        if classification_text in types:
            json_data["type"] = types[classification_text]
            json_data["desc"] = classification_text
            if classification_text == "饮食":
                json_data["subtype"] = 1
                json_data["subdesc"] = "餐食"

                # 后端给的改进建议建议，判断是否需要额外做饮食识别
                if diet_recog is True:
                    # 直接调用多模态大模型识别菜品名称以及数量
                    messages = [{
                        "role": "user",
                        "content":[
                            {"type": "text", "text": self.prompts["菜品直接识别"]},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }]
                    try:
                        generate_text = await acallLLM(
                            history=messages, max_tokens=1024, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=45
                        )
                    except Exception as error:
                        logger.error("image_type_recog error recog diet image {0}".format(image_url))
                        generate_text = "" # 报错则返回空，表示未识别，进入后续异常处理

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

            elif classification_text == "运动":
                # 调用大模型进一步区分报告类别
                messages = [{
                    "role": "user",
                    "content": f"{self.prompts['运动分类']} {image_caption}",
                }]
                sub_type_text = await acallLLM(
                    history=messages, max_tokens=64, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
                )

                # 判断返回值匹配类别
                if sub_type_text in subtypes[types["运动"]]:
                    json_data["subtype"] = subtypes[types["运动"]].index(sub_type_text)
                    json_data["subdesc"] = sub_type_text
                else:
                    json_data["subtype"] = 0
                    json_data["subdesc"] = "其他"
            elif classification_text == "报告":
                # 调用大模型进一步区分报告类别
                messages = [{
                    "role": "user",
                    "content": f"{self.prompts['报告分类']} {image_caption}",
                }]
                sub_type_text = await acallLLM(
                    history=messages, max_tokens=64, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
                )

                # 判断返回值匹配类别
                if sub_type_text in subtypes[types["报告"]]:
                    json_data["subtype"] = subtypes[types["报告"]].index(sub_type_text)
                    json_data["subdesc"] = sub_type_text
                else:
                    json_data["subtype"] = 0
                    json_data["subdesc"] = "其他报告"

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

        # 直接调用多模态大模型识别菜品名称以及数量
        messages = [{
            "role": "user",
            "content":[
                {"type": "text", "text": self.prompts["菜品直接识别"]},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        try:
            generate_text = await acallLLM(
                history=messages, max_tokens=1024, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=45
            )
        except Exception as error:
            logger.error("image_type_recog error recog diet image {0}".format(image_url))
            generate_text = "" # 报错则返回空，表示未识别，进入后续异常处理

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

    async def diet_recog_customer(self, **kwargs):
        """C端菜品识别，提取菜品名称、数量、单位、克重、能量信息"""
        image_url = kwargs.get("image_url")
        logger.debug(f"diet_recog_customer image_url: {image_url}")
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")

        # 检查图片 URL 是否可访问
        async with aiohttp.ClientSession() as session:
            async with session.head(image_url) as response:
                if not (response.status == 200):
                    return self._get_result(400, {}, "image_url is not accessible.")

        # 直接调用多模态大模型识别菜品名称以及数量
        messages = [{
            "role": "user",
            "content":[
                {"type": "text", "text": self.prompts["C端菜品识别"]},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        try:
            generate_text = await acallLLM(
                history=messages, max_tokens=1024, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=45
            )
        except Exception as error:
            logger.error("image_type_recog error recog diet image {0}".format(image_url))
            generate_text = "" # 报错则返回空，表示未识别，进入后续异常处理

        # 处理结果
        json_result = None
        try:  # 去掉json串中多余的内容
            json_result = self._get_food_energy_info_json(generate_text)
        except Exception as error:
            logger.error("diet_recog_customer error check json {0}".format(image_url))
        if json_result:
            json_data = {"status": 0, "foods": json_result["foods"]}
        else:
            json_data = {"status": -1, "foods": []}
        
        # 进一步处理食材内容
        if json_data["status"] == 0:
            # 请求LLM
            query = json.dumps(json_data["foods"], ensure_ascii=False)
            messages = [{
                'role': 'user',
                'content': "# 已知信息\n{0}\n\n{1}".format(query, self.prompts['C端菜品食材拆解'])
            }]
            generate_text = await acallLLM(
                history=messages, max_tokens=1536, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
            )
            # 处理返回内容
            json_result = None
            try:  # 去掉json串中多余的内容
                json_result = self._get_food_energy_info_json(generate_text, ingredient_check=True)
            except Exception as error:
                logger.error("diet_recog_customer error check ingredients json {0}".format(image_url))
            if json_result:
                json_data["foods"] = json_result["foods"]
            else:
                for i in range(len(json_data["foods"])):
                    json_data["foods"][i]["ingredients"] = []

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"diet_recog_customer success: {result}")
        return result

    async def diet_eval(self, **kwargs):
        """饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        user_info = kwargs.get("user_info", {}) or {}
        diet_info = kwargs.get("diet_info", []) or []
        management_tag = kwargs.get("management_tag", "") or ""
        diet_period = kwargs.get("diet_period", "") or ""
        diet_time = kwargs.get("diet_time", time.time())
        logger.debug(f"diet_eval user_info: {user_info} diet_info: {diet_info} management_tag: {management_tag} diet_period: {diet_period} diet_time: {diet_time}")

        # 将unix时间转换为日期格式
        try:
            diet_time = int(diet_time)
            logger.debug(f"diet_eval diet_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diet_time))}")
            diet_time = time.strftime('%H:%M', time.localtime(diet_time))
        except Exception as _:
            logger.error("diet_eval error diet_time: {0}".format(diet_time))
            return self._get_result(400, {}, "diet_time param error.")

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
        for key in [["age", "年龄"], ["gender", "性别"], ["height", "身高"], ["weight", "体重"], ["disease", "现患疾病"], ["allergy", "过敏史"]]:
            query += f"{key[1]}：{user_info.get(key[0], '') or ''}\n"
        query += f"用户管理标签：{management_tag}\n"
        query += f"用餐时间：{diet_time}\n"
        query += f"饮食信息：{json.dumps(diet_info, ensure_ascii=False)}"
        logger.debug(f"diet_eval query: {query}")

        # 请求LLM
        messages = [{
            'role': 'user',
            'content': f"{self.prompts['饮食一句话建议'].format(query)}",
        }]
        generate_text = await acallLLM(
            history=messages, max_tokens=512, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
        )
        json_data["content"] = generate_text

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"diet_eval success: {result}")
        return result

    async def diet_eval_customer(self, **kwargs):
        """C端饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        user_info = kwargs.get("user_info", {}) or {}
        diet_info = kwargs.get("diet_info", []) or []
        management_tag = kwargs.get("management_tag", "") or ""
        diet_period = kwargs.get("diet_period", "") or ""
        diet_time = kwargs.get("diet_time", time.time())
        logger.debug(f"diet_eval_customer user_info: {user_info} diet_info: {diet_info} management_tag: {management_tag} diet_period: {diet_period} diet_time: {diet_time}")

        # 将unix时间转换为日期格式
        try:
            diet_time = int(diet_time)
            logger.debug(f"diet_eval_customer diet_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diet_time))}")
            diet_time = time.strftime('%H:%M', time.localtime(diet_time))
        except Exception as _:
            logger.error("diet_eval_customer error diet_time: {0}".format(diet_time))
            return self._get_result(400, {}, "diet_time param error.")

        # 初始化返回内容
        json_data = {
            "status": 0,
            "content": "无论您选择何种餐食，都请记得关注食物的多样性和营养均衡，细嚼慢咽享受美味的一餐吧~"
        }

        # 如果没有饮食信息也就不用出其他内容了
        if len(diet_info) <= 0:
            result = self._get_result(200, json_data, "")
            logger.debug(f"diet_eval_customer success: {result}")
            return result

        # 拼接用户信息
        query = ""
        query += f"饮食信息：{json.dumps(diet_info, ensure_ascii=False)}\n"
        query += f"用餐时间：{diet_time}\n"
        for key in [
            ["age", "年龄"],
            ["gender", "性别"],
            ["bmi", "BMI"],
            ["weight_status", "体重状态"],
            ["disease", "现患疾病"],
            ["manage_object", "管理目标"],
            ["allergy_food", "食物过敏"],
            ["special_diet", "特殊饮食习惯"],
            ["taste_preference", "口味偏好"],
            ["special_period", "是否特殊生理期"],
            ["constitution", "中医体质"],
        ]:
            if key[0] in user_info and len(f"{user_info.get(key[0], '') or ''}") > 0:
                query += f"{key[1]}：{user_info.get(key[0], '') or ''}\n"
        if len(management_tag) > 0:
            query += f"用户管理标签：{management_tag}\n"
        logger.debug(f"diet_eval_customer query: {query}")

        # 请求LLM
        messages = [{
            'role': 'user',
            'content': f"{self.prompts['C端饮食评价'].format(query)}",
        }]
        generate_text = await acallLLM(
            history=messages, max_tokens=768, temperature=0, seed=42, model="Qwen1.5-32B-Chat", timeout=45
        )
        json_data["content"] = generate_text

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"diet_eval_customer success: {result}")
        return result

    async def general_recog(self, **kwargs):
        """菜品识别，提取菜品名称、数量、单位信息"""
        image_url = kwargs.get("image_url")
        prompt = kwargs.get("prompt", "") or "简述图片内容"
        logger.debug(f"general_recog image_url: {image_url} prompt: {prompt}")
        if not image_url:
            return self._get_result(400, {}, "image_url is required and cannot be empty.")

        # 检查图片 URL 是否可访问
        async with aiohttp.ClientSession() as session:
            async with session.head(image_url) as response:
                if not (response.status == 200):
                    return self._get_result(400, {}, "image_url is not accessible.")

        # 初始化返回内容
        json_data = {
            "status": 0,
            "content": ""
        }

        # 直接调用多模态大模型识别菜品名称以及数量
        messages = [{
            "role": "user",
            "content":[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        try:
            generate_text = await acallLLM(
                history=messages, max_tokens=1024, temperature=0, seed=42, is_vl=True, model="Qwen-VL-base-0.0.1", timeout=45
            )
        except Exception as error:
            logger.error("general_recog error recog diet image {0}".format(image_url))
            generate_text = "抱歉，我没能识别出这张图片的内容。请尝试上传一张更清晰的图片，或者提供更多描述信息。" # 报错则返回一句保底内容
        
        json_data["content"] = generate_text

        # 处理返回内容
        result = self._get_result(200, json_data, "")
        logger.debug(f"general_recog success: {result}")
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
注意如果是饮食评估报告，则返回"报告"。
如果图片中是运动器材、体能、健身的内容，则返回"运动"。
如果图片中是包含体重、体脂的截图内容，则返回"报告"。
如果图片中是药物、血压计血糖仪等医疗器械，或其他内容，则返回"其他"。
不要返回其他内容，仅返回类别名称。
以下是图片描述：
""",
            "运动分类": """请根据图片描述，判断内容属于"运动图片"、"运动后的报告"。
如果是健身器材或用具，则返回"运动图片"。
如果是运动后具体的报告内容，则返回"运动后的报告"。
如果以上均不符合，则返回"其他"。
不要返回其他内容，仅返回类别名称。
以下是图片描述：
""",
            "报告分类": """请根据图片描述，判断内容属于"体检报告"、"检查报告"、"检验报告"、"体重报告"、"血压报告"、"血糖报告"、"饮食报告"。
如果以上均不符合，则返回"其他报告"。
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
            "饮食一句话建议": """# 已知信息
{0}

# 任务描述
请你扮演一名资深营养师的角色，分析评价我提供的饮食清单，考虑我的健康状态如现患疾病等信息，给出专业的指导话术。
# 输出要求
- 依据饮食清单，评估是否满足当前餐次的营养需求，请考虑疾病与健康状况。
- 正餐的食物评价请根据“211法则”，调整蔬菜、蛋白质、主食比例至2:1:1，提出增减建议，增减建议给出量化标准，例如一个拳头大小，一个手掌心大小等，方便我理解。
- 如果是加餐时间的食物，可以不遵循“211法则”，你可以评价加餐食物选择是否合理，食用量是否合理等。
- 如果我提交的食物信息以及时间均利于我的健康管理，可以给予肯定话术，简单说明该餐的营养价值。
- 输出的文本应该通顺、简单明了、符合营养学观点。
- 输出内容不超过50字。
# 输出示例参考
- 你午餐选择的食材都很不错，但是蔬菜的量不足，你可以再增加1个拳头大小的深色绿叶蔬菜，比如凉拌菠菜。另外，可以再适当增加虾2-3只，补充蛋白质。

Begins!""",
            "C端饮食评价": """# 已知信息
{0}

# 任务描述
你扮演一位经验丰富的营养师，请你根据用户上传的饮食图片、用户画像、按照输出要求分析用户的饮食并给出专业评价和建议。
# 输出要求
- 你要说明当餐饮食都包含哪些食物、当餐食物的热量信息、结合用户的健康情况给出相关建议。
- 整体字符不超过200字。
# 输出示例
您图片中的美食有三明治、鸡蛋、牛奶。估算您早餐的食物热量大约为300-400kcal。您早餐的营养搭配比较均衡，补充了碳水化合物、优质蛋白质，建议可以再增加些绿叶蔬菜，或者小番茄，来保证膳食纤维、维生素的摄入。""",
            "菜品直接识别": """# 任务描述
你是一名健康饮食管理助手，你需要识别出图中食物名称、数量、单位。

# 输出要求：
- 仔细分析图片，精确识别出所有可见的食材，并对每种食材进行详细的数量统计。
- 食物名称要尽可能精确到具体食材（如炒花菜、豆芽炒肉、白米饭、紫米饭等），而非泛泛的类别。
- 根据食材的特点，给出准确且恰当的数量描述和单位。例如，使用'个'来表示完整的水果（如'1个（小）苹果'、'2个橘子'），如果是一半根黄瓜则为'0.5根黄瓜'，用'片'来表示切片的食材（如'3片面包'），对于堆积的食物可以使用'堆'、'把'等（如'1堆瓜子'、'1把葡萄'），对于肉类可以用'掌心大小'、'克'、'块'等来表示分量，蔬菜类可以用'拳头大小'、'克'、'份'等来表示分量。确保所有计数均准确无误，单位使用得当。
- 输出食物必须来自图片中，禁止自己创造。
- 以json格式输出，严格按照`输出格式样例`形式。

输出格式样例：
```json
[
    {"foodname": "玉米", "count": "2", "unit": "根"},
    {"foodname": "苹果", "count": "1", "unit": "个（小）"},
    {"foodname": "苹果", "count": "1", "unit": "个（中等）"},
    {"foodname": "黄瓜", "count": "0.5", "unit": "根"},
    {"foodname": "鸡胸肉", "count": "1", "unit": "掌心大小"},
    {"foodname": "炒花菜", "count": "1", "unit": "拳头大小"},
    {"foodname": "芹菜炒肉", "count": "1", "unit": "份"},
    {"foodname": "五花肉", "count": "3", "unit": "块"},
    {"foodname": "米饭", "count": "1", "unit": "碗"},
    {"foodname": "馒头", "count": "0.5", "unit": "块"},
    {"foodname": "西红柿炒鸡蛋", "count": "1", "unit": "份"}
]
```

Begins!""",
            "C端菜品识别-长": """# 任务描述
你是一名健康饮食管理助手，你需要识别出图中菜品名称、数量、单位、克重、能量。

# 输出要求：
- 仔细分析图片，精确识别出所有可见的菜品，并对每种菜品进行详细的数量统计。
- 食物名称要尽可能精确到具体菜品（如炒花菜、豆芽炒肉、白米饭、紫米饭等），而非泛泛的类别。
- 根据菜品的特点，给出准确且恰当的数量描述和单位。例如，使用'个'来表示完整的水果（如'1个（小）苹果'、'2个橘子'），如果是一半根黄瓜则为'0.5根黄瓜'，用'片'来表示切片的食材（如'3片面包'），对于堆积的食物可以使用'堆'、'把'等（如'1堆瓜子'、'1把葡萄'），菜品可以用'克'、'份'等来表示分量。确保所有计数均准确无误，单位使用得当。
- 克重单位为'克'，能量单位为'千卡'。
- 输出食物必须来自图片中，禁止自己创造。
- 以json格式输出，严格按照`输出格式样例`形式。

输出格式样例：
```json
[
    {"foodname": "玉米", "count": "2", "unit": "根", "weight": "180", "energy": "172"},
    {"foodname": "苹果", "count": "1", "unit": "个（小）", "weight": "100", "energy": "52"},
    {"foodname": "炒花菜", "count": "1", "unit": "份", "weight": "150", "energy": "75"},
    {"foodname": "芹菜炒肉", "count": "1", "unit": "份", "weight": "200", "energy": "220"},
    {"foodname": "米饭", "count": "1", "unit": "碗", "weight": "150", "energy": "200"},
    {"foodname": "馒头", "count": "0.5", "unit": "块", "weight": "35", "energy": "77"},
    {"foodname": "西红柿炒鸡蛋", "count": "1", "unit": "份", "weight": "200", "energy": "280"}
]
```

Begins!""",
            "C端菜品识别": """# 任务描述
你是一名健康饮食管理助手，你需要识别出图中菜品名称、克重、能量。

# 输出要求：
- 仔细分析图片，精确识别出所有可见的菜品，并对每种菜品进行详细的统计。
- 食物名称要尽可能精确到具体菜品，而非泛泛的类别。
- 克重单位为'克'，能量单位为'千卡'。
- 输出食物必须来自图片中，禁止自己创造。
- 以json格式输出，严格按照`输出格式样例`形式。

输出格式样例：
```json
[
    {"foodname": "西红柿炒鸡蛋", "weight": "200", "energy": "280"},
    {"foodname": "苹果", "weight": "150", "energy": "76"},
]
```

Begins!""",
            "C端菜品食材拆解": """# 任务描述
你是一名健康饮食管理助手，你需要识别出已知信息中每道菜的食材名称、克重、能量，并将全部信息进行补充汇总。

# 输出要求：
- 仔细分析已知信息，对每种菜品进行详细的主材分析。
- 克重单位为'克'，能量单位为'千卡'。
- 输出内容必须来自已知信息中，不明确的内容可以进行适当估计，尽量给出对应结果。
- 以json格式输出，严格按照`输出格式样例`形式。

假设已知信息如下：
[{"foodname": "西红柿炒鸡蛋", "weight": "200", "energy": "280"}, {"foodname": "苹果", "weight": "150", "energy": "76"}]
则输出格式样例为：
```json
[
    {"foodname": "西红柿炒鸡蛋", "weight": "200", "energy": "280", "ingredients": [{"name": "西红柿", "weight": "120", "energy": "40"}, {"name": "鸡蛋", "weight": "80", "energy": "240"}]},
    {"foodname": "苹果", "weight": "150", "energy": "76", "ingredients": [{"name": "苹果", "weight": "150", "energy": "76"}]},
]
```

Begins!""",
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

        if isinstance(info, list):
            info = {"foods": info}
        if "foods" not in info or not isinstance(info["foods"], list):
            raise Exception
        for i, food in enumerate(info["foods"]):
            if "foodname" not in food or "unit" not in food or "count" not in food:
                raise Exception
            # try:  # 检查数量是否为数字，不是则默认为1
            #     count = int(food["count"])
            #     if count >= 30: # 数量超过30份，大概率是大模型输出有误，返回默认为1份
            #         food["count"] = "1"
            # except Exception as _:
            #     food["count"] = "1"

        return info

    def _get_food_energy_info_json(self, result, ingredient_check=False):
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
        info = json.loads(result.replace("```", "").replace("json", "").replace("返回", "").replace("。", "").replace("\n", "").replace("},]", "}]"))

        if isinstance(info, list):
            info = {"foods": info}
        if "foods" not in info or not isinstance(info["foods"], list):
            raise Exception
        for i, food in enumerate(info["foods"]):
            if "foodname" not in food or "weight" not in food or "energy" not in food:
                raise Exception
            if ingredient_check:
                if "ingredients" not in food or not isinstance(food["ingredients"], list):
                    raise Exception
                for j, ingredient in enumerate(food["ingredients"]):
                    if "name" not in ingredient or "weight" not in ingredient or "energy" not in ingredient:
                        raise Exception

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
        # 通用识别
        result = loop.run_until_complete(multimodal_model.general_recog(**image_data[2]))
        print("#########", result)
        # C端饮食评价
        result = loop.run_until_complete(multimodal_model.diet_eval_customer(**input_data))
        print("#########", result)
    except Exception as e:
        print("发生错误：", e)
    finally:
        loop.close()
