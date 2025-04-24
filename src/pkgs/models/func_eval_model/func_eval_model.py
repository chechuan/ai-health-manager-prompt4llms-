import json
import time

from src.pkgs.models.func_eval_model.func_eva_util import *
from src.prompt.model_init import acallLLM, callLLM
from src.utils.Logger import logger
from data.jiahe_util import get_userInfo

from src.pkgs.models.func_eval_model.glucose_analysis import GlucoseAnalyzer
from src.pkgs.models.func_eval_model.daily_image_analysis import analyze_diet_and_glucose

async def image_recog(img):
        """识别图片类型"""
        if not img:
            yield {"image_type": "", "head": 402, "err_msg":"请上传图片", "end":True}
        prompt = get_func_eval_prompt('img_recog_prompt')
        messages = [
            {
                "role": "user",
                "content":[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img}
                    }
                ]
            }
        ]
        logger.debug(
            "图片识别模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=200,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            is_vl=True,
            model="Qwen-VL-base-0.0.1",
        )
        logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("图片识别模型输出： " + generate_text)
        img_type = get_standard_img_type(generate_text)
        yield {"image_type": img_type, "head": 200, "err_msg":"", "end":True}


async def diet_image_recog(img):
    if not img:
        yield {"content": "请上传图片", "head": 402, "err_msg": "请上传图片", "end": True}
    yield_item = image_recog(img)
    async for item in yield_item:
        img_type = {**item}.get('image_type', '')
    if img_type == "diet_image":
        prompt = get_func_eval_prompt('diet_image_recog_prompt')
        messages = [
            {
                "role": "user",
                "content":[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img}
                    }
                ]
            }
        ]
        logger.debug(
            "饮食图片识别模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=1000,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            is_vl=True,
            model="Qwen-VL-base-0.0.1",
        )
        logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("饮食图片识别模型输出： " + generate_text)
        generate_text = generate_text.replace("`", '').replace("json", '').strip()
        generate_text = generate_text[generate_text.find('['): generate_text.rfind(']') + 1]
        content = json.loads(generate_text.strip())

        yield {"content": content, "head": 200, "err_msg":"", "end":True}


async def extract_imgInfo(history, daily_diet_info):
    """
    修改时间：2025年4月17日
    1. 将 `img_info` 改为字典形式，存储图片 URL 和对应索引，确保正确匹配 `daily_diet_info` 或 `history`。
    2. 更新时通过字典中的索引来准确更新对应项，避免索引错位。
    3. 添加 JSON 解析异常捕获，确保文本格式正确。
    4. 修改饮食识别提示词，增加“无法识别食材时不输出任何信息”的要求。
    """
    img_info = []

    # 如果有 history，提取图片链接和索引
    if history:
        for idx, h in enumerate(history):
            if h['content'].startswith('http'):
                img_info.append({'url': h['content'], 'index': idx})  # 用字典存储 URL 和索引
    elif daily_diet_info:
        for i, info in enumerate(daily_diet_info):
            if info.get('diet_image', '') and not info.get('diet_info', ''):
                img_info.append({'url': info['diet_image'], 'index': i})  # 用字典存储 URL 和索引

    # 遍历 img_info，根据字典中的索引来更新 daily_diet_info
    for img_entry in img_info:
        img_url = img_entry['url']
        index = img_entry['index']

        prompt = get_func_eval_prompt('img_sumUp_prompt') if history else diet_image_recog_prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url}
                    }
                ]
            }
        ]
        logger.debug(
            "图片信息总结/识别模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=1000,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            is_vl=True,
            model="Qwen-VL-base-0.0.1",
        )
        logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("图片信息总结/识别模型输出： " + generate_text)

        # 检查生成的文本是否为空或无效
        if not generate_text.strip():
            logger.error("图片信息识别模型返回为空或无效内容.")
            continue

        # 清理和截取有效的 JSON 部分
        if history:
            history[img_entry['index']]['content'] = "上传了一张图片，图片信息为：" + generate_text
        else:
            generate_text = generate_text.replace("`", '').replace("json", '').strip()
            generate_text = generate_text[generate_text.find('['): generate_text.rfind(']') + 1]
            try:
                content = json.loads(generate_text.strip())
                # 使用字典中的索引直接更新 daily_diet_info
                if not daily_diet_info[index].get('diet_info', []):
                    daily_diet_info[index]['diet_info'] = content
                else:
                    logger.debug(f"Skipping modification for {index}, diet_info already exists.")
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解码错误： {e}")
                continue

    yield history if history else daily_diet_info

async def schedule_tips_modify(schedule_template, history, cur_time):
    """日程tips修改"""
    yield_item = extract_imgInfo(history=history, daily_diet_info=[])
    his = []
    async for item in yield_item:
        his = item
    his_str = get_history_info(his)
    prompt = get_func_eval_prompt('schedule_modify_prompt')
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                schedule_template,
                his_str,
                cur_time,
            ),
        }
    ]
    logger.debug(
        "日程tips修改模型输入： " + json.dumps(messages, ensure_ascii=False)
    )
    start_time = time.time()
    generate_text = await acallLLM(
        history=messages,
        max_tokens=1000,
        top_p=0.9,
        temperature=0.8,
        do_sample=True,
        is_vl=True,
        model="Qwen1.5-32B-Chat",
    )
    logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
    logger.debug("日程tips修改模型输出： " + generate_text)
    yield {"schedule_tips": generate_text, "head": 200, "err_msg": "", "end": True}

async def sport_schedule_tips_modify(schedule, history, cur_time):
    """运动日程修改"""
    yield_item = extract_imgInfo(history=history, daily_diet_info=[])
    his = []
    async for item in yield_item:
        his = item
    his_str = get_history_info(his[-4:])
    sch_str = get_sch_str(schedule)
    # prompt = get_func_eval_prompt('sport_schedule_recog_prompt')
    prompt = sport_schedule_recog_prompt
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                sch_str,
                his_str,
                cur_time
            ),
        }
    ]
    logger.debug(
        "运动日程修改模型输入： " + prompt.format(
                sch_str,
                his_str,
                cur_time,
            )
    )
    start_time = time.time()
    generate_text = await acallLLM(
        history=messages,
        max_tokens=1000,
        top_p=0.9,
        temperature=0.8,
        # stream=True,
        is_vl=True,
        model="Qwen1.5-32B-Chat",
    )
    logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
    logger.debug("运动日程修改模型输出： " + generate_text)
    generate_text = generate_text.replace("`", '').replace("json", '').strip()
    generate_text = generate_text[generate_text.find('{'): generate_text.rfind('}') + 1].replace('\n\\', '').replace('\n', '').replace(' ', '')
    content = json.loads(generate_text.strip())
    if not content.get('is_modify'):
        return {"is_modify":False, "category": "","modify_reason": "", "modify_suggestion": "", "head": 200, "err_msg": "", "days":"无", "end": True}
    else:
        cat_code = sport_category_mapping.get(content.get('category', ''),'HS004')
        return {"is_modify":True, "category": cat_code,"modify_reason": content.get('reason', ''), "modify_suggestion": content.get('suggestion', ''), "days":content.get('days', '3'), "head": 200, "err_msg": "", "end": True}


async def daily_diet_degree(userInfo, daily_diet_info, daily_blood_glucose, management_tag='血糖管理'):
    """一日饮食状态"""
    yield_item = extract_imgInfo(history=[], daily_diet_info=daily_diet_info)
    daily_diet_info = []
    async for item in yield_item:
        daily_diet_info = item
    daily_diet_str = get_daily_diet_str(daily_diet_info, daily_blood_glucose)
    prompt = daily_diet_degree_prompt
    userInfo_str = get_userInfo(userInfo)
    daily_bg, meal_minmax_bg = get_daily_key_bg(daily_blood_glucose, daily_diet_info)
    bg_str = get_daily_blood_glucose_str(daily_bg)
    glucose_analyses = GlucoseAnalyzer().analyze_glucose_data(daily_blood_glucose)
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                userInfo_str,
                daily_diet_str,
                bg_str,
                glucose_analyses.get('summary', ''),
                meal_minmax_bg,
                management_tag,
            ),
        }
    ]
    logger.debug(
        "一日饮食等级模型输入： " + prompt.format(userInfo_str, daily_diet_str, bg_str,
                                                     glucose_analyses.get('summary', ''),meal_minmax_bg, management_tag)
    )
    start_time = time.time()
    generate_text = await acallLLM(
        history=messages,
        max_tokens=1000,
        top_p=0.9,
        temperature=0.8,
        # stream=True,
        is_vl=True,
        model="Qwen1.5-32B-Chat",
    )
    logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
    logger.debug("一日饮食等级模型输出： " + generate_text)
    generate_text = generate_text[generate_text.index('Output')+7:].strip().split('\n')[0]
    if '欠佳' in generate_text:
        res = '欠佳'
    elif '尚可' in generate_text:
        res = '尚可'
    elif '极佳' in generate_text:
        res = '极佳'
    else:
        res = '尚可'
    return {"dietStatus": res}

async def daily_diet_eval(userInfo, daily_diet_info, daily_blood_glucose, management_tag='血糖管理'):
    """一日饮食评估建议"""
    yield_item = extract_imgInfo(history=[], daily_diet_info=daily_diet_info)
    daily_diet_info = []
    async for item in yield_item:
        daily_diet_info = item
    daily_diet_str = get_daily_diet_str(daily_diet_info, daily_blood_glucose)

    # prompt = get_func_eval_prompt('daily_diet_eval_prompt')
    prompt = daily_diet_eval_prompt
    userInfo_str = get_userInfo(userInfo)
    daily_bg, meal_minmax_bg = get_daily_key_bg(daily_blood_glucose, daily_diet_info)
    bg_str = get_daily_blood_glucose_str(daily_bg)
    glucose_analyses = GlucoseAnalyzer().analyze_glucose_data(daily_blood_glucose)
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                userInfo_str,
                daily_diet_str,
                bg_str,
                glucose_analyses.get('summary', ''),
                meal_minmax_bg,
                management_tag,
            ),
        }
    ]
    logger.debug(
        "一日饮食评估建议模型输入： " + prompt.format(userInfo_str,daily_diet_str, bg_str, glucose_analyses.get('summary', ''), meal_minmax_bg, management_tag)
    )
    start_time = time.time()
    generate_text = await acallLLM(
        history=messages,
        max_tokens=1000,
        top_p=0.9,
        temperature=0.8,
        do_sample=True,
        stream=True,
        is_vl=True,
        model="Qwen1.5-32B-Chat",
    )
    # logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
    # logger.debug("一日饮食评估建议模型输出： " + generate_text)
    # yield {"message": generate_text.replace('\n\n', '\n'), "end": True}
    #response_time = time.time()
    content = ""
    printed = False
    async for i in generate_text:
        t = time.time()
        msg = i.choices[0].delta.to_dict()
        text_stream = msg.get("content", "").replace('\n\n', '\n')
        if text_stream:
            if not printed:
                print(f"latency first token {t - start_time:.2f} s")
                printed = True
            content += text_stream
            yield {'message': text_stream, 'end': False}
    logger.debug("一日饮食评估建议模型输出： " + content)
    yield {'message': "", 'prompt': prompt.format(userInfo_str, daily_diet_str, bg_str, glucose_analyses.get('summary', ''), meal_minmax_bg, management_tag), 'end': True}
