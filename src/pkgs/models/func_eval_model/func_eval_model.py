import json
import time

from src.pkgs.models.func_eval_model.func_eva_util import *
from src.prompt.model_init import acallLLM, callLLM
from src.utils.Logger import logger

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
            max_tokens=500,
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

async def convert_imgInfo_to_text_of_history(history):
    for i, h in enumerate(history):
        if h['content'].startswith('http'):
            prompt = get_func_eval_prompt('img_info_prompt')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": h['content']}
                        }
                    ]
                }
            ]
            logger.debug(
                "图片信息总结模型输入： " + json.dumps(messages, ensure_ascii=False)
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
            logger.debug("图片信息总结模型输出： " + generate_text)
            history[i]['content'] = "上传了一张图片，图片信息为：" + generate_text

        yield history


async def schedule_tips_modify(schedule_template, history, cur_time):
    """日程修改"""
    yield_item = convert_imgInfo_to_text_of_history(history)
    his = []
    async for item in yield_item:
        his = item
    his_str = get_history_info(his)
    # prompt = get_func_eval_prompt('schedule_modify_prompt')
    prompt = schedule_modify_prompt
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
        max_tokens=200,
        top_p=0.9,
        temperature=0.8,
        do_sample=True,
        # stream=True,
        is_vl=True,
        model="Qwen1.5-32B-Chat",
    )
    logger.debug(f"latency {time.time() - start_time:.2f} s -> response")
    logger.debug("日程tips修改模型输出： " + generate_text)
    yield {"schedule_tips": generate_text, "head": 200, "err_msg": "", "end": True}