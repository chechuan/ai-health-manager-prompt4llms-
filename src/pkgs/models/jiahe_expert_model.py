import json
import time
from datetime import datetime, timedelta
import asyncio
from typing import Generator
from string import Template
from data.jiahe_prompt import *
from data.jiahe_util import *

from src.prompt.model_init import ChatMessage, acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import parse_generic_content


class JiaheExpertModel:
    def __init__(self, gsr: InitAllResource) -> None:
        self.gsr = gsr

    @staticmethod
    def is_gather_userInfo(userInfo={}, history=[]):
        """判断是否需要收集用户信息"""
        info, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_confirm_collect_userInfo.format(info, his_prompt),
            }
        ]
        logger.debug(
            "判断是否收集信息模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("判断是否收集信息模型输出： " + generate_text)
        if "是" in generate_text:
            if history:
                # 1. 判断回复是否在语境中
                messages = [
                    {
                        "role": "user",
                        "content": jiahe_collect_userInfo_in_context_prompt.format(
                            his_prompt
                        ),
                    }
                ]
                logger.debug(
                    "判断是否在语境中模型输入： "
                    + json.dumps(messages, ensure_ascii=False)
                )
                generate_text = callLLM(
                    history=messages,
                    max_tokens=2048,
                    top_p=0.9,
                    temperature=0.8,
                    do_sample=True,
                    model="Qwen1.5-72B-Chat",
                )
                logger.debug("判断是否在语境中模型输出： " + generate_text)
                generate_text = (
                    generate_text[generate_text.find("Output") + 6:]
                    .split("\n")[0]
                    .strip()
                )
                if "否" in generate_text:
                    return {"result": "outContext"}
                else:
                    # 2. 判断是否终止
                    messages = [
                        {
                            "role": "user",
                            "content": jiahe_confirm_terminal_prompt.format(his_prompt),
                        }
                    ]
                    logger.debug(
                        "判断是否终止模型输入： "
                        + json.dumps(messages, ensure_ascii=False)
                    )
                    generate_text = callLLM(
                        history=messages,
                        max_tokens=2048,
                        top_p=0.9,
                        temperature=0.8,
                        do_sample=True,
                        model="Qwen1.5-72B-Chat",
                    )
                    logger.debug("判断是否终止模型输出： " + generate_text)
                    if "中止" in generate_text:
                        return {"result": "terminal"}
                    else:
                        return {"result": "order"}
        else:
            return {"result": "terminal"}

    @staticmethod
    async def gather_userInfo(userInfo={}, history=[]):
        """生成收集用户信息问题"""
        info, his_prompt = get_userInfo_history(userInfo, history)
        # 生成收集信息问题
        messages = [
            {
                "role": "user",
                "content": jiahe_collect_userInfo.format(info, his_prompt),
            }
        ]
        logger.debug("收集信息模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        logger.debug("收集信息模型输出： " + generate_text)
        yield {"message": generate_text, "terminal": False, "end": True}

    @staticmethod
    async def eat_health_qa(query):
        messages = [
            {
                "role": "system",
                "content": jiahe_health_qa_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]  # + history
        logger.debug(
            "健康吃知识问答模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen2-7B-Instruct",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("健康吃知识问答模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_diet_principle(cur_date, location, personal_dietary_requirements,history=[], userInfo={}):
        """出具饮食调理原则"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_daily_diet_principle_prompt.format(
                    userInfo, cur_date, location, his_prompt, personal_dietary_requirements
                ),
            }
        ]
        logger.debug(
            "出具饮食调理原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        # logger.debug(
        #     "出具饮食调理原则模型输入：\n" + "\n".join(
        #         [
        #             f"{item['role']}: {item['content']}"
        #             for item in messages
        #         ]
        #     )
        # )

        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("出具饮食调理原则模型输出latancy： " + str(time.time() - start_time))
        logger.debug("出具饮食调理原则模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_family_principle(
            users, cur_date, location, history=[], requirements=[]
    ):
        """出具家庭饮食原则"""
        roles, familyInfo, his_prompt = get_familyInfo_history(users, history)
        t = Template(jiahe_family_diet_principle_prompt)
        prompt = t.substitute(
            num=len(users),
            roles=roles,
            requirements="，".join(requirements),
            family_info=familyInfo,
            cur_date=cur_date,
            location=location,
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        # logger.debug(
        #     "出具家庭饮食原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        # )

        logger.debug(
            "出具饮食调理原则模型输入：\n" + "\n".join(
                [
                    f"{item['role']}: {item['content']}"
                    for item in messages
                ]
            )
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-72B-Chat",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("出具家庭饮食原则模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_family_diet(
            users,
            cur_date,
            location,
            family_principle,
            days,
            history=[],
            requirements=[],
            reference_diet=[],
    ):
        """出具家庭N日饮食计划"""
        roles, familyInfo, his_prompt = get_familyInfo_history_0914(users, history)
        diet_cont = []
        if reference_diet:
            diet_cont.extend(reference_diet)
        days_str = str(days) + "天"

        # 生成家庭每日饮食计划的异步函数
        async def generate_day_plan(i):
            cur_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            ref_diet_str = "\n".join(diet_cont[-2:])

            prompt = jiahe_family_diet_prompt.format(
                num=len(users),
                roles=roles,
                requirements="，".join(requirements),
                family_info=familyInfo,
                cur_date=cur_date,
                location=location,
                family_principle=family_principle,
                reference_diet=ref_diet_str,
                days=days_str,
            )
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            logger.debug(
                f"出具家庭第{i + 1}日饮食计划模型输入： "
                + json.dumps(messages, ensure_ascii=False)
            )
            start_time = time.time()
            generate_text = await acallLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen2-72B-Instruct",
            )

            # 打印生成的文本格式
            logger.debug(f"生成的文本内容：\n{generate_text}")

            diet_cont.append(generate_text)
            logger.debug(f"出具家庭第{i + 1}日饮食计划模型输出：{generate_text}耗时: {time.time() - start_time:.2f} s")

            # 返回生成的文本
            return {"day": i + 1, "plan": generate_text}

        # 创建并发任务列表
        tasks = [generate_day_plan(i) for i in range(days)]

        # 使用 asyncio.as_completed() 逐步输出完成的计划
        for task in asyncio.as_completed(tasks):
            result = await task
            yield {"message": result["plan"], "end": False}

        # 输出结束标记
        yield {"message": "", "end": True}

    @staticmethod
    async def gen_nutrious_principle(cur_date, location, history=[], userInfo={}):
        """出具营养素原则"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_nutrious_principle_prompt.format(
                    userInfo, cur_date, location, his_prompt
                ),
            }
        ]
        logger.debug(
            "出具营养素原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("出具营养素原则模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_nutrious(
            cur_date, location, nutrious_principle, history=[], userInfo={}
    ):
        """营养素计划"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_nutrious_prompt.format(
                    userInfo, cur_date, location, his_prompt, nutrious_principle
                ),
            }
        ]
        logger.debug("营养素计划模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("营养素计划模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_guess_asking(userInfo, scene_flag, question="", diet=""):
        """猜你想问"""
        userInfo, _ = get_userInfo_history(userInfo)
        # 1. 生成猜你想问问题列表
        if scene_flag == "intent":
            prompt = jiahe_guess_asking_userInfo_prompt.format(userInfo)
        elif scene_flag == "user_query":
            prompt = jiahe_guess_asking_userQuery_prompt.format(question, userInfo)
        else:
            prompt = jiahe_guess_asking_diet_prompt.format(diet, userInfo)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        logger.debug(
            "猜你想问问题模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen2-7B-Instruct",
        )
        logger.debug("猜你想问问题模型输出： " + generate_text.replace("\n", " "))

        # 2. 对问题列表做饮食子意图识别
        messages = [
            {
                "role": "user",
                "content": jiahe_guess_asking_intent_query_prompt.format(generate_text),
            }
        ]
        logger.debug(
            "猜你想问意图识别模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("猜你想问模型意图识别输出： " + generate_text.replace("\n", " "))
        qs = generate_text.split("\n")
        res = []
        for i in qs:
            try:
                x = json.loads(i)
                if "其他" in x["intent"]:
                    continue
                res.append(x["question"])
            except Exception as err:
                continue
            finally:
                continue
        yield {"message": "\n".join(res[:3]), "end": True}

    @staticmethod
    async def gen_diet_effect(diet):
        """食谱功效"""
        messages = [
            {
                "role": "user",
                "content": jiahe_physical_efficacy_prompt.format(diet),
            }
        ]
        logger.debug(
            "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        response_time = time.time()
        print(f"latency {response_time - start_time:.2f} s -> response")
        logger.debug("一日食物功效模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    # @staticmethod
    # async def gen_daily_diet(cur_date, location, diet_principle, reference_daily_diets, history=[], userInfo={}):
    #     """个人一日饮食计划"""
    #     userInfo, his_prompt = get_userInfo_history(userInfo, history)
    #
    #     # 1. 生成一日食谱
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": jiahe_daily_diet_principle_prompt.format(userInfo, cur_date, location, his_prompt, his_prompt),
    #         }
    #     ]
    #     logger.debug(
    #         "一日饮食计划模型输入： " + json.dumps(messages, ensure_ascii=False)
    #     )
    #     generate_text = callLLM(
    #         history=messages,
    #         max_tokens=1024,
    #         top_p=0.9,
    #         temperature=0.8,
    #         do_sample=True,
    #         model="Qwen1.5-72B-Chat",
    #     )
    #
    #     # 2. 生成食谱的实物功效
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": jiahe_physical_efficacy_prompt.format(generate_text),
    #         }
    #     ]
    #     logger.debug(
    #         "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
    #     )
    #     start_time = time.time()
    #     generate_text = callLLM(
    #         history=messages,
    #         max_tokens=1024,
    #         top_p=0.9,
    #         temperature=0.8,
    #         do_sample=True,
    #         stream=True,
    #         model="Qwen1.5-72B-Chat",
    #     )
    #     response_time = time.time()
    #     print(f"latency {response_time - start_time:.2f} s -> response")
    #     content = ""
    #     printed = False
    #     for i in generate_text:
    #         t = time.time()
    #         msg = i.choices[0].delta.to_dict()
    #         text_stream = msg.get("content")
    #         if text_stream:
    #             if not printed:
    #                 print(f"latency first token {t - start_time:.2f} s")
    #                 printed = True
    #             content += text_stream
    #             yield {'message': text_stream, 'end': False}
    #     logger.debug("一日食物功效模型输出： " + content)
    #     yield {'message': "", 'end': True}

    @staticmethod
    async def gen_n_daily_diet(
            cur_date,
            location,
            diet_principle,
            personal_dietary_requirements,
            reference_daily_diets,
            days,
            history=[],
            userInfo={}
    ):
        """生成N日饮食计划"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        diet_cont = []
        if reference_daily_diets:
            diet_cont.extend(reference_daily_diets)

        # 生成每日饮食计划的异步函数
        async def generate_day_plan(i):
            cur_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            ref_diet_str = "\n".join(diet_cont[-2:])
            messages = [
                {
                    "role": "user",
                    "content": jiahe_daily_diet_prompt.format(
                        userInfo,
                        cur_date,
                        location,
                        his_prompt,
                        diet_principle,
                        ref_diet_str,
                        personal_dietary_requirements,
                    ),
                }
            ]
            logger.debug(f"生成第 {i + 1} 天饮食计划输入： " + json.dumps(messages, ensure_ascii=False))
            start_time = time.time()
            generate_text = await acallLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                model="Qwen1.5-32B-Chat",
            )
            logger.info(f"第 {i + 1} 天饮食计划生成时间：{time.time() - start_time} 秒")
            diet_cont.append(generate_text)
            return {"day": i + 1, "plan": generate_text}

        # 统一使用并发方式生成计划
        tasks = [generate_day_plan(i) for i in range(days)]

        # 使用 asyncio.as_completed() 逐步输出完成的计划
        for task in asyncio.as_completed(tasks):
            result = await task
            yield {"message": result["plan"], "end": False}

        # 输出结束标记
        yield {"message": "", "end": True}

    @staticmethod
    async def long_term_nutritional_management(userInfo={}, history=[]):
        """长期营养管理意图识别"""
        info, his_prompt = get_userInfo_history(userInfo, history[-1:])
        #
        messages = [
            {
                "role": "user",
                "content": jiahe_recognition_nutritious_manage_prompt.format(his_prompt),
            }
        ]
        logger.debug("长期营养管理识别模型输入： " + json.dumps(messages, ensure_ascii=False))
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("长期营养管理识别模型输出： " + generate_text)
        generate_text = generate_text[generate_text.find('Output')+6:].split('\n')[0].strip()
        if '否' in generate_text:
            yield {'underlying_intent': False, 'message': '', 'end': True}
        else:
            messages = [
                {
                    "role": "user",
                    "content": jiahe_nutritious_manage_prompt.format(userInfo, his_prompt)
                }
            ]
            logger.debug("长期营养管理话术模型输入： " + json.dumps(messages, ensure_ascii=False))
            generate_text = callLLM(
                history=messages,
                max_tokens=2048,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                model="Qwen1.5-32B-Chat",
            )
            logger.debug("长期营养管理话术模型输出： " + generate_text)
            yield {"message": generate_text, "underlying_intent": True, "end": True}

    @staticmethod
    async def gen_current_diet(
            cur_date,
            location,
            diet_principle,
            reference_daily_diets,
            meal_number,
            history=[],
            userInfo={},
            today_diet=''
    ):
        """生成当餐食谱"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        ref_diet_str = "\n".join(reference_daily_diets[-2:])
        messages = [
            {
                "role": "user",
                "content": jiahe_current_diet_prompt.format(
                    userInfo,
                    cur_date,
                    location,
                    his_prompt,
                    diet_principle,
                    ref_diet_str,
                    today_diet,
                    meal_number
                ),
            }
        ]
        logger.debug(
            "当餐食谱模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.info("当餐食谱模型生成时间：" + str(time.time() - start_time))
        logger.debug(
            "当餐食谱模型输出： " + json.dumps(generate_text, ensure_ascii=False)
        )
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def guess_asking_child(userInfo, dish_efficacy, nutrient_efficacy):
        """儿童猜你想问"""
        userInfo = get_userInfo(userInfo)
        messages = [
            {
                "role": "user",
                "content": jiahe_child_guess_asking.format(
                    userInfo, dish_efficacy, nutrient_efficacy
                ),
            }
        ]
        logger.debug(
            "儿童猜你想问模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=512,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("儿童猜你想问模型输出latancy： " + str(time.time() - start_time))
        logger.debug("儿童猜你想问模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}


    @staticmethod
    async def gen_child_diet_principle(userInfo):
        """儿童饮食调理原则"""
        userInfo = get_userInfo(userInfo)
        messages = [
            {
                "role": "user",
                "content": jiahe_child_diet_principle.format(
                    userInfo
                ),
            }
        ]
        logger.debug(
            "儿童饮食原则模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=512,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("儿童饮食原则模型输出latancy： " + str(time.time() - start_time))
        logger.debug("儿童饮食原则模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_child_nutrious_effect(userInfo, cur_date, location):
        """儿童营养素补充剂及功效"""
        info = get_userInfo(userInfo)
        messages = [
            {
                "role": "user",
                "content": jiahe_child_nutrient_effect_prompt.format(
                    info, cur_date, location
                ),
            }
        ]
        logger.debug(
            "儿童营养素补充剂及功效模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = await acallLLM(
            history=messages,
            max_tokens=512,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("儿童营养素补充剂及功效模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}


    @staticmethod
    async def child_dish_rec(userInfo, cur_date, location, ref_dish, dish_principle):
        # 1. 生成菜品、功效
        userInfo_str = get_userInfo(userInfo)
        messages = [
            {
                "role": "user",
                "content": jiahe_child_dish_effect.format(
                    userInfo_str, cur_date, location, ref_dish, dish_principle
                ),
            }
        ]
        logger.debug(
            "儿童菜品功效模型输入： " + json.dumps(messages, ensure_ascii=False)
        )
        start_time = time.time()
        generate_text = callLLM(
            history=messages,
            max_tokens=512,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            # stream=True,
            model="Qwen1.5-32B-Chat",
        )
        logger.debug("儿童菜品功效模型输出latancy： " + str(time.time() - start_time))
        logger.debug("儿童菜品功效模型输出： " + generate_text)
        generate_text = generate_text.replace("`", '').replace("json", '').strip()
        generate_text = generate_text[generate_text.find('{'): generate_text.rfind('}')+1]
        dish = json.loads(generate_text.strip())
        name = dish.get('菜肴名称', '')
        effect = dish.get('菜肴营养价值功效', '')

        # 2. 匹配库里字段
        dish_data = get_dish_from_database(name, userInfo)
        image = ''
        if dish_data:
            caloric = dish_data.get('calories', 0)
            carbon_water = dish_data.get('carbon_water', 0)
            protein = dish_data.get('protien', 0)
            fat = dish_data.get('fat', 0)
            image = dish_data.get('image', '')
        else:
            messages = [
                {
                    "role": "user",
                    "content": jiahe_gen_dish_nutrient_caloric.format(
                        name
                    ),
                }
            ]
            logger.debug(
                "儿童菜品热量模型输入： " + json.dumps(messages, ensure_ascii=False)
            )
            start_time = time.time()
            generate_text = callLLM(
                history=messages,
                max_tokens=512,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                # stream=True,
                model="Qwen1.5-32B-Chat",
            )
            logger.debug("儿童菜品热量模型输出latancy： " + str(time.time() - start_time))
            logger.debug("儿童菜品热量模型输出： " + generate_text)

            generate_text = generate_text.replace("`", '').replace("json", '').replace("python", '').strip()
            generate_text = generate_text[generate_text.find('{'): generate_text.rfind('}') + 1]
            d = json.loads(generate_text.strip())
            caloric = d.get('每100克可食用部分', {}).get('热量值', 0)
            carbon_water = d.get('每100克可食用部分', {}).get('碳水化合物含量', 0)
            protein  = d.get('每100克可食用部分', {}).get('蛋白质含量', 0)
            fat = d.get('每100克可食用部分', {}).get('脂肪含量', 0)

        if caloric == 0:
            carbon_water_ratio = 0
            protein_ratio = 0
            fat_ratio = 0
        else:
            carbon_water_ratio = round(float(carbon_water * 4 / caloric) * 100, 2)
            protein_ratio = round(float(protein * 4 / caloric) * 100, 2)
            fat_ratio = round(float(fat * 9 / caloric) * 100, 2)

        if carbon_water_ratio + protein_ratio + fat_ratio > 100:
            carbon_water_ratio = max(round(0.0, 1), round(float(99.5 - protein_ratio - fat_ratio), 2))

        yield {"message": {"dish_name": name, "dish_effect": effect, "image": image,
                           "nutrient_elements": [
                               {"nutrient_name": "碳水化合物", "content": round(float(carbon_water), 2),
                                "caloric_ratio": carbon_water_ratio},
                               {"nutrient_name": "蛋白质", "content": round(float(protein), 2),
                                "caloric_ratio": protein_ratio},
                               {"nutrient_name": "脂肪", "content": round(float(fat), 2),
                                "caloric_ratio": fat_ratio}
                           ]}, "end": True}






        # start_time = time.time()
        # generate_text = callEmbedding(
        #     inputs=dish,
        #     model="bce-embedding-base-v1",
        # )
        #
        # logger.info("bce embedding模型生成时间：" + str(time.time() - start_time))
        # logger.debug(
        #     "bce embedding模型输出： " + json.dumps(generate_text, ensure_ascii=False)
        # )


    async def call_function(self, **kwargs) -> Union[str, Generator]:
        """调用函数
        - Args:
            intentCode (str): 意图代码
            prompt (str): 问题
            options (List[str]): 选项列表

        - Returns:
            Union[str, AsyncGenerator]: 返回字符串或异步生成器
        """
        intent_code = kwargs.get("intentCode")
        try:
            # 使用 getattr 获取类中的方法
            func = getattr(self, intent_code)
        except AttributeError:
            raise ValueError(f"Code '{intent_code}' does not对应一个有效的异步函数.")

        try:
            # 调用异步或同步函数
            if asyncio.iscoroutinefunction(func):
                content = await func(**kwargs)
            else:
                content = func(**kwargs)

        except Exception as e:
            logger.exception(f"call_function {intent_code} error: {e}")
            raise e

        return content


    async def aigc_jiahe_seasonal_ingredients_generation(self, **kwargs):
        """
        推荐时令食材，结合节气、地域和营养管理目标

        需求文档地址: https://your-document-link.com

        参数:
        - cur_date (str): 当前日期，系统自动获取，格式为"YYYY年MM月DD日"。
        - location (str): 用户所在的地域信息，如"北京"。若未知，则不拼入提示词。
        - nutrition_goal (str): 营养管理目标，如"控制血糖"、"维持心脏健康"等。若未知，则不拼入提示词。

        返回:
        - 生成符合时令的食材名称及其营养价值列表，格式为List[Dict]，包括20种食材及其营养描述。
        """
        # 当前日期，传递则使用，未传递则自动获取
        cur_date = kwargs.get('cur_date', '') or datetime.now().strftime("%Y年%m月%d日")

        # 其他参数
        location = kwargs.get('location', '')
        nutrition_goal = kwargs.get('nutrition_goal', '')

        # 拼接用户和历史数据
        userInfo, his_prompt = get_userInfo_history({}, [])

        model_args = kwargs.get("model_args", {})
        stream = model_args.get("stream", False)

        # 构建提示词，基于当前日期、地域和营养目标
        messages = [
            {
                "role": "user",
                "content": jiahe_seasonal_ingredients_prompt.format(
                    cur_date=cur_date,
                    location=location,
                    nutrition_goal=nutrition_goal
                ),
            }
        ]

        # 日志记录模型输入
        # logger.debug("时令食材推荐模型输入： " + json.dumps(messages, ensure_ascii=False))
        logger.debug("时令食材推荐模型输入：" + repr(messages[0]['content']))

        start_time = time.time()

        # 调用大语言模型生成推荐食材及其营养价值
        generate_text: Union[str, Generator] = await acallLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=stream,  # 如果需要流式输出可以启用
            model="Qwen2-7B-Instruct",  # 指定使用的模型
        )

        # 日志记录模型响应延迟和结果
        logger.debug("时令食材推荐模型输出latency： " + str(time.time() - start_time))
        # logger.debug("时令食材推荐模型输出： " + generate_text)
        # 返回生成的推荐结果
        content = await parse_generic_content(generate_text)
        return content


    async def aigc_jiahe_ingredient_pairing_suggestion(self, **kwargs):
        """
        根据已知食材生成搭配建议，输出搭配后的营养价值

        需求文档地址: https://your-document-link.com

        参数:
        - ingredient_name (str): 已知食材名称，用户输入的必填项。
        - cur_date (str): 当前日期，系统自动获取，格式为"YYYY年MM月DD日"。若未知，则不拼入提示词。
        - location (str): 用户所在的地域信息，如"北京"。若未知，则不拼入提示词。
        - nutrition_goal (str): 营养管理目标，如"控制血糖"、"维持心脏健康"等。若未知，则不拼入提示词。

        返回:
        - 生成最多3个与已知食材完美搭配的家常菜肴组合及其营养价值。
        """
        # 获取传递的参数
        ingredient_name = kwargs.get('ingredient_name', '')

        # 校验必填项
        if not ingredient_name:
            raise ValueError("食材名称为必填项，不能为空！")

        cur_date = kwargs.get('cur_date', '') or datetime.now().strftime("%Y年%m月%d日")
        location = kwargs.get('location', '')
        nutrition_goal = kwargs.get('nutrition_goal', '')

        # 拼接用户和历史数据
        # userInfo, his_prompt = get_userInfo_history({}, [])

        model_args = kwargs.get("model_args", {})
        stream = model_args.get("stream", False)

        # 构建提示词，基于食材、日期、地域和营养目标
        messages = [
            {
                "role": "user",
                "content": jiahe_ingredient_pairing_prompt.format(
                    ingredient_name=ingredient_name,
                    cur_date=cur_date,
                    location=location,
                    nutrition_goal=nutrition_goal
                ),
            }
        ]

        # 日志记录模型输入
        logger.debug("食材搭配建议模型输入：" + repr(messages[0]['content']))

        start_time = time.time()

        # 调用大语言模型生成食材搭配建议
        generate_text: Union[str, Generator] = await acallLLM(
            history=messages,
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,  # 如果需要流式输出可以启用
            model="Qwen1.5-32B-Chat",  # 指定使用的模型
        )

        # 日志记录模型响应延迟和结果
        logger.debug("食材搭配建议模型输出latency： " + str(time.time() - start_time))

        # 返回生成的推荐结果
        content = await parse_generic_content(generate_text)
        return content


    async def aigc_jiahe_ingredient_pairing_taboo(self, **kwargs):
        """
        判断食材搭配禁忌，结合食材的营养属性与搭配的禁忌。

        需求文档地址: https://your-document-link.com

        参数:
        - ingredient_name (str): 食材名称，必填字段。

        返回:
        - 返回搭配禁忌事项，或者返回"空"。
        """
        _event1 = "判断食材是否有禁忌"
        _event2 = "食材搭配禁忌事项"
        # 获取食材名称参数，校验是否为空

        ingredient_name = kwargs.get('ingredient_name')
        if not ingredient_name:
            raise ValueError("食材名称不能为空")

        # 模型参数
        model_args = kwargs.get("model_args", {})
        stream = model_args.get("stream", False)

        # 判断是否存在搭配禁忌的提示词
        taboo_check_prompt = jiahe_ingredient_taboo_existence_prompt.format(ingredient_name=ingredient_name)

        # 日志记录输入提示词
        logger.debug(f"AIGC JIAHE Functions {_event1} LLM Input: {(taboo_check_prompt)}")

        # 记录开始时间
        start_time = time.time()

        # 调用大模型判断是否存在搭配禁忌
        has_taboo_pairing: Union[str, Generator] = await acallLLM(
            history=[{"role": "user", "content": taboo_check_prompt}],
            max_tokens=2048,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=stream,
            model="Qwen1.5-32B-Chat",
        )

        # 记录模型响应时间
        logger.info(f"AIGC JIAHE Functions {_event1} LLM Output: {(has_taboo_pairing)}")

        # 记录模型响应时间
        logger.debug(f"{_event1}模型输出时间: {time.time() - start_time}秒")

        # 解析判断结果
        has_taboo_pairing_dict = await parse_generic_content(has_taboo_pairing)

        # 判断是否有禁忌
        if has_taboo_pairing_dict.get("has_taboo_pairing") == "是":
            # 如果存在禁忌，生成详细的搭配禁忌信息
            taboo_details_prompt = jiahe_ingredient_taboo_details_prompt.format(ingredient_name=ingredient_name)

            # 日志记录详细禁忌提示词
            logger.debug(f"AIGC JIAHE Functions {_event1} LLM Input: {(taboo_details_prompt)}")

            # 记录生成禁忌搭配详情开始时间
            start_time = time.time()

            # 调用大模型获取禁忌搭配详情
            taboo_details: Union[str, Generator] = await acallLLM(
                history=[{"role": "user", "content": taboo_details_prompt}],
                max_tokens=2048,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                stream=stream,
                model="Qwen1.5-32B-Chat",
            )

            logger.info(f"AIGC JIAHE Functions {_event2} LLM Output: {(taboo_details)}")

            logger.debug(f"{_event2}模型输出时间: {time.time() - start_time}秒")

            # 解析禁忌搭配详情
            taboo_details_content = await parse_generic_content(taboo_details)

            return taboo_details_content

        # 没有禁忌时返回"空"
        else:
            return None


    async def aigc_jiahe_ingredient_selection_guide(self, **kwargs):
        """
        根据食材名称生成挑选方法，分别从外观、手感和气味等方面提供详细挑选方法

        需求文档地址：: https://your-document-link.com

        参数:
        - ingredient_name (str): 食材名称，用户输入的必填项。

        返回:
        - 生成关于该食材挑选方法的文本，包括外观、手感和气味方面的详细说明。
        """

        # 校验必填项
        ingredient_name = kwargs.get('ingredient_name', '')
        if not ingredient_name:
            raise ValueError("食材名称为必填项，不能为空！")

        # 拼接提示词
        model_args = kwargs.get("model_args", {})
        stream = model_args.get("stream", False)

        messages = [
            {
                "role": "user",
                "content": ingredient_selection_prompt.format(
                    ingredient_name=ingredient_name
                ),
            }
        ]

        # 日志记录模型输入
        logger.debug("食材挑选方法模型输入：" + repr(messages[0]['content']))

        start_time = time.time()

        # 调用大语言模型生成挑选方法建议
        generate_text: Union[str, Generator] = await acallLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-32B-Chat",  # 指定使用的模型
        )

        # 日志记录模型响应延迟和结果
        logger.debug("食材挑选方法模型输出latency： " + str(time.time() - start_time))

        # 解析并返回生成的结果
        content = generate_text
        return content


    async def aigc_jiahe_ingredient_taboo_groups(self, **kwargs):
        """
        输入食材名称，智能助手告知该食材的禁忌人群，如特定疾病患者、特殊体质等，并提供替代食材建议，确保饮食既健康又安全。

        需求文档地址: https://your-document-link.com

        参数:
        - ingredient_name (str): 食材名称，用户输入的必填项。

        返回:
        - 生成关于该食材禁忌人群的文本，列出禁忌人群及替代食材建议。
        """

        # 校验必填项
        ingredient_name = kwargs.get('ingredient_name', '')
        if not ingredient_name:
            raise ValueError("食材名称为必填项，不能为空！")

        # 拼接提示词
        model_args = kwargs.get("model_args", {})
        stream = model_args.get("stream", False)

        messages = [
            {
                "role": "user",
                "content": ingredient_taboo_groups_prompt.format(
                    ingredient_name=ingredient_name
                ),
            }
        ]

        # 日志记录模型输入
        logger.debug("食材禁忌人群模型输入：" + repr(messages[0]['content']))

        start_time = time.time()

        # 调用大语言模型生成禁忌人群说明
        generate_text: Union[str, Generator] = await acallLLM(
            history=messages,
            max_tokens=1024,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            stream=True,
            model="Qwen1.5-72B-Chat",  # 指定使用的模型
        )

        # 日志记录模型响应延迟和结果
        logger.debug("食材禁忌人群模型输出latency： " + str(time.time() - start_time))

        # 解析并返回生成的结果
        content = generate_text
        return content





