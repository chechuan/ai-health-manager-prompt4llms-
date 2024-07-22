import json
import time
from string import Template
from data.jiahe_prompt import *
from data.jiahe_util import *

from src.pkgs.models.utils import ParamTools
from src.prompt.model_init import ChatMessage, acallLLM, callLLM
from src.utils.api_protocal import *
from src.utils.Logger import logger
from src.utils.module import (
    InitAllResource,
)

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
            model="Qwen1.5-32B-Chat",
        )
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("健康吃知识问答模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_diet_principle(cur_date, location, history=[], userInfo={}):
        """出具饮食调理原则"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        messages = [
            {
                "role": "user",
                "content": jiahe_daily_diet_principle_prompt.format(
                    userInfo, cur_date, location, his_prompt
                ),
            }
        ]
        logger.debug(
            "出具饮食调理原则模型输入： " + json.dumps(messages, ensure_ascii=False)
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
        logger.debug(
            "出具家庭饮食原则模型输入： " + json.dumps(messages, ensure_ascii=False)
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
        print(f"latency {time.time() - start_time:.2f} s -> response")
        logger.debug("出具家庭饮食原则模型输出： " + generate_text)
        yield {"message": generate_text, "end": True}

    @staticmethod
    async def gen_family_diet(
            users,
            cur_date,
            location,
            family_principle,
            history=[],
            requirements=[],
            reference_diet="",
            days=1,
    ):
        """出具家庭N日饮食计划"""
        roles, familyInfo, his_prompt = get_familyInfo_history(users, history)
        temp = Template(jiahe_family_diet_prompt)
        diet_cont = []
        if reference_diet:
            diet_cont.extend(reference_diet)
        days = 1
        for i in range(days):
            # cur_date = (datetime.datetime.now() + datetime.timedelta(days=+i)).strftime("%Y-%m-%d")
            ref_diet_str = "\n".join(diet_cont[-2:])

            prompt = temp.substitute(
                num=len(users),
                roles=roles,
                requirements="，".join(requirements),
                family_info=familyInfo,
                cur_date=cur_date,
                location=location,
                family_principle=family_principle,
                reference_diet=ref_diet_str,
                days="1天",
            )
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            logger.debug(
                "出具家庭一日饮食计划模型输入： "
                + json.dumps(messages, ensure_ascii=False)
            )
            start_time = time.time()
            generate_text = await acallLLM(
                history=messages,
                max_tokens=1024,
                top_p=0.9,
                temperature=0.8,
                do_sample=True,
                # stream=True,
                model="Qwen1.5-72B-Chat",
            )
            diet_cont.append(generate_text)
            print(f"latency {time.time() - start_time:.2f} s -> response")
            logger.debug("出具家庭一日饮食计划模型输出： " + generate_text)
            yield {"message": generate_text, "end": True}

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
            model="Qwen1.5-32B-Chat",
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
            reference_daily_diets,
            days,
            history=[],
            userInfo={},
    ):
        """个人N日饮食计划"""
        userInfo, his_prompt = get_userInfo_history(userInfo, history)
        diet_cont = []
        if reference_daily_diets:
            diet_cont.extend(reference_daily_diets)
        import datetime

        for i in range(days):
            cur_date = (datetime.datetime.now() + datetime.timedelta(days=+i)).strftime(
                "%Y-%m-%d"
            )
            # 生成一日食谱
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
                    ),
                }
            ]
            logger.debug(
                "一日饮食计划模型输入： " + json.dumps(messages, ensure_ascii=False)
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
            logger.info("一日饮食计划模型生成时间：" + str(time.time() - start_time))
            diet_cont.append(generate_text)
            yield {"message": generate_text, "end": False}

            # logger.debug(
            #     "一日饮食计划模型输出： " + generate_text
            # )
            # messages = [
            #     {
            #         "role": "user",
            #         "content": jiahe_physical_efficacy_prompt.format(generate_text),
            #     }
            # ]
            # logger.debug(
            #     "一日食物功效模型输入： " + json.dumps(messages, ensure_ascii=False)
            # )
            # start_time = time.time()
            # generate_text = callLLM(
            #     history=messages,
            #     max_tokens=2048,
            #     top_p=0.9,
            #     temperature=0.8,
            #     do_sample=True,
            #     stream=True,
            #     model="Qwen1.5-72B-Chat",
            # )

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
        #     logger.debug("一日食谱模型输出： " + content)
        #     diet_cont.append(content)
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

    # @staticmethod
    # async def guess_asking_child_userInfo():

    @staticmethod
    async def gen_child_diet_principle(userInfo={}):
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
        generate_text = callLLM(
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