# -*- encoding: utf-8 -*-
"""
@Time    :   2023-11-14 17:38:45
@desc    :   server
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import re
import sys
import time
import json
import asyncio
import traceback
from typing import AnyStr, AsyncGenerator, Dict, Generator, List, Tuple, Union
from fastapi.responses import StreamingResponse
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Response, Request, APIRouter

sys.path.append(str(Path(__file__).parent.parent.absolute()))


from chat.qwen_chat import Chat
from src.pkgs.models.small_expert_model import expertModel, Agents
from src.pkgs.pipeline import Chat_v2
from src.utils.api_protocal import (
    AigcFunctionsResponse,
    RolePlayRequest,
    AigcFunctionsRequest,
)
from src.utils.Logger import logger
from src.utils.module import (
    check_aigc_request,
    InitAllResource,
    NpEncoder,
    curr_time,
    dumpJS,
    format_sse_chat_complete,
    response_generator,
)


async def accept_param(request: Request):
    p = await request.json()
    backend_history = p.get("backend_history", [])
    p["backend_history"] = (
        json.loads(backend_history)
        if isinstance(backend_history, str)
        else backend_history
    )
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


def accept_param_purge(request: Request):
    p = request.json()
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


async def async_accept_param_purge(request: Request):
    p = await request.json()
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


def make_result(
    head=200, msg=None, items=None, ret_response=True, **kwargs
) -> Union[Response, StreamingResponse]:
    if not isinstance(items, AsyncGenerator):
        if not items and head == 200:
            head = 600
        res = {"head": head, "msg": msg, "items": items, **kwargs}
        res = json.dumps(res, cls=NpEncoder, ensure_ascii=False)
        if ret_response:
            return Response(
                res, media_type=kwargs.get("media_type", "application/json")
            )
        else:
            return res
    else:
        return StreamingResponse(items, media_type="text/event-stream")


def make_stream_result(): ...


def yield_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head": head, "message": msg, "items": items, **kwargs, "end": True}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    yield res


def mount_rule_endpoints(app: FastAPI):
    @app.route("/rules/blood_pressure_level", methods=["post"])
    async def _rules_blood_pressure_level(request: Request):
        """计算血压等级及处理规则"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.tool_rules_blood_pressure_level(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rules/emotions", methods=["post"])
    async def _rules_enotions_level(request: Request):
        """情志分级"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.emotions(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rules/weight_trend", methods=["post"])
    async def _rules_weight_trend(request: Request):
        """体重趋势"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.weight_trend(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rules/fat_reduction", methods=["post"])
    async def _rules_fat_reduction(request: Request):
        """体重减脂"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.fat_reduction(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/blood_pressure_trend_analysis", methods=["post"])
    async def _health_blood_pressure_trend_analysis(request: Request):
        """血压趋势分析"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.health_blood_pressure_trend_analysis(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/warning_solutions_early", methods=["post"])
    async def _health_warning_solutions_early(request: Request):
        """预警解决方案"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.health_warning_solutions_early(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret


def mount_rec_endpoints(app: FastAPI):

    @app.route("/rec/diet/food_purchasing_list/manage", methods=["post"])
    async def _rec_diet_create_food_purchasing_list_manage(request: Request):
        """食材采购清单管理"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.food_purchasing_list_manage(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rec/diet/food_purchasing_list/generate_by_content", methods=["post"])
    async def _rec_diet_create_food_purchasing_list_generate_by_content(
        request: Request,
    ):
        """食材采购清单管理"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.food_purchasing_list_generate_by_content(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rec/diet/reunion_meals/restaurant_selection", methods=["post"])
    async def _rec_diet_reunion_meals_restaurant_selection(request: Request):
        """年夜饭, 结合群组对话和餐厅信息选择偏好餐厅"""

        async def decorate_text_stream(generator):
            while True:
                yield_item = next(generator)
                yield format_sse_chat_complete(
                    json.dumps(yield_item, ensure_ascii=False), "delta"
                )
                if yield_item["end"] is True:
                    break

        try:
            param = await async_accept_param_purge(request)
            resp = expert_model.rec_diet_reunion_meals_restaurant_selection(**param)
            generator = decorate_text_stream(resp)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return StreamingResponse(generator, media_type="text/event-stream")

    @app.route("/rec/diet/evaluation", methods=["post"])
    async def _rec_diet_evaluation(request: Request):
        """膳食摄入评估"""
        try:
            param = await async_accept_param_purge(request)
            ret = expert_model.rec_diet_eval(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret


def mount_aigc_functions(app: FastAPI):
    """挂载aigc函数"""

    @app.route("/aigc/functions", methods=["post"])
    @app.route("/aigc/functions/consultation_summary", methods=["post"])
    @app.route("/aigc/functions/diagnosis", methods=["post"])
    @app.route("/aigc/functions/reason_for_care_plan", methods=["post"])
    @app.route("/aigc/functions/drug_recommendation", methods=["post"])
    @app.route("/aigc/functions/food_principle", methods=["post"])
    @app.route("/aigc/functions/sport_principle", methods=["post"])
    @app.route("/aigc/functions/chinese_therapy", methods=["post"])
    @app.route("/aigc/functions/mental_principle", methods=["post"])
    async def _async_aigc_functions(request: AigcFunctionsRequest) -> Response:
        """aigc函数"""
        try:
            param = await async_accept_param_purge(request)
            err_check_ret = await check_aigc_request(param)
            if err_check_ret is not None:
                raise AssertionError(err_check_ret)
            ret = await agents.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream"):
                ret: AsyncGenerator = response_generator(ret)
            ret = make_result(items=ret)
        except Exception as err:
            logger.error(err)
            ret = make_result(head=601, msg=err.args[0])
        finally:
            return ret


def create_app():
    app = FastAPI(
        title="智能健康管家-算法",
        description="",
        version="0.0.0",
    )
    router = APIRouter()
    prepare_for_all()

    def decorate_chat_complete(
        generator, return_mid_vars=False, return_backend_history=False
    ):
        try:
            while True:
                yield_item = next(generator)
                yield_item["data"]["appendData"] = yield_item["appendData"]
                item = {**yield_item["data"]}
                logger.info(
                    "Output (except mid_vars & backend_history):\n"
                    + json.dumps(item, ensure_ascii=False)
                )
                if return_mid_vars:
                    if item["end"] is True:
                        item["mid_vars"] = yield_item["mid_vars"]
                    else:
                        item["mid_vars"] = []
                if return_backend_history:
                    if item["end"] is True:
                        item["backend_history"] = yield_item["history"]
                    else:
                        item["backend_history"] = []
                yield format_sse_chat_complete(
                    json.dumps(item, ensure_ascii=False), "delta"
                )
                if yield_item["data"]["end"] == True:
                    break
        except Exception as err:
            logger.exception(err)
            item = make_result(head=600, message=repr(err), end=True)
            yield format_sse_chat_complete(
                json.dumps(item, ensure_ascii=False), "delta"
            )

    @app.route("/chat_gen", methods=["post"])
    async def get_chat_gen(request: Request):
        global chat

        try:
            param = await accept_param(request)
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"),
                mid_vars=[],
                use_sys_prompt=False,
                **param,
            )
            result = decorate_chat_complete(
                generator, return_mid_vars=False, return_backend_history=True
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/chat/complete", methods=["post"])
    async def _chat_complete_stream_midvars(request: Request):
        """demo,主要用于展示返回的中间变量"""
        try:
            param = await accept_param(request)
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"),
                mid_vars=[],
                use_sys_prompt=True,
                **param,
            )
            result = decorate_chat_complete(
                generator, return_mid_vars=True, return_backend_history=True
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/chat/role_play", methods=["post"])
    async def _chat_role_play(request: RolePlayRequest):
        """角色扮演对话"""
        try:
            param = await accept_param(request)
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"),
                mid_vars=[],
                use_sys_prompt=True,
                **param,
            )
            result = decorate_chat_complete(
                generator, return_mid_vars=True, return_backend_history=True
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/intent/query", methods=["post"])
    async def intent_query(request: Request):
        global chat
        try:
            param = await accept_param(request)
            item = chat.intent_query(
                param.get("history", []),
                task=param.get("task", ""),
                prompt=param.get("prompt", ""),
                userInfo=param.get("promptParam", ""),
                intentPrompt=param.get("intentPrompt", ""),
                subIntentPrompt=param.get("subIntentPrmopt", ""),
                scene_code=param.get("scene_code", "default"),
            )
            result = make_result(items=item)
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            return Response(dumpJS(result), content_type="application/json")

    @app.route("/reload_prompt", methods=["get"])
    async def _reload_prompt(request: Request):
        """重启chat实例"""
        global chat
        try:
            prepare_for_all()
            ret = {"head": 200, "success": True, "msg": "restart success"}
        except Exception as err:
            logger.exception(err)
            ret = {"head": 500, "success": False, "msg": repr(err)}
        finally:
            return Response(dumpJS(ret), media_type="application/json")

    @app.route("/fetch_intent_code", methods=["get"])
    async def _fetch_intent_code(request: Request):
        """获取意图代码"""
        global chat
        try:
            ret = chat.fetch_intent_code()
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/search/duckduckgo", methods=["post"])
    async def _search_duckduckgo(request: Request):
        """DuckDuckGo搜索"""
        try:
            param = await async_accept_param_purge(request)
            ret = chat_v2.funcall.search_qa_chain.ddg_search_chain.call(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/search/crawler/sougou", methods=["post"])
    async def _search_crawler_sougou(request: Request):
        """爬虫 - 搜狗搜索"""
        try:
            param = await async_accept_param_purge(request)
            ret = chat_v2.funcall.call_search_engine(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/test/sync", methods=["post"])
    async def _test_sync(request: Request) -> Response:
        """异步测试"""
        p = await async_accept_param_purge(request)
        t1 = curr_time()
        await asyncio.sleep(2)
        ret = {"start": t1, "end": curr_time()}
        logger.debug(ret)
        return Response(dumpJS(ret), media_type="application/json")

    mount_aigc_functions(app)
    mount_rule_endpoints(app)
    mount_rec_endpoints(app)
    app.include_router(router)
    return app


def prepare_for_all():
    global chat
    global args
    global chat_v2
    global expert_model
    global gsr
    global agents

    gsr = InitAllResource()
    args = gsr.args
    chat = Chat(gsr)
    chat_v2 = Chat_v2(gsr)
    expert_model = expertModel(gsr)
    agents = Agents(gsr)


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")
    # uvicorn.run("src.server:app", host=args.ip, port=args.port, log_level="info")
