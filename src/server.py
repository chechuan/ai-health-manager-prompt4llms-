# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-20 17:17:57
@desc    :   健康模块功能实现
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

# 标准库导入
import asyncio, json, sys, traceback
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Union

# 第三方库导入
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

# 本地模块导入
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from chat.qwen_chat import Chat
from src.pkgs.models.agents import Agents
from src.pkgs.models.expert_model import expertModel
from src.pkgs.models.jiahe_expert_model import JiaheExpertModel
from src.pkgs.models.health_expert_model import HealthExpertModel
from src.pkgs.models.itinerary_model import ItineraryGenerator
from src.pkgs.pipeline import Chat_v2
from src.utils.api_protocal import (
    AigcFunctionsCompletionResponse, AigcFunctionsDoctorRecommendRequest, AigcFunctionsRequest,
    AigcFunctionsResponse, AigcSanjiRequest, BaseResponse, BodyFatWeightManagementRequest,
    OutpatientSupportRequest, RolePlayRequest, SanJiKangYangRequest, TestRequest
)
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    MakeFastAPIOffline, NpEncoder, build_aigc_functions_response, curr_time,
    dumpJS, format_sse_chat_complete, response_generator
)


async def accept_param(request: Request, endpoint: str = None):
    p = await request.json()
    backend_history = p.get("backend_history", [])
    p["backend_history"] = (
        json.loads(backend_history)
        if isinstance(backend_history, str)
        else backend_history
    )
    p_jsonfiy = json.dumps(p, ensure_ascii=False)
    endpoint = endpoint if endpoint else "Undefined"
    logger.info(f"Endpoint: {endpoint}, Input Param: {p_jsonfiy}")
    return p


def accept_param_purge(request: Request):
    if isinstance(request, BaseModel):
        p = request.model_dump()
    elif isinstance(request, Request):
        p = request.json()
    p_jsonfiy = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {p_jsonfiy}")
    return p


async def async_accept_param_purge(request: Request, endpoint: str = None):
    if isinstance(request, BaseModel):
        p = request.model_dump(exclude_unset=True, exclude_none=True)
    else:
        p = await request.json()
    p_jsonfiy = json.dumps(p, ensure_ascii=False)
    endpoint = endpoint if endpoint else "Undefined"
    logger.info(f"Endpoint: {endpoint}, Input Param: {p_jsonfiy}")
    return p


def make_result(
    head=200, msg=None, items=None, ret_response=True, **kwargs
) -> Union[Response, StreamingResponse]:
    if not items and head == 200:
        head = 600
    res = {"head": head, "msg": msg, "items": items, **kwargs}
    res = json.dumps(res, cls=NpEncoder, ensure_ascii=False)
    if ret_response:
        return Response(res, media_type=kwargs.get("media_type", "application/json"))
    else:
        return res


def make_stream_result(): ...


def yield_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head": head, "message": msg, "items": items, **kwargs, "end": True}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    yield res


def mount_rule_endpoints(app: FastAPI):
    # @app.route("/rules/blood_pressure_level", methods=["post"])
    # async def _rules_blood_pressure_level(request: Request):
    #     """计算血压等级及处理规则"""
    #     try:
    #         param = await async_accept_param_purge(
    #             request, endpoint="/rules/blood_pressure_level"
    #         )
    #         ret = expert_model.tool_rules_blood_pressure_level(**param)
    #         ret = make_result(items=ret)
    #     except Exception as err:
    #         logger.exception(err)
    #         ret = make_result(head=500, msg=repr(err))
    #     finally:
    #         return ret

    @app.route("/rules/emotions", methods=["post"])
    async def _rules_enotions_level(request: Request):
        """情志分级"""
        try:
            param = await async_accept_param_purge(request, endpoint="/rules/emotions")
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
            param = await async_accept_param_purge(
                request, endpoint="/rules/weight_trend"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/rules/fat_reduction"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_pressure_trend_analysis"
            )
            ret = expert_model.health_blood_pressure_trend_analysis(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/blood_glucose_trend_analysis", methods=["post"])
    async def _health_blood_glucose_trend_analysis(request: Request):
        """血糖趋势分析"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_glucose_trend_analysis"
            )
            ret = expert_model.health_blood_glucose_trend_analysis(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/key_extraction", methods=["post"])
    async def _key_extraction(request: Request):
        """关键词抽取"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/key_extraction"
            )
            ret = expert_model.health_key_extraction(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/literature_interact", methods=["post"])
    async def _key_extraction(request: Request):
        """文献1"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/literature_interact"
            )
            ret = expert_model.health_literature_interact(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/literature_generation", methods=["post"])
    async def _literature_generation(request: Request):
        """文献2"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/literature_generation"
            )
            ret = await expert_model.health_literature_generation(param)
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
            param = await async_accept_param_purge(
                request, endpoint="/health/warning_solutions_early"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/rec/diet/food_purchasing_list/manage"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/rec/diet/food_purchasing_list/generate_by_content"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/rec/diet/reunion_meals/restaurant_selection"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/rec/diet/evaluation"
            )
            ret = expert_model.rec_diet_eval(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret


def mount_aigc_functions(app: FastAPI):
    """挂载aigc函数"""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # 提取并格式化错误信息
        errors = exc.errors()
        return JSONResponse(
            status_code=500,
            content={"head": 500, "items": None, "msg": errors},
        )

    async def _async_aigc_functions(
        request_model: AigcFunctionsRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """aigc函数"""
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/functions"
            )
            response: Union[str, AsyncGenerator] = await agents.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=True)
        finally:
            return build_aigc_functions_response(_return)

    async def _async_aigc_sanji(
        request_model: AigcSanjiRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """aigc函数"""
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/sanji"
            )
            response: Union[str, AsyncGenerator] = await agents.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=True)
        finally:
            return build_aigc_functions_response(_return)

    async def _async_aigc_functions_doctor_recommend(
        request_model: AigcFunctionsDoctorRecommendRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """aigc函数"""
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/functions/doctor_recommend"
            )
            response: Union[str, AsyncGenerator] = await agents.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(response)
            else:
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=False)
        finally:
            return build_aigc_functions_response(_return)

    async def _aigc_functions_outpatient_support(
        request_model: OutpatientSupportRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """处理西医决策支持的aigc函数"""
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/outpatient_support"
            )
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=True)
        finally:
            return build_aigc_functions_response(_return)

    async def _async_aigc_functions_sanji_kangyang(
        request_model: SanJiKangYangRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """三济康养方案的AIGC函数"""
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/sanji/kangyang"
            )
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=True)
        finally:
            return build_aigc_functions_response(_return)

    async def _aigc_functions_generate_itinerary(request: Request):
        # 这里调用生成行程的逻辑
        user_data = await request.json()
        generator = ItineraryGenerator()
        response = generator.generate(user_data)
        return response

    async def _aigc_functions_generate_bath_plan(request: Request):
        user_data = await request.json()
        # intentcode = user_data.get("intentcode_bath_plan", "default_bath_code")
        generator = ItineraryGenerator()
        response = generator.generate_bath_plan(user_data)
        return response

    app.post("/aigc/functions", description="AIGC函数")(_async_aigc_functions)

    app.post("/aigc/functions/doctor_recommend")(_async_aigc_functions_doctor_recommend)

    app.post("/aigc/sanji")(_async_aigc_sanji)

    app.post("/aigc/outpatient_support")(_aigc_functions_outpatient_support)

    app.post("/aigc/sanji/kangyang")(_async_aigc_functions_sanji_kangyang)


    app.post("/aigc/itinerary/make", description="根据用户的偏好和需求生成个性化行程清单")(_aigc_functions_generate_itinerary)

    app.post("/aigc/bath_plan/make", description="生成泡浴方案")(_aigc_functions_generate_bath_plan)



def create_app():
    app: FastAPI = FastAPI(
        title="智能健康管家-算法",
        description="",
        version=f"{datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
    )
    prepare_for_all()

    async def document():  # 用于展示接口文档
        return RedirectResponse(url="/docs")

    # @app.exception_handler(ValidationError)
    # async def validation_exception_handler(request: Request, exc: ValidationError):
    #     error_messages = []
    #     for error in exc.errors():
    #         error_messages.append({'loc': error['loc'], 'msg': error['msg']})
    #     return JSONResponse(content={'success': False, 'errors': error_messages}, status_code=422)

    async def decorate_chat_complete(
        generator, return_mid_vars=False, return_backend_history=False
    ) -> AsyncGenerator:
        try:
            # while True:
            async for yield_item in generator:
                # yield_item = await next(generator)
                yield_item["data"]["appendData"] = yield_item["appendData"]
                item = {**yield_item["data"]}
                logger.info(
                    "Output (except mid_vars & backend_history):"
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

    async def decorate_general_complete(
            generator
    ) -> AsyncGenerator:
        try:
            # while True:
            async for yield_item in generator:
                # yield_item = await next(generator)
                item = {**yield_item}
                logger.info(
                    "Output (except mid_vars & backend_history):"
                    + json.dumps(item, ensure_ascii=False)
                )
                yield format_sse_chat_complete(
                    json.dumps(item, ensure_ascii=False), "delta"
                )
                if yield_item["end"] == True:
                    break
        except Exception as err:
            logger.exception(err)
            item = make_result(head=600, message=repr(err), end=True)
            yield format_sse_chat_complete(
                json.dumps(item, ensure_ascii=False), "delta"
            )

    async def _test_sync(request_model: TestRequest) -> JSONResponse:
        """异步测试response_model=BaseResponse"""
        p = accept_param_purge(request_model)
        t1 = curr_time()
        await asyncio.sleep(2)
        ret = {"start": t1, "end": curr_time()}
        logger.debug(ret)
        return JSONResponse(ret, media_type="application/json")

    app.get("/", response_model=BaseResponse, summary="swagger 文档")(document)

    app.post("/test/sync")(_test_sync)

    @app.route("/health_qa", methods=["post"])
    async def _health_qa(request: Request):
        """健康知识问答"""
        try:
            param = await accept_param(request, endpoint="/health_qa")
            generator: AsyncGenerator = JiaheExpertModel.eat_health_qa(param.get("query", ""))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_userInfo_question", methods=["post"])
    async def _gen_userInfo_question(request: Request):
        """生成收集用户信息问题"""
        try:
            param = await accept_param(request, endpoint="/gen_userInfo_question")
            generator: AsyncGenerator = JiaheExpertModel.gather_userInfo(param.get('userInfo', {}), param.get('history', []))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_diet_principle", methods=["post"])
    async def _gen_diet_principle(request: Request):
        """饮食调理原则接口"""
        try:
            param = await accept_param(request, endpoint="/gen_diet_principle")
            generator: AsyncGenerator = JiaheExpertModel.gen_diet_principle(param.get('cur_date', ''),
                                                                       param.get('location', ''),
                                                                       param.get('history', []),
                                                                       param.get('userInfo', {}))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/child_diet_principle", methods=["post"])
    async def _gen_child_diet_principle(request: Request):
        """儿童饮食原则接口"""
        try:
            param = await accept_param(request, endpoint="/child_diet_principle")
            generator: AsyncGenerator = JiaheExpertModel.gen_child_diet_principle(param.get('userInfo', {}))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/child_nutrient_effect", methods=["post"])
    async def _gen_child_nutrient_effect(request: Request):
        """儿童饮食原则接口"""
        try:
            param = await accept_param(request, endpoint="/child_nutrient_effect")
            generator: AsyncGenerator = JiaheExpertModel.gen_child_nutrious_effect(param.get('userInfo', {}),
                                                                                   param.get('cur_date', ''),
                                                                                   param.get('location', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/child_dish_rec", methods=["post"])
    async def _gen_child_dish_rec(request: Request):
        """儿童饮食推荐接口"""
        try:
            param = await accept_param(request, endpoint="/child_dish_rec")
            generator: AsyncGenerator = JiaheExpertModel.child_dish_rec(param.get('userInfo', {}),
                                                                                   param.get('cur_date', ''),
                                                                                   param.get('location', ''),
                                                                                   param.get('history_dishes', ''),
                                                                                   param.get('child_diet_principle', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_embedding", methods=["post"])
    async def _gen_embedding(request: Request):
        try:
            param = await accept_param(request, endpoint="/gen_embedding")
            # generator: AsyncGenerator = JiaheExpertModel.call_embedding(param.get('inouts', []))
            item = JiaheExpertModel.call_embedding(param.get('inputs', []))
            result = make_result(items=item)
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)

        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)

        finally:
            return result

    @app.route("/family_diet_principle", methods=["post"])
    async def _gen_diet_principle(request: Request):
        """家庭饮食推荐原则接口"""
        try:
            param = await accept_param(request, endpoint="/family_diet_principle")
            generator: AsyncGenerator = JiaheExpertModel.gen_family_principle(param.get('users', ''),
                                                                         param.get('cur_date', ''),
                                                                         param.get('location', ''),
                                                                         param.get('history', []),
                                                                         param.get('requirements', []))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/family_daily_diet", methods=["post"])
    async def _gen_family_daily_diet(request: Request):
        """家庭N日饮食计划接口"""
        try:
            param = await accept_param(request, endpoint="/family_daily_diet")
            generator: AsyncGenerator = JiaheExpertModel.gen_family_diet(param.get('users', ''),
                                                                    param.get('cur_date', ''),
                                                                    param.get('location', ''),
                                                                    param.get('family_diet_principle', ''),
                                                                    param.get('history', []),
                                                                    param.get('requirements', []),
                                                                    param.get('reference_diet', []),
                                                                    param.get('days', 1))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/long_term_nutritional_management", methods=["post"])
    async def _gen_long_term_nutritional_management_recognition(request: Request):
        """长期营养管理识别"""
        try:
            param = await accept_param(request, endpoint="/long_term_nutritional_management")
            generator: AsyncGenerator = JiaheExpertModel.long_term_nutritional_management(param.get('userInfo', []),
                                                                    param.get('history', []))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/nutritious_supplementary_principle", methods=["post"])
    async def _gen_diet_principle(request: Request):
        """营养素推荐原则接口"""
        try:
            param = await accept_param(request, endpoint="/nutritious_supplementary_principle")
            generator: AsyncGenerator = JiaheExpertModel.gen_nutrious_principle(param.get('cur_date', ''),
                                                                           param.get('location', ''),
                                                                           param.get('history', []),
                                                                           param.get('userInfo', []))
            result = decorate_general_complete(
                generator
            )

        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/nutritious_supplementary", methods=["post"])
    async def _gen_nutritious_supplementary(request: Request):
        """营养素计划接口"""
        try:
            param = await accept_param(request, endpoint="/nutritious_supplementary")
            generator: AsyncGenerator = JiaheExpertModel.gen_nutrious(param.get('cur_date', ''),
                                                                 param.get('location', ''),
                                                                 param.get('nutrious_principle', ''),
                                                                 param.get('history', []),
                                                                 param.get('userInfo', []),
                                                                 )
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_daily_diet", methods=["post"])
    async def _gen_daily_diet(request: Request):
        """个人N日饮食计划接口"""
        try:
            param = await accept_param(request, endpoint="/gen_daily_diet")
            generator: AsyncGenerator = JiaheExpertModel.gen_n_daily_diet(param.get('cur_date', ''),
                                                                     param.get('location', ''),
                                                                     param.get('diet_principle', ''),
                                                                     param.get('reference_daily_diets', []),
                                                                     param.get('days', 0),
                                                                     param.get('history', []),
                                                                     param.get('userInfo', {}))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_current_diet", methods=["post"])
    async def _gen_current_diet(request: Request):
        """生成当餐食谱接口"""
        try:
            param = await accept_param(request, endpoint="/gen_current_diet")
            generator: AsyncGenerator = JiaheExpertModel.gen_current_diet(param.get('cur_date', ''),
                                                                     param.get('location', ''),
                                                                     param.get('diet_principle', ''),
                                                                     param.get('reference_daily_diets', []),
                                                                    param.get('meal_number', ''),
                                                                     param.get('history', []),
                                                                     param.get('userInfo', {}),
                                                                          param.get('today_diet', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/gen_diet_effect", methods=["post"])
    async def _gen_diet_effect(request: Request):
        """饮食功效接口"""
        try:
            param = await accept_param(request, endpoint="/gen_daily_diet")
            generator: AsyncGenerator = JiaheExpertModel.gen_diet_effect(param.get('diet', ''))
            result = decorate_general_complete(
                generator

            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/guess_asking_nutrition_counseling", methods=["post"])
    async def _guess_asking_intent(request: Request):
        """猜你想问-饮食咨询接口"""
        try:
            param = await accept_param(request, endpoint="/guess_asking_nutrition_counseling")
            generator: AsyncGenerator = JiaheExpertModel.gen_guess_asking(param.get('userInfo', {}), scene_flag='intent')
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/guess_asking_query", methods=["post"])
    async def _guess_asking_query(request: Request):
        """猜你想问-用户问题接口"""
        try:
            param = await accept_param(request, endpoint="/guess_asking_query")
            generator: AsyncGenerator = JiaheExpertModel.gen_guess_asking(param.get('userInfo', {}), scene_flag='user_query',
                                                                     question=param.get('query', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/guess_asking_diet", methods=["post"])
    async def _guess_asking_diet(request: Request):
        """猜你想问-食谱接口"""
        try:
            param = await accept_param(request, endpoint="/guess_asking_diet")
            generator: AsyncGenerator = JiaheExpertModel.gen_guess_asking(param.get('userInfo', {}), scene_flag='diet',
                                                                     question=param.get('diet', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/child_guess_asking", methods=["post"])
    async def _guess_asking_child(request: Request):
        """儿童猜你想问"""
        try:
            param = await accept_param(request, endpoint="/child_guess_asking")
            generator: AsyncGenerator = JiaheExpertModel.guess_asking_child(param.get('userInfo', {}), param.get('dish_efficacy', ''),
                                                                          param.get('nutrient_efficacy', ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/confirm_collect_userInfo", methods=["post"])
    async def _confirm_collect_userInfo(request: Request):
        """收集信息确认接口"""
        try:
            param = await accept_param(request, endpoint="/confirm_collect_userInfo")
            item = JiaheExpertModel.is_gather_userInfo(
                param.get("userInfo", {}), param.get("history", [])
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
            return result

    @app.route("/aigc/jiahe/food_recommendation", methods=["post"])
    async def _recommend_seasonal_ingredients(request: Request):
        """家和V930-食材/菜品推荐"""
        try:
            # 获取请求参数
            param = await async_accept_param_purge(request, endpoint="/aigc/jiahe/food_recommendation")

            # 调用大模型处理
            response: Union[str, AsyncGenerator] = await jiahe_expert.call_function(**param)
            # 判断是否需要流式输出
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 流式响应处理，构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)
            else:
                # 非流式响应，构造标准的JSON字符串
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)

        except Exception as err:
            msg = repr(err)
            # 处理错误的流式响应
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:
                # 处理错误的普通JSON响应
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
                _return: str = ret.model_dump_json(exclude_unset=True)

        finally:
            # 返回构造的响应
            return build_aigc_functions_response(_return)

    @app.route("/chat_gen", methods=["post"])
    async def get_chat_gen(request: Request):
        global chat

        try:
            param = await accept_param(request, endpoint="/chat_gen")
            generator: AsyncGenerator = chat_v2.general_yield_result(
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
    async def _chat_complete(request: Request):
        """demo,主要用于展示返回的中间变量"""
        try:
            param = await accept_param(request, endpoint="/chat/complete")
            generator = await chat_v2.general_yield_result(
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
            param = await accept_param(request, endpoint="/chat/role_play")
            generator = await chat_v2.general_yield_result(
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
            param = await accept_param(request, endpoint="/intent/query")
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
            return result

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
            param = await async_accept_param_purge(
                request, endpoint="/search/duckduckgo"
            )
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
            param = await async_accept_param_purge(
                request, endpoint="/search/crawler/sougou"
            )
            ret = chat_v2.funcall.call_search_engine(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    async def _aigc_functions_weight_management(request_model: BodyFatWeightManagementRequest):
        """体脂体重管理
        """
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/weight_management"
            )
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(response)
            else:
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return: AsyncGenerator = response_generator(msg, error=True)
            else:
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
            _return: str = ret.model_dump_json(exclude_unset=True)
        finally:
            return build_aigc_functions_response(_return)

    app.route("/aigc/weight_management", methods=["post"])(_aigc_functions_weight_management)

    mount_aigc_functions(app)
    mount_rule_endpoints(app)
    mount_rec_endpoints(app)
    MakeFastAPIOffline(app)
    return app


def prepare_for_all():
    global chat
    global args
    global chat_v2
    global expert_model
    global gsr
    global agents
    global health_expert_model
    global jiahe_expert

    gsr = InitAllResource()
    args = gsr.args
    chat = Chat(gsr)
    chat_v2 = Chat_v2(gsr)
    expert_model = expertModel(gsr)
    agents = Agents(gsr)
    health_expert_model = HealthExpertModel(gsr)
    jiahe_expert = JiaheExpertModel(gsr)


# app = create_app()

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")
    # name_app = os.path.splitext(os.path.basename(__file__))[0]
    # uvicorn.run(app=f"{name_app}:app", host=args.ip, port=args.port)
