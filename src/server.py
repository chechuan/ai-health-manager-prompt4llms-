# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-20 17:17:57
@desc    :   健康模块功能实现
@Author  :   车川
@Contact :   chechuan1204@gmail.com
"""

import asyncio
import copy
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Union, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from chat.qwen_chat import Chat
from src.pkgs.models.agents import Agents
from src.pkgs.models.expert_model import expertModel
from src.pkgs.models.jiahe_expert_model import JiaheExpertModel
from src.pkgs.models.health_expert_model import HealthExpertModel
from src.pkgs.models.itinerary_model import ItineraryModel
from src.pkgs.models.bath_plan_model import BathPlanModel
from src.pkgs.models.multimodal_model import MultiModalModel
from src.pkgs.pipeline import Chat_v2
from src.utils.api_protocal import (
    AigcFunctionsCompletionResponse,
    AigcFunctionsDoctorRecommendRequest,
    AigcFunctionsRequest,
    AigcFunctionsResponse,
    AigcSanjiRequest,
    BaseResponse,
    BodyFatWeightManagementRequest,
    OutpatientSupportRequest,
    RolePlayRequest,
    SanJiKangYangRequest,
    TestRequest,
    JunWangGongJianRequest
)
from src.utils.Logger import logger
from src.utils.resources import InitAllResource
from src.utils.module import (
    MakeFastAPIOffline,
    NpEncoder,
    build_aigc_functions_response,
    curr_time,
    dumpJS,
    format_sse_chat_complete,
    response_generator,
    replace_you,
    monitor_interface,
    call_mem0_add_memory,
    call_mem0_get_all_memories
)

from src.pkgs.models.func_eval_model.func_eval_model import (
    schedule_tips_modify,
    sport_schedule_tips_modify,
    daily_diet_eval,
    daily_diet_degree
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
        head=200, msg=None, items=None, ret_response=True, log=False, **kwargs
) -> Union[Response, StreamingResponse]:
    if not items and head == 200:
        head = 600
    res = {"head": head, "msg": msg, "items": items, **kwargs}
    if log:
        res_json = json.dumps(res, cls=NpEncoder, ensure_ascii=False)
        logger.info(f"Response content: {res_json}")
    res = json.dumps(res, cls=NpEncoder, ensure_ascii=False)
    res = res.replace("您", "你")
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


# 统一的监控记录函数
async def record_monitoring_data(items, params, start_time=None, end_time=None):
    """统一的异步记录监控数据，确保失败不影响接口返回"""
    if end_time is None:
        end_time = time.time()
    if start_time is None:
        start_time = time.time()

    # 检查 items 中的每个元素是否为 Response 对象，若是则提取其内容
    items_to_log = []
    for item in items:
        if isinstance(item, Response):
            # 如果是 Response 对象，检查其是否是 JSONResponse
            try:
                if isinstance(item, JSONResponse):
                    # 如果是 JSONResponse 类型，直接调用 .json() 提取内容
                    item_content = await item.json()
                else:
                    # 如果不是 JSONResponse 类型，直接获取内容（如字节流等）
                    item_content = str(item)  # 或者根据需要将其转换为字符串
                items_to_log.append(item_content)
            except Exception as e:
                # 捕获并记录错误，继续执行
                logger.error(f"Error extracting content from Response: {e}")
                items_to_log.append("Error extracting content")  # 记录错误信息
        else:
            items_to_log.append(item)

    # 使用 try-except 捕获监控记录中的所有错误，避免影响接口的正常返回
    try:
        response_output = json.dumps(items_to_log, ensure_ascii=False)  # 确保所有内容都是可序列化的
        await monitor_interface(
            tags=params.get("tags"),
            interface_name=params.get('endpoint_name'),  # 接口名称
            user_id=params.get("user_id"),
            session_id=params.get("session_id"),
            request_input=params,
            response_output=response_output,  # 记录所有生成器内容
            langfuse=gsr.langfuse_client,
            release="v1.0.0",
            metadata={"team": "AI", "project": "health_service"},
            start_time=start_time,  # 传递 start_time
            end_time=end_time,  # 传递 end_time
            tokenizer=gsr.qwen_tokenizer,
        )
    except Exception as e:
        # 捕获监控记录过程中的异常，避免影响接口的返回
        logger.error(f"Error recording monitoring data: {e}")


# 日志包装和流式输出逻辑
async def logging_wrapper(gen: AsyncGenerator, param, start_time=None):
    """包装生成器，记录输出到 all_items，并保持流式传递"""
    if start_time is None:
        start_time = time.time()
    all_items = []

    try:
        async for item in gen:
            # 确保 item 是 JSON 字符串
            if isinstance(item, dict):
                item_str = json.dumps(item, ensure_ascii=False)
            else:
                item_str = str(item)

            all_items.append(item_str)
            yield item
    finally:
        end_time = time.time()  # ⚠️ 在生成器真正结束时记录
        if param.get("endpoint_name") in ["daily_diet_eval"]:
            await record_monitoring_data(all_items[-1:], param, start_time, end_time)
        else:
            await record_monitoring_data(all_items, param, start_time, end_time)


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
        
    @app.route("/health/spe_qa", methods=["post"])
    async def _health_spe_qa(request: Request):
        """问题回答"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/spe_qa"
            )
            ret = await expert_model.health_spe_qa(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/blood_glucose_trend_analysis", methods=["POST"])
    async def _health_blood_glucose_trend_analysis(request: Request):
        """血糖趋势分析"""
        start_time = curr_time()
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_glucose_trend_analysis"
            )
            param['endpoint_name'] = "health_blood_glucose_trend_analysis"
            param['tags'] = ["health_blood_glucose_trend_analysis", "血糖趋势", "健康监测", "健康管理"]
            ret = expert_model.health_blood_glucose_trend_analysis(param)
            end_time = time.time()
            resp = ret
            ret = make_result(items=ret)

            await record_monitoring_data([resp], param, start_time, end_time)  # 记录到 Langfuse
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))

            end_time = time.time()
            # 记录错误监控数据
            await record_monitoring_data([ret], param, start_time, end_time)  # 记录到 Langfuse
        finally:
            return ret
        
    @app.route("/health/blood_glucose_deal", methods=["post"])
    async def _health_blood_glucose_deal(request: Request):
        """血糖预警处理"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_glucose_deal"
            )
            ret = await expert_model.health_blood_glucose_deal(param)
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

    @app.route("/health/blood_glucose_warning", methods=["post"])
    async def _health_blood_glucose_warning(request: Request):
        """血糖预警"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_glucose_warning"
            )
            ret = await expert_model.health_blood_glucose_warning(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
        
    @app.route("/health/blood_pressure_warning", methods=["post"])
    async def _health_blood_pressure_warning(request: Request):
        """血糖预警"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/blood_pressure_warning"
            )
            ret = await expert_model.health_blood_pressure_warning(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
        
    @app.route("/health/open_extract", methods=["post"])
    async def _health_open_extract(request: Request):
        """页面打开"""
        try:
            param = await async_accept_param_purge(
                request, endpoint="/health/open_extract"
            )
            ret = await expert_model.health_open_extract(param)
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

    @app.route("/aigc/functions", methods=["POST"])
    async def _async_aigc_functions(
            request_model: AigcFunctionsRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """aigc函数"""
        start_time = time.time()
        try:
            param = await async_accept_param_purge(
                request_model, endpoint="/aigc/functions"
            )
            param["endpoint_name"] = "aigc_functions"
            param["tags"] = ["aigc/functions", "AIGC", "智能服务", "人工智能"]
            response: Union[str, AsyncGenerator] = await agents.call_function(**param)

            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                _return: AsyncGenerator = response_generator(response)

                # 包装生成器，记录数据
                wrapped_generator = logging_wrapper(_return, param, start_time)
                return StreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=200, items=response
                )
                _return: str = ret.model_dump_json(exclude_unset=False)

                end_time = time.time()
                # 记录监控数据
                await record_monitoring_data([response], param, start_time, end_time)
                return build_aigc_functions_response(_return)

        except Exception as err:
            msg = repr(err)
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 流式错误响应
                _return: AsyncGenerator = response_generator(msg, error=True)
                return StreamingResponse(_return, media_type="text/event-stream")
            else:  # 处理str响应 构造json str
                ret: BaseModel = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
                _return: str = ret.model_dump_json(exclude_unset=True)

                end_time = time.time()
                # 记录错误监控数据
                await record_monitoring_data([msg], param, start_time, end_time)
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

    async def _async_aigc_functions_outpatient_support(
            request_model: OutpatientSupportRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """处理西医决策支持的AIGC函数"""
        endpoint = "/aigc/outpatient_support"
        try:
            param = await async_accept_param_purge(
                request_model, endpoint=endpoint
            )
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)

            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应
                _return = response_generator(response)
            else:
                # 非流式响应
                response = replace_you(response)  # 直接替换“您”为“你”
                if isinstance(response, str):
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
                else:
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
                logger.info(f"Endpoint: {endpoint}, Final response: {_return}")

        except Exception as err:
            # 错误处理
            msg = repr(err)
            msg = replace_you(msg)  # 错误消息统一替换
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return = response_generator(msg, error=True)  # 流式错误响应
            else:
                ret = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
                _return = ret.model_dump_json(exclude_unset=True)  # 非流式错误响应
        finally:
            return build_aigc_functions_response(_return)  # 确保返回值有赋值

    @app.route("/aigc/sanji/kangyang", methods=["POST"])
    async def _async_aigc_functions_sanji_kangyang(
            request_model: SanJiKangYangRequest,
    ) -> Response:
        """三济康养方案的AIGC函数"""
        endpoint = "/aigc/sanji/kangyang"
        endpoint_name = "sanji_kangyang"

        start_time = time.time()
        try:
            param = await async_accept_param_purge(request_model, endpoint=endpoint)
            param["endpoint_name"] = endpoint_name
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)
            # 记录监控数据
            tags = ["aigc_sanji_kangyang", "健康方案", "个性化营养", "三济康养"]

            param['tags'] = tags

            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应 构造返回数据的AsyncGenerator
                generator = response_generator(response)

                # 包装生成器，记录数据
                wrapped_generator = logging_wrapper(generator, param, start_time)
                return StreamingResponse(wrapped_generator, media_type="text/event-stream")
            else:
                # 处理非流式响应
                response = replace_you(response)  # 如果 replace_you 是异步函数，请使用 await
                if isinstance(response, str):
                    ret = AigcFunctionsCompletionResponse(head=200, items=response)
                    _return = ret.model_dump_json(exclude_unset=False)
                else:
                    ret = AigcFunctionsCompletionResponse(head=200, items=response)
                    _return = ret.model_dump_json(exclude_unset=False)

                end_time = time.time()
                logger.info(f"Endpoint: {endpoint}, Final response: {_return}")

                await record_monitoring_data([response], param, start_time, end_time)

                return build_aigc_functions_response(_return)

        except Exception as err:
            # 错误处理
            msg = replace_you(repr(err))  # 如果 replace_you 是异步函数，请使用 await
            logger.error(f"Error in {endpoint}: {msg}")

            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 流式错误响应
                generator = response_generator(msg, error=True)
                return StreamingResponse(generator, media_type="text/event-stream")
            else:
                # 非流式错误响应
                ret = AigcFunctionsCompletionResponse(head=601, msg=msg, items=None)
                _return = ret.model_dump_json(exclude_unset=True)

                end_time = time.time()
                # 记录错误监控数据
                await record_monitoring_data([msg], param, start_time, end_time)
                return build_aigc_functions_response(_return)

    async def _async_aigc_functions_jia_kang_bao_support(
            request_model: OutpatientSupportRequest,
    ) -> Union[AigcFunctionsResponse, AigcFunctionsCompletionResponse]:
        """处理西医决策支持的AIGC函数"""
        endpoint = "/aigc/jkbao"
        try:
            param = await async_accept_param_purge(
                request_model, endpoint=endpoint
            )
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)

            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 处理流式响应
                _return = response_generator(response, replace=False)
            else:
                # 非流式响应
                if isinstance(response, str):
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
                else:
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
                logger.info(f"Endpoint: {endpoint}, Final response: {_return}")

        except Exception as err:
            # 错误处理
            msg = repr(err)
            msg = replace_you(msg)  # 错误消息统一替换
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return = response_generator(msg, error=True, replace=False)  # 流式错误响应
            else:
                ret = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
                _return = ret.model_dump_json(exclude_unset=True)  # 非流式错误响应
        finally:
            return build_aigc_functions_response(_return)  # 确保返回值有赋值

    async def _async_aigc_functions_junwang_gongjian(
            request_model: JunWangGongJianRequest,
    ) -> Response:
        """
        郡网共建健康分析与建议AIGC函数的启动接口文件示例
        """

        endpoint = "/aigc/junwang/gongjian"
        try:
            # 清洗和获取参数
            param = await async_accept_param_purge(request_model, endpoint=endpoint)

            # 调用AIGC模型生成健康分析与建议
            response: Union[str, AsyncGenerator] = await itinerary_model.call_function(**param)

            # 根据参数决定是否使用流式响应
            if param.get("model_args") and param["model_args"].get("stream") is True:
                # 流式输出
                generator = response_generator(response)
                return StreamingResponse(generator, media_type="text/event-stream")
            else:
                # 非流式输出
                response = replace_you(response)  # 如果 replace_you 是异步函数，请使用 await replace_you(response)
                if isinstance(response, str):
                    ret = AigcFunctionsCompletionResponse(head=200, items=response)
                    _return = ret.model_dump_json(exclude_unset=False)
                else:
                    ret = AigcFunctionsCompletionResponse(head=200, items=response)
                    _return = ret.model_dump_json(exclude_unset=False)

                logger.info(f"Endpoint: {endpoint}, Final response: {_return}")
                return build_aigc_functions_response(_return)

        except Exception as err:
            # 异常处理
            msg = replace_you(repr(err))  # 如果 replace_you 是异步函数，请使用 await replace_you(repr(err))
            # 判断是否为流式请求
            if 'param' in locals() and param.get("model_args") and param["model_args"].get("stream") is True:
                # 流式错误响应
                generator = response_generator(msg, error=True)
                return StreamingResponse(generator, media_type="text/event-stream")
            else:
                # 非流式错误响应
                ret = AigcFunctionsCompletionResponse(head=601, msg=msg, items=None)
                _return = ret.model_dump_json(exclude_unset=True)
                return build_aigc_functions_response(_return)

    async def _aigc_functions_generate_itinerary(request: Request):
        try:
            # 记录请求参数
            user_data = await async_accept_param_purge(request, endpoint="_aigc_functions_generate_itinerary")

            # 调用生成行程的逻辑
            result = await itinerary_model.generate_itinerary(user_data)

            # 统一格式化响应
            return make_result(head=200, msg="成功生成行程", items=result, log=True)

        except Exception as e:
            # 处理异常
            logger.error(f"Error in _aigc_functions_generate_itinerary: {e}")
            return make_result(head=500, msg="生成行程失败", items=None)

    async def _aigc_functions_generate_bath_plan(request: Request):
        try:
            # 记录请求参数
            user_data = await async_accept_param_purge(request, endpoint="_aigc_functions_generate_bath_plan")

            # 调用生成浴池计划的逻辑
            result = await bath_plan_model.generate_bath_plan(user_data)

            # 统一格式化响应
            return make_result(head=200, msg="成功生成浴池计划", items=result, log=True)

        except Exception as e:
            # 处理异常
            logger.error(f"Error in _aigc_functions_generate_bath_plan: {e}")
            return make_result(head=500, msg="生成浴池计划失败", items=None)

    async def _aigc_functions_generate_itinerary_v1_1_0(request: Request):
        try:
            # 记录请求参数
            user_data = await async_accept_param_purge(request, endpoint="_aigc_functions_generate_itinerary_v1_1_0")

            # 调用生成行程的逻辑
            result = await itinerary_model.generate_itinerary_v1_1_0(user_data)

            # 统一格式化响应
            return make_result(head=200, msg="成功生成行程v1.1.0", items=result, log=True)

        except Exception as e:
            # 处理异常
            logger.error(f"Error in _aigc_functions_generate_itinerary_v1_1_0: {e}")
            return make_result(head=500, msg="生成行程v1.1.0失败", items=None)

    async def _aigc_functions_generate_bath_plan_v1_1_0(request: Request):
        try:
            # 记录请求参数
            user_data = await async_accept_param_purge(request, endpoint="_aigc_functions_generate_bath_plan_v1_1_0")

            # 调用生成浴池计划的逻辑
            result = await bath_plan_model.generate_bath_plan_v1_1_0(user_data)

            # 统一格式化响应
            return make_result(head=200, msg="成功生成浴池计划v1.1.0", items=result)

        except Exception as e:
            # 处理异常
            logger.error(f"Error in _aigc_functions_generate_bath_plan_v1_1_0: {e}")
            return make_result(head=500, msg="生成浴池计划v1.1.0失败", items=None)

    async def _aigc_functions_likang_introduction_v1_1_0(request: Request):
        try:
            # 记录请求参数
            params = await async_accept_param_purge(request, endpoint="_aigc_functions_likang_introduction_v1_1_0")

            # 调用生成 Likang 介绍的逻辑
            response = await itinerary_model.aigc_functions_likang_introduction(**params)

            # 统一格式化响应
            return make_result(head=200, msg="成功生成Likang介绍", items=response, log=True)

        except Exception as e:
            # 处理异常
            logger.error(f"Error in _aigc_functions_likang_introduction_v1_1_0: {e}")
            return make_result(head=500, msg="生成Likang介绍失败", items=None)

    @app.route("/health/user_labels", methods=["POST"])
    async def _aigc_functions_health_user_active_label(
            request_model: Optional[SanJiKangYangRequest] = None,
    ) -> Response:
        """健康用户标签提取接口"""
        endpoint = "/health/user_labels"
        endpoint_name = "health_user_labels"
        param = {}  # 提前定义，避免后续引用异常

        start_time = time.time()
        try:
            try:
                # 允许为空，不抛异常
                if request_model is None:
                    param = {}
                else:
                    param = await async_accept_param_purge(request_model, endpoint=endpoint)
            except Exception as e:
                logger.warning(f"Param parse error: {e}")
                param = {}

            # 设置默认 intentCode 和 endpoint_name
            param.setdefault("intentCode", "aigc_functions_health_user_active_label")
            param.setdefault("endpoint_name", endpoint_name)

            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)

            # 获取 tags
            intent_code = param.get("intentCode")
            if intent_code:
                try:
                    prompt = gsr.langfuse_client.get_prompt(intent_code)
                    tags = prompt.tags if prompt else []
                    param['endpoint_name'] = f"health_user_labels_{intent_code}"
                except Exception as e:
                    logger.error(f"Error fetching tags for intent_code {intent_code}: {str(e)}")
                    tags = ["default_tag"]
                    param['endpoint_name'] = "health_user_labels"
            else:
                tags = ["aigc_health_user_labels", "健康管理", "用户标签", "健康画像"]

            param['tags'] = tags

            # 不需要流式处理，直接返回结果
            response = replace_you(response) if callable(replace_you) and hasattr(replace_you,
                                                                                        "__await__") else replace_you(
                response)
            ret = AigcFunctionsCompletionResponse(head=200, items=response)
            _return = ret.model_dump_json(exclude_unset=False)

            end_time = time.time()
            logger.info(f"Endpoint: {endpoint}, Final response: {_return}")
            await record_monitoring_data([response], param, start_time, end_time)

            return build_aigc_functions_response(_return)

        except Exception as err:
            msg = replace_you(repr(err)) if callable(replace_you) and hasattr(replace_you, "__await__") else repr(
                err)
            logger.error(f"Error in {endpoint}: {msg}")

            ret = AigcFunctionsCompletionResponse(head=601, msg=msg, items=None)
            _return = ret.model_dump_json(exclude_unset=True)

            end_time = time.time()
            await record_monitoring_data([msg], param, start_time, end_time)

            return build_aigc_functions_response(_return)

    @app.post("/health/get_all_memories", description="获取所有用户记忆数据接口")
    async def get_all_memories(request_model: Optional[SanJiKangYangRequest] = None) -> Response:
        """获取所有记忆数据接口"""
        endpoint = "/health/get_all_memories"
        endpoint_name = "get_all_memories"
        param = {}  # 可选参数

        start_time = time.time()

        try:
            # 解析请求参数
            if request_model is None:
                param = {}
            else:
                param = await async_accept_param_purge(request_model, endpoint=endpoint)

            # 设置默认 intentCode 和 endpoint_name
            param.setdefault("intentCode", "aigc_functions_get_all_memories")
            param.setdefault("endpoint_name", endpoint_name)

            # 获取mem0_url，假设 gsr 存在并且包含了 mem0_url
            if hasattr(gsr, "mem0_url") and gsr.mem0_url:
                mem0_url = gsr.mem0_url
            else:
                raise ValueError("Mem0 URL is missing in the gsr context!")

            # 调用获取所有记忆的异步函数
            memories = await call_mem0_get_all_memories(mem0_url=mem0_url)

            if memories is not None:
                # 成功返回记忆数据
                ret = AigcFunctionsCompletionResponse(head=200, items=memories)
                _return = ret.model_dump_json(exclude_unset=False)
            else:
                # 如果获取失败，返回错误
                ret = AigcFunctionsCompletionResponse(head=500, msg="Failed to retrieve memories.", items=None)
                _return = ret.model_dump_json(exclude_unset=True)

            end_time = time.time()
            logger.info(f"Endpoint: {endpoint}, Final response: {_return}")
            await record_monitoring_data([memories], param, start_time, end_time)

            return build_aigc_functions_response(_return)

        except Exception as err:
            msg = replace_you(repr(err)) if callable(replace_you) and hasattr(replace_you, "__await__") else repr(err)
            logger.error(f"Error in {endpoint}: {msg}")

            ret = AigcFunctionsCompletionResponse(head=601, msg=msg, items=None)
            _return = ret.model_dump_json(exclude_unset=True)

            end_time = time.time()
            await record_monitoring_data([msg], param, start_time, end_time)

            return build_aigc_functions_response(_return)

    @app.route("/aigc/intervene", methods=["POST"])
    async def _aigc_functions_intervention_entry(
            request_model: Optional[SanJiKangYangRequest] = None,
    ) -> Response:
        """327干预计划统一入口"""
        endpoint = "/aigc/intervene"
        endpoint_name = "aigc_intervene"
        param = {}

        start_time = time.time()
        try:
            param = await async_accept_param_purge(request_model, endpoint=endpoint)
            # 如果没有提供 intent_code，会报错
            intent_code = param.get("intentCode")
            if not intent_code:
                raise ValueError("缺少必填字段：intentCode")

            # 设置默认 endpoint_name（用于langfuse trace）
            param.setdefault("endpoint_name", f"{endpoint_name}_{intent_code}")

            # 执行干预能力调用（支持四个）
            response: Union[str, AsyncGenerator] = await health_expert_model.call_function(**param)

            # 设置 tags（尝试从 langfuse 获取）
            try:
                prompt = gsr.langfuse_client.get_prompt(intent_code)
                tags = prompt.tags if prompt else []
            except Exception as e:
                logger.warning(f"[{endpoint}] Failed to fetch tags for {intent_code}: {str(e)}")
                tags = ["aigc_intervention", "327干预计划", "姚树坤"]

            param["tags"] = tags

            # 处理返回值
            response = replace_you(response) if callable(replace_you) and hasattr(replace_you, "__await__") else replace_you(
                response)
            ret = AigcFunctionsCompletionResponse(head=200, items=response)
            _return = ret.model_dump_json(exclude_unset=False)

            logger.info(f"[{endpoint}] Final response: {_return}")
            end_time = time.time()
            await record_monitoring_data([response], param, start_time, end_time)

            return build_aigc_functions_response(_return)

        except Exception as err:
            msg = replace_you(repr(err)) if callable(replace_you) and hasattr(replace_you, "__await__") else repr(
                err)
            logger.error(f"[{endpoint}] Error: {msg}")

            ret = AigcFunctionsCompletionResponse(head=601, msg=msg, items=None)
            _return = ret.model_dump_json(exclude_unset=True)

            end_time = time.time()
            await record_monitoring_data([msg], param, start_time, end_time)

            return build_aigc_functions_response(_return)

    # 挂载 POST 路由
    app.post("/aigc/functions", description="AIGC函数")(_async_aigc_functions)

    app.post("/aigc/functions/doctor_recommend")(_async_aigc_functions_doctor_recommend)

    app.post("/aigc/sanji")(_async_aigc_sanji)

    app.post("/aigc/outpatient_support")(_async_aigc_functions_outpatient_support)

    app.post("/aigc/jkbao", description="家康宝")(_async_aigc_functions_jia_kang_bao_support)

    app.post("/aigc/sanji/kangyang")(_async_aigc_functions_sanji_kangyang)

    app.post("/aigc/junwang/gongjian")(_async_aigc_functions_junwang_gongjian)

    app.post("/aigc/itinerary/make", description="根据用户的偏好和需求生成个性化行程清单")(_aigc_functions_generate_itinerary)

    app.post("/aigc/bath_plan/make", description="生成泡浴方案")(_aigc_functions_generate_bath_plan)

    app.post("/aigc/v1_1_0/itinerary", description="根据用户的偏好和需求生成个性化行程清单（V1.1.0）")(_aigc_functions_generate_itinerary_v1_1_0)

    app.post("/aigc/v1_1_0/bath_plan", description="生成泡浴方案（V1.1.0）")(_aigc_functions_generate_bath_plan_v1_1_0)

    app.post("/aigc/v1_1_0/likang_introduction", description="固安来康郡介绍（V1.1.0）")(_aigc_functions_likang_introduction_v1_1_0)

    app.post("/health/user_labels", description="健康用户标签提取接口")(_aigc_functions_health_user_active_label)

    app.post("/aigc/intervene", description="327干预计划统一入口")(_aigc_functions_intervention_entry)


def mount_multimodal_endpoints(app: FastAPI):

    @app.route("/func_eval/image_type_recog", methods=["post"])
    async def _func_eval_image_type_recog(request: Request):
        """图片分类，包含：饮食、运动、报告、其他"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/image_type_recog")
            ret = await multimodal_model.image_type_recog(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/func_eval/diet_recog", methods=["post"])
    async def _func_eval_diet_recog(request: Request):
        """菜品识别，提取菜品名称、数量、单位信息"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/diet_recog")
            ret = await multimodal_model.diet_recog(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/func_eval/diet_recog_customer", methods=["post"])
    async def _func_eval_diet_recog_customer(request: Request):
        """C端菜品识别，提取菜品名称、数量、单位、克重、能量信息"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/diet_recog_customer")
            ret = await multimodal_model.diet_recog_customer(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/func_eval/diet_eval", methods=["post"])
    async def _func_eval_diet_eval(request: Request):
        """饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/diet_eval")
            ret = await multimodal_model.diet_eval(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/func_eval/diet_eval_customer", methods=["post"])
    async def _func_eval_diet_eval_customer(request: Request):
        """C端饮食评估，根据用户信息、饮食信息、用户管理标签、餐段信息，生成一句话点评"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/diet_eval_customer")
            ret = await multimodal_model.diet_eval_customer(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/func_eval/general_recog", methods=["post"])
    async def _func_eval_general_recog(request: Request):
        """通用评估，根据图片和提示词（可选）返回结果"""
        try:
            param = await async_accept_param_purge(request, endpoint="/func_eval/general_recog")
            ret = await multimodal_model.general_recog(**param)
            ret = make_result(head=ret["head"], items=ret["items"], msg=ret["msg"])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret


def create_app():
    app: FastAPI = FastAPI(
        title="智能健康管家-算法",
        description="",
        version=f"{datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
    )
    prepare_for_all()

    async def document():  # 用于展示接口文档
        return RedirectResponse(url="/docs")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
            request: Request, exc: RequestValidationError
    ):
        # 提取并格式化错误信息
        errors = exc.errors()
        return JSONResponse(
            status_code=200,
            content={"head": 500, "items": None, "msg": repr(errors)},
        )

    # 全局异常处理器：用于处理其他未捕获的异常
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(exc)  # 打印异常日志，便于调试
        return JSONResponse(
            status_code=200,  # 强制返回HTTP 200
            content={"head": 500, "items": None, "msg": repr(exc)},  # 自定义错误信息
        )

    @app.get("/health", summary="健康检查接口")
    async def health_check():
        """健康检查接口"""
        start_time = time.time()  # 记录开始时间

        # 返回健康状态
        response = {"status": "healthy"}

        # 记录响应时间
        response_time = time.time() - start_time
        logger.info(f"\033[32m[monitor]\033[0m Health check completed in {response_time:.2f} seconds.")
        return response

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
                res = json.dumps(item, ensure_ascii=False).replace('您', '你') if item.get(
                    "intentCode") != "jia_kang_bao" else json.dumps(item, ensure_ascii=False)
                yield format_sse_chat_complete(
                    res, "delta"
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
                res = json.dumps(item, ensure_ascii=False).replace('您', '你')
                yield format_sse_chat_complete(
                    res, "delta"
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

    # @app.route("/func_eval/image_type_recog", methods=["post"])
    # async def _image_type_recog(request: Request):
    #     """图片类型识别"""
    #     try:
    #         param = await accept_param(request, endpoint="/func_eval/image_type_recog")
    #         generator: AsyncGenerator = image_recog(param.get("image_url", ""))
    #         result = decorate_general_complete(
    #             generator
    #         )
    #     except Exception as err:
    #         logger.exception(err)
    #         result = yield_result(head=600, msg=repr(err), items=param)
    #     finally:
    #         return StreamingResponse(result, media_type="text/event-stream")
    #
    # @app.route("/func_eval/diet_image_recog", methods=["post"])
    # async def _image_type_recog(request: Request):
    #     """饮食图片识别"""
    #     try:
    #         param = await accept_param(request, endpoint="/func_eval/diet_image_recog")
    #         generator: AsyncGenerator = diet_image_recog(param.get("image_url", ""))
    #         result = decorate_general_complete(
    #             generator
    #         )
    #     except Exception as err:
    #         logger.exception(err)
    #         result = yield_result(head=600, msg=repr(err), items=param)
    #     finally:
    #         return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/func_eval/schedule_tips_modification", methods=["post"])
    async def _schedule_tips_modify(request: Request):
        """用户日程tips修改"""
        try:
            param = await accept_param(request, endpoint="/func_eval/schedule_tips_modification")
            generator: AsyncGenerator = schedule_tips_modify(param.get("schedule_tips", ""),
                                                             param.get("history", []),
                                                             param.get("cur_time", ''))
            result = decorate_general_complete(
                generator
            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return StreamingResponse(result, media_type="text/event-stream")

    @app.route("/func_eval/sport_schedule_modify_suggestion", methods=["post"])
    async def _schedule_tips_modify(request: Request):
        """用户运动日程修改"""
        try:
            param = await accept_param(request, endpoint="/func_eval/sport_schedule_modify_suggestion")
            item = await sport_schedule_tips_modify(param.get("schedule", []),
                                              param.get("history", []),
                                              param.get("cur_time", ''))
            result = make_result(items=item)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            return result

    @app.route("/func_eval/daily_diet_status", methods=["post"])
    async def _daily_diet_status(request: Request):
        """一日饮食状态"""
        try:
            param = await accept_param(request, endpoint="/func_eval/daily_diet_status")
            if not param.get("daily_diet_info", []):
                result = make_result(items={}, msg='请检查入参格式', head=402)
                return result
            item = await daily_diet_degree(param.get("userInfo", {}),
                                   param.get("daily_diet_info", []),
                                   param.get("daily_blood_glucose", ''),
                                   param.get("management_tag", '血糖管理'))
            result = make_result(items=item)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param,head=402)
        finally:
            return result

    @app.route("/func_eval/daily_diet_eval", methods=["POST"])
    async def _daily_diet_eval(request: Request):
        """一日血糖饮食建议"""

        start_time = time.time()
        try:
            param = await accept_param(request, endpoint="/func_eval/daily_diet_eval")
            param['endpoint_name'] = "daily_diet_eval"
            param['tags'] = ["daily_diet_eval", "饮食建议", "健康管理", "血糖控制"]
            # 获取生成器
            generator: AsyncGenerator = daily_diet_eval(
                param.get("userInfo", {}),
                param.get("daily_diet_info", []),
                param.get("daily_blood_glucose", ''),
                param.get("management_tag", '血糖管理')
            )

            # 包装生成器，记录数据
            wrapped_generator = logging_wrapper(generator, param, start_time)

            result = decorate_general_complete(
                wrapped_generator
            )

            # 返回流式响应
            return StreamingResponse(result, media_type="text/event-stream")

        except Exception as err:
            logger.exception(err)

            # 异常处理返回流式错误
            async def error_generator():
                yield json.dumps({"error": repr(err)}, ensure_ascii=False)

            end_time = time.time()
            asyncio.create_task(record_monitoring_data([], param, start_time, end_time))

            return StreamingResponse(error_generator(), media_type="text/event-stream")

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
                                                                       param.get('personal_dietary_requirements', ''),
                                                                       param.get('history', []),
                                                                       param.get('userInfo', {}),
                                                                       )
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
                                                                    param.get('days'),
                                                                    param.get('history', []),
                                                                    param.get('requirements', []),
                                                                    param.get('reference_diet', []),
                                                                    )
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
                                                                     param.get('personal_dietary_requirements', ''),
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
            elif param.get("intentCode") in ["aigc_jiahe_ingredient_pairing_suggestion", "aigc_jiahe_ingredient_selection_guide", "aigc_jiahe_ingredient_taboo_groups"]:
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

    @app.route("/chat_gen", methods=["POST"])
    async def get_chat_gen(request: Request):
        global chat

        start_time = time.time()
        try:
            # 获取参数并设置默认 tags 和 endpoint_name
            param = await accept_param(request, endpoint="/chat_gen")
            param['tags'] = ["chat_gen", "会话生成", "对话管理", "用户交互"]
            param['endpoint_name'] = "chat_gen"  # 设置端点名称

            generator: AsyncGenerator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"),
                mid_vars=[],
                use_sys_prompt=False,
                **param,
            )

            # 包装生成器，记录数据
            wrapped_generator = logging_wrapper(generator, param, start_time)

            user_id = param.get("customId", "unknown")
            messages = param.get("history", [])
            # **异步调用 mem0 记录对话**
            if hasattr(gsr, "mem0_url") and gsr.mem0_url:
                try:
                    asyncio.create_task(call_mem0_add_memory(gsr.mem0_url, user_id, messages))
                except Exception as mem0_err:
                    pass
                    # logger.exception(f"mem0 调用失败: {mem0_err}")

            # 原逻辑保持不变
            result = decorate_chat_complete(
                wrapped_generator,  # 使用包装后的生成器
                return_mid_vars=False,
                return_backend_history=True
            )

            # 返回流式响应
            return StreamingResponse(result, media_type="text/event-stream")

        except Exception as err:
            logger.exception(err)

            # 异常处理返回流式错误
            async def error_generator():
                yield json.dumps({"error": repr(err)}, ensure_ascii=False)

            end_time = time.time()
            # 创建异步任务记录监控数据
            asyncio.create_task(record_monitoring_data([], param, start_time, end_time))

            return StreamingResponse(error_generator(), media_type="text/event-stream")

    @app.route("/chat/complete", methods=["post"])
    async def _chat_complete(request: Request):
        """demo,主要用于展示返回的中间变量"""
        try:
            param = await accept_param(request, endpoint="/chat/complete")
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
            param = await accept_param(request, endpoint="/chat/role_play")
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
        endpoint = "/intent/query"

        start_time = time.time()
        try:
            # 获取请求参数
            param = await accept_param(request, endpoint=endpoint)

            # 执行 intent_query 方法
            item = chat.intent_query(
                param.get("history", []),
                task=param.get("task", ""),
                prompt=param.get("prompt", ""),
                userInfo=param.get("promptParam", ""),
                intentPrompt=param.get("intentPrompt", ""),
                subIntentPrompt=param.get("subIntentPrmopt", ""),
                scene_code=param.get("scene_code", "default"),
            )

            # 记录监控数据
            tags = ["intent_query", "意图查询", "智能问答", "对话系统"]
            param['tags'] = tags
            param["endpoint_name"] = endpoint
            logger.info(f"Intent query result: {repr(item)}")

            # 构造返回的结果
            result = make_result(items=item)

            end_time = time.time()
            logger.info(f"Endpoint: {endpoint}, Final response: {repr(result)}")

            # 记录监控数据
            await record_monitoring_data([item], param, start_time, end_time)

        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)

        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)

        finally:
            # 在finally块中确保记录监控数据
            end_time = time.time()
            await record_monitoring_data([result], param, start_time, end_time)
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
                # 处理流式响应
                _return = response_generator(response)
            else:
                # 非流式响应
                response = replace_you(response)  # 直接替换“您”为“你”
                if isinstance(response, str):
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
                else:
                    ret = AigcFunctionsCompletionResponse(
                        head=200, items=response
                    )
                    _return = ret.model_dump_json(exclude_unset=False)
        except Exception as err:
            # 错误处理
            msg = repr(err)
            msg = replace_you(msg)  # 错误消息统一替换
            if param.get("model_args") and param["model_args"].get("stream") is True:
                _return = response_generator(msg, error=True)  # 流式错误响应
            else:
                ret = AigcFunctionsCompletionResponse(
                    head=601, msg=msg, items=None
                )
                _return = ret.model_dump_json(exclude_unset=True)  # 非流式错误响应
        finally:
            return build_aigc_functions_response(_return)  # 确保返回值有赋值

    app.route("/aigc/weight_management", methods=["post"])(_aigc_functions_weight_management)

    mount_aigc_functions(app)
    mount_rule_endpoints(app)
    mount_rec_endpoints(app)
    mount_multimodal_endpoints(app)
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
    global itinerary_model
    global bath_plan_model
    global multimodal_model

    gsr = InitAllResource()
    args = gsr.args
    chat = Chat(gsr)
    chat_v2 = Chat_v2(gsr)
    expert_model = expertModel(gsr)
    agents = Agents(gsr)
    health_expert_model = HealthExpertModel(gsr)
    jiahe_expert = JiaheExpertModel(gsr)
    itinerary_model = ItineraryModel(gsr)
    bath_plan_model = BathPlanModel(gsr)
    multimodal_model = MultiModalModel(gsr)


# app = create_app()

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host=args.ip, port=args.port, log_level="info")
    # name_app = os.path.splitext(os.path.basename(__file__))[0]
    # uvicorn.run(app=f"{name_app}:app", host=args.ip, port=args.port)
