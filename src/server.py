# -*- encoding: utf-8 -*-
"""
@Time    :   2023-11-14 17:38:45
@desc    :   server
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
from crypt import methods
from fileinput import filename

from gevent import monkey, pywsgi

monkey.patch_all()
import json
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import time

from flask import Flask, Response, request

from chat.qwen_chat import Chat
from src.pkgs.models.small_expert_model import expertModel
from src.pkgs.pipeline import Chat_v2
from src.utils.api_protocal import RolePlayRequest
from src.utils.Logger import logger
from src.utils.module import (InitAllResource, NpEncoder, curr_time, decorate_text_stream, dumpJS,
                              format_sse_chat_complete)


def accept_param():
    p = json.loads(request.data.decode("utf-8"))
    backend_history = p.get("backend_history", [])
    p["backend_history"] = json.loads(backend_history) if isinstance(backend_history, str) else backend_history
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


def accept_param_purge():
    p = request.get_json()
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p


def make_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head": head, "msg": msg, "items": items, **kwargs}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    return res


def yield_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head": head, "message": msg, "items": items, **kwargs, "end": True}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    yield res


def decorate_chat_complete(generator, return_mid_vars=False, return_backend_history=False):
    try:
        while True:
            yield_item = next(generator)
            yield_item["data"]['appendData'] = yield_item['appendData']
            item = {**yield_item["data"]}
            logger.info("Output (except mid_vars & backend_history):\n" + json.dumps(item, ensure_ascii=False))
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
            yield format_sse_chat_complete(json.dumps(item, ensure_ascii=False), "delta")
            if yield_item["data"]["end"] == True:
                break
    except Exception as err:
        logger.exception(err)
        item = make_result(head=600, message=repr(err), end=True)
        yield format_sse_chat_complete(json.dumps(item, ensure_ascii=False), "delta")


def create_app():
    app = Flask(__name__)

    @app.route("/chat_gen", methods=["post"])
    def get_chat_gen():
        global chat
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"), mid_vars=[], use_sys_prompt=False, **param
            )
            result = decorate_chat_complete(generator, return_mid_vars=False, return_backend_history=True)
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype="text/event-stream")

    @app.route("/chat/complete", methods=["post"])
    def _chat_complete_stream_midvars():
        """demo,主要用于展示返回的中间变量"""
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"), mid_vars=[], use_sys_prompt=True, **param
            )
            result = decorate_chat_complete(generator, return_mid_vars=True, return_backend_history=True)
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype="text/event-stream")

    @app.route("/chat/role_play", methods=["post"])
    def _chat_role_play(request: RolePlayRequest):
        """角色扮演对话"""
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(
                sys_prompt=param.get("prompt"), mid_vars=[], use_sys_prompt=True, **param
            )
            result = decorate_chat_complete(generator, return_mid_vars=True, return_backend_history=True)
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype="text/event-stream")

    @app.route("/intent/query", methods=["post"])
    def intent_query():
        global chat
        try:
            param = accept_param()
            item = chat.intent_query(
                param.get("history", []),
                task=param.get("task", ""),
                prompt=param.get("prompt", ""),
                userInfo=param.get("promptParam", ""),
                intentPrompt=param.get("intentPrompt", ""),
                subIntentPrompt=param.get("subIntentPrmopt", ""),
                scene_code=param.get('scene_code', 'default')
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
    def _reload_prompt():
        """重启chat实例"""
        global chat
        try:
            prepare_for_all()
            ret = {"head": 200, "success": True, "msg": "restart success"}
        except Exception as err:
            logger.exception(err)
            ret = {"head": 500, "success": False, "msg": repr(err)}
        finally:
            return Response(dumpJS(ret), content_type="application/json")

    @app.route("/fetch_intent_code", methods=["get"])
    def _fetch_intent_code():
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

    @app.route("/rec/diet/food_purchasing_list/manage", methods=["post"])
    def _rec_diet_create_food_purchasing_list_manage():
        """食材采购清单管理"""
        try:
            param = accept_param_purge()
            ret = expert_model.food_purchasing_list_manage(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rec/diet/food_purchasing_list/generate_by_content", methods=["post"])
    def _rec_diet_create_food_purchasing_list_generate_by_content():
        """食材采购清单管理"""
        try:
            param = accept_param_purge()
            ret = expert_model.food_purchasing_list_generate_by_content(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/rec/diet/reunion_meals/restaurant_selection", methods=["post"])
    def _rec_diet_reunion_meals_restaurant_selection():
        """年夜饭, 结合群组对话和餐厅信息选择偏好餐厅"""
        try:
            param = accept_param_purge()
            generator = expert_model.rec_diet_reunion_meals_restaurant_selection(**param)
            ret = decorate_text_stream(generator)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return Response(ret, mimetype="text/event-stream")

    @app.route("/rec/diet/evaluation", methods=["post"])
    def _rec_diet_evaluation():
        """膳食摄入评估"""
        try:
            param = accept_param_purge()
            ret = expert_model.rec_diet_eval(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/blood_pressure_trend_analysis", methods=["post"])
    def _health_blood_pressure_trend_analysis():
        """血压趋势分析"""
        try:
            param = accept_param_purge()
            ret = expert_model.health_blood_pressure_trend_analysis(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/health/warning_solutions_early", methods=["post"])
    def _health_warning_solutions_early():
        """预警解决方案"""
        try:
            param = accept_param_purge()
            ret = expert_model.health_warning_solutions_early(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/search/duckduckgo", methods=["post"])
    def _search_duckduckgo():
        """DuckDuckGo搜索"""
        try:
            param = accept_param_purge()
            ret = chat_v2.funcall.search_qa_chain.ddg_search_chain.call(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/search/crawler/sougou", methods=["post"])
    def _search_crawler_sougou():
        """爬虫 - 搜狗搜索"""
        try:
            param = accept_param_purge()
            ret = chat_v2.funcall.call_search_engine(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/aigc/functions", methods=["post"])
    def _aigc_functions():
        """aigc函数"""
        try:
            param = accept_param_purge()
            ret = expert_model.call_function(**param)
            ret = make_result(items=ret)
        except RuntimeError as err:
            logger.error(err)
            ret = make_result(head=601, msg=err.args[0])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg="Unknown error.")
        finally:
            return ret
    
    @app.route("/aigc/functions/report_interpretation", methods=["post"])
    def _aigc_functions_report_interpretation():
        """aigc函数-报告解读"""
        try:
            # param = accept_param_purge()
            upload_file = request.files.get("file")
            filename = upload_file.filename
            tmp_path = Path(f".tmp/images")
            if not tmp_path.exists():
                tmp_path.mkdir(parents=True)
            file_path = tmp_path.joinpath(filename)
            upload_file.save(file_path)
            ret = expert_model.call_function(intentCode="report_interpretation", file_path=file_path)
            ret = make_result(items=ret)
        except RuntimeError as err:
            logger.error(err)
            ret = make_result(head=601, msg=err.args[0])
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg="Unknown error.")
        finally:
            return ret

    @app.route("/rules/blood_pressure_level", methods=["post"])
    def _rules_blood_pressure_level():
        """计算血压等级及处理规则"""
        try:
            param = accept_param_purge()
            ret = expert_model.tool_rules_blood_pressure_level(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
    
    @app.route("/rules/emotions", methods=["post"])
    def _rules_enotions_level():
        """情志分级"""
        try:
            param = accept_param_purge()
            ret = expert_model.emotions(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
        
    @app.route("/rules/weight_trend", methods=["post"])
    def _rules_weight_trend():
        """体重趋势"""
        try:
            param = accept_param_purge()
            ret = expert_model.weight_trend(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
        
    @app.route("/rules/fat_reduction", methods=["post"])
    def _rules_fat_reduction():
        """体重减脂"""
        try:
            param = accept_param_purge()
            ret = expert_model.fat_reduction(**param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route("/test/sync", methods=["post"])
    def _test_sync():
        """异步测试"""
        t1 = curr_time()
        time.sleep(2)
        ret = {"start": t1, "end": curr_time()}
        logger.debug(ret)
        return Response(dumpJS(ret), content_type="application/json")

    return app


def prepare_for_all():
    global chat
    global args
    global chat_v2
    global expert_model
    global global_share_resource

    global_share_resource = InitAllResource()
    args = global_share_resource.args
    chat = Chat(global_share_resource)
    chat_v2 = Chat_v2(global_share_resource)
    expert_model = expertModel(global_share_resource)


def server_forever():
    global app
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    logger.success(f"serve at http://{args.ip}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    prepare_for_all()
    app = create_app()
    server_forever()
