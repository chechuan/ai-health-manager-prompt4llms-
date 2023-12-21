# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-14 17:38:45
@desc    :   server
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
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
from src.utils.api_protocal import RolePlayRequest, healthBloodPressureTrendAnalysis
from src.utils.Logger import logger
from src.utils.module import NpEncoder, clock, curr_time, dumpJS, initAllResource


def accept_param():
    p = json.loads(request.data.decode("utf-8"))
    backend_history = p.get('backend_history', [])
    p['backend_history'] = json.loads(backend_history) if isinstance(backend_history, str) else backend_history
    pstr = json.dumps(p, ensure_ascii=False)
    logger.info(f"Input Param: {pstr}")
    return p

def make_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head":head,"msg":msg,"items":items, **kwargs}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    return res

def yield_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head":head,"message":msg,"items":items, **kwargs, "end":True}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    yield res

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    

def decorate(generator):
    try:
        for item in generator:
            item['backend_history'] = []
            yield format_sse(json.dumps(item, ensure_ascii=False), 'delta')
    except Exception as err:
        logger.exception(err)

def format_sse_chat_complete(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    

def decorate_chat_complete(generator, return_mid_vars=False, return_backend_history=False):
    try:
        while True:
            yield_item = next(generator)
            item = {**yield_item['data']}
            logger.info('Output (except mid_vars & backend_history):\n' + json.dumps(item, ensure_ascii=False))
            if return_mid_vars:
                if item['end'] is True:
                    item['mid_vars'] = yield_item['mid_vars']
                else:
                    item['mid_vars'] = []
            if return_backend_history:
                if item['end'] is True:
                    item['backend_history'] = yield_item['history']
                else:
                    item['backend_history'] = []
            yield format_sse_chat_complete(json.dumps(item, ensure_ascii=False), 'delta')
            if yield_item['data']['end'] == True:
                break
    except Exception as err:
        logger.exception(err)
        item = make_result(head=600, message=repr(err), end=True)
        yield format_sse_chat_complete(json.dumps(item, ensure_ascii=False), 'delta')

def create_app():
    app = Flask(__name__)
    
    # @app.route('/chat_gen', methods=['post'])
    # def get_chat_gen():
    #     try:
    #         param = accept_param()
    #         task = param.get('task', 'chat')
    #         if task == 'chat':
    #             result = chat.yield_result(sys_prompt=param.get('prompt'), 
    #                                        return_mid_vars=False, 
    #                                        use_sys_prompt=False, 
    #                                        mid_vars=[],
    #                                        **param)
    #     except AssertionError as err:
    #         logger.exception(err)
    #         result = yield_result(head=601, msg=repr(err), items=param)
    #     except Exception as err:
    #         logger.exception(err)
    #         logger.error(traceback.format_exc())
    #         result = yield_result(msg=repr(err), items=param)
    #     finally:
    #         return Response(decorate(result), mimetype='text/event-stream')

    @app.route('/chat_gen', methods=['post'])
    def get_chat_gen():
        global chat
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(sys_prompt=param.get('prompt'), 
                                                    mid_vars=[], 
                                                    use_sys_prompt=False, 
                                                    **param)
            result = decorate_chat_complete(generator, 
                                            return_mid_vars=False,
                                            return_backend_history=True
                                            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype='text/event-stream')

    @app.route('/chat/complete', methods=['post'])
    def _chat_complete_stream_midvars():
        """demo,主要用于展示返回的中间变量
        """
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(sys_prompt=param.get('prompt'), 
                                                  mid_vars=[], 
                                                  use_sys_prompt=True, 
                                                  **param)
            result = decorate_chat_complete(generator, 
                                            return_mid_vars=True,
                                            return_backend_history=True
                                            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype='text/event-stream')

    @app.route('/intent/query', methods=['post'])
    def intent_query():
        global chat
        try:
            param = accept_param()
            item = chat.intent_query(param.get('history',[]), task=param.get('task',
                ''), prompt=param.get('prompt', ''),
                userInfo=param.get('promptParam', ''),
                intentPrompt=param.get('intentPrompt', ''),
                subIntentPrompt=param.get('subIntentPrmopt', ''))
            result = make_result(items=item)
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            return Response(dumpJS(result), content_type='application/json')

    @app.route('/reload_prompt', methods=['get'])
    def _reload_prompt():
        """重启chat实例
        """
        global chat
        try:
            prepare_for_all()
            ret = {"head": 200, "success": True, "msg": "restart success"}
        except Exception as err:
            logger.exception(err)
            ret = {"head": 500, "success": False, "msg": repr(err)}
        finally:
            return Response(dumpJS(ret), content_type='application/json')
    
    @app.route('/fetch_intent_code', methods=['get'])
    def _fetch_intent_code():
        """获取意图代码
        """
        global chat
        try:
            ret = chat.fetch_intent_code()
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret

    @app.route('/rec/diet/evaluation', methods=['post'])
    def _rec_diet_evaluation():
        """获取意图代码
        """
        try:
            param = accept_param()
            ret = expert_model.__rec_diet_eval__(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
    
    @app.route('/health/blood_pressure_trend_analysis', methods=['post'])
    def _health_blood_pressure_trend_analysis():
        """血压趋势分析
        """
        try:
            param = request.get_json()
            ret = expert_model.__blood_pressure_trend_analysis__(param)
            ret = make_result(items=ret)
        except Exception as err:
            logger.exception(err)
            ret = make_result(head=500, msg=repr(err))
        finally:
            return ret
        
    @app.route('/chat/role_play', methods=['post'])
    def _chat_role_play(request: RolePlayRequest):
        """角色扮演对话
        """
        try:
            param = accept_param()
            generator = chat_v2.general_yield_result(sys_prompt=param.get('prompt'), 
                                                  mid_vars=[], 
                                                  use_sys_prompt=True, 
                                                  **param)
            result = decorate_chat_complete(generator, 
                                            return_mid_vars=True,
                                            return_backend_history=True
                                            )
        except Exception as err:
            logger.exception(err)
            result = yield_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(result, mimetype='text/event-stream')
        
    @app.route('/test/sync', methods=['post'])
    def _test_sync():
        """获取意图代码
        """
        t1 = curr_time()
        time.sleep(2)
        ret = {"start":t1, "end": curr_time()}
        logger.debug(ret)
        return Response(dumpJS(ret), content_type='application/json')    
    return app

def prepare_for_all():
    global chat
    global chat_v2
    global expert_model
    global global_share_resource
    global args

    global_share_resource = initAllResource()
    args = global_share_resource.args    
    chat = Chat(global_share_resource)
    chat_v2 = Chat_v2(global_share_resource)
    expert_model = expertModel()
    
def server_forever():
    global app
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    logger.success(f"serve at {args.ip}:{args.port}")
    server.serve_forever()

if __name__ == '__main__':
    prepare_for_all()
    app = create_app()
    server_forever()
