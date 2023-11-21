# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-14 17:38:45
@desc    :   server
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import argparse
import json
import traceback

from flask import Flask, Response, request
from gevent import pywsgi

from chat.qwen_chat import Chat
from src.utils.Logger import logger
from src.utils.module import NpEncoder, clock


def accept_param():
    p = json.loads(request.data.decode("utf-8"))
    logger.info(f"=============Input Param===========\n{p}")
    return p

def make_result(head=200, msg=None, items=None, cls=False, **kwargs):
    if not items and head == 200:
        head = 600
    res = {"head":head,"msg":msg,"items":items, **kwargs}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    return res

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    

def decorate(generator):
    try:
        for item in generator:
            yield format_sse(json.dumps(item, ensure_ascii=False), 'delta')
    except Exception as err:
        logger.exception(err)

def create_app():
    app = Flask(__name__)
    global chat

    # @app.route('/chat', methods=['post'])
    # def get_chat_reponse():
    #     global chat
    #     try:
    #         param = accept_param()
    #         task = param.get('task', 'chat')
    #         customId = param.get('customId', '')
    #         orgCode = param.get('orgCode', '')
    #         userInfo = param.get('promptParam', {}) 
    #         if task == 'chat':
    #             result = chat.run_prediction(param.get('history',[]),
    #                                         param.get('prompt',''),
    #                                         param.get('intentCode','default_code'), 
    #                                         customId=customId,
    #                                         orgCode=orgCode, 
    #                                         streaming=param.get('streaming', True),
    #                                         userInfo=userInfo
    #                                         )
    #     except AssertionError as err:
    #         logger.exception(err)
    #         result = make_result(head=601, msg=repr(err), items=param)
    #     except Exception as err:
    #         logger.exception(err)
    #         logger.error(traceback.format_exc())
    #         result = make_result(msg=repr(err), items=param)
    #     finally:
    #         return Response(decorate(result), mimetype='text/event-stream')
    
    @app.route('/chat_gen', methods=['post'])
    def get_chat_gen():
        global chat
        try:
            param = accept_param()
            task = param.get('task', 'chat')
            customId = param.get('customId', '')
            orgCode = param.get('orgCode', '')
            userInfo = param.get('promptParam', {}) 
            if task == 'chat':
                result = chat.chat_gen(param.get('history',[]),
                                            param.get('prompt',''),
                                            param.get('intentCode','default_code'), 
                                            customId=customId,
                                            orgCode=orgCode, 
                                            streaming=param.get('streaming', True),
                                            userInfo=userInfo
                                            )
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            return Response(decorate(result), mimetype='text/event-stream')
        
    @app.route('/intent/query', methods=['post'])
    def intent_query():
        global chat
        try:
            param = accept_param()
            item = chat.intent_query(param.get('history',[]))
            result = make_result(items=item)
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            #return Response(decorate(result), mimetype='text/event-stream')
            return Response(json.dumps(result, ensure_ascii=False),
                    content_type='application/json')
        
    @app.route('/chat/complete', methods=['post'])
    def _chat_complete():
        """demo,主要用于展示返回的中间变量
        """
        global chat
        try:
            param = accept_param()
            task = param.get('task', 'chat')
            if task == 'chat':
                generator = chat.chat_gen(sys_prompt=param.get('prompt'),
                                          mid_vars = [],
                                          streaming=False,
                                          use_sys_prompt=True,
                                          **param)
                out_text, mid_vars = next(generator)
                del out_text['end']
                result = make_result(head=200, msg="success", items={'mid_vars':mid_vars, **out_text})
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            result = make_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(json.dumps(result, ensure_ascii=False), content_type="application/json")
        
    @app.route('/test_generator', methods=['get'])
    def _test_generator():
        """测试生成器
        """
        def func():
            yield json.dumps({"end": False, "num": "1"})
            yield json.dumps({"end": False, "num": "2"})
        generator = func()
        return Response(decorate(generator), mimetype='text/event-stream')

    @app.route('/reload_prompt', methods=['get'])
    def _reload_prompt():
        """重启chat实例
        """
        global chat
        try:
            chat.reload_prompt()
            ret = {"head": 200, "success": True, "msg": "restart success"}
        except Exception as err:
            logger.exception(err)
            ret = {"head": 500, "success": False, "msg": repr(err)}
        finally:
            return ret
    
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
        
    return app

def server_forever(args):
    global app
    # app.run(host=args.ip, port=args.port, debug=True)
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    logger.debug(f"serve at {args.ip}:{args.port}")
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="local", help='env: local, dev, test, prod')
    parser.add_argument('--ip', type=str, default="0.0.0.0", help='ip')
    parser.add_argument('--port', type=int, default=6500, help='port')
    args = parser.parse_args()

    chat = Chat(args.env)
    app = create_app()
    server_forever(args)