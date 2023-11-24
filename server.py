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
from src.chat.pipeline import Conv
from src.utils.Logger import logger
from src.utils.module import NpEncoder, clock


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

def format_sse_chat_complete(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    

def decorate_chat_complete(generator, ret_mid=False, ret_his=False):
    try:
        for yield_item in generator:
            item = {**yield_item['data']}
            if ret_mid:
                if item['end'] is True:
                    item['mid_vars'] = yield_item['mid_vars']
                else:
                    item['mid_vars'] = []
            if ret_his:
                if item['end'] is True:
                    item['backend_history'] = yield_item['history']
                else:
                    item['backend_history'] = []
            yield format_sse_chat_complete(json.dumps(item, ensure_ascii=False), 'delta')
    except Exception as err:
        logger.exception(err)

def create_app():
    app = Flask(__name__)
    global chat
    
    @app.route('/chat_gen', methods=['post'])
    def get_chat_gen():
        global chat
        try:
            param = accept_param()
            task = param.get('task', 'chat')
            if task == 'chat':
                result = chat.yield_result(sys_prompt=param.get('prompt'), 
                                           return_mid_vars=False, 
                                           use_sys_prompt=False, 
                                           **param)
        except AssertionError as err:
            logger.exception(err)
            result = make_result(head=601, msg=repr(err), items=param)
        except Exception as err:
            logger.exception(err)
            logger.error(traceback.format_exc())
            result = make_result(msg=repr(err), items=param)
        finally:
            return Response(decorate(result), mimetype='text/event-stream')

    @app.route('/chat/complete', methods=['post'])
    def _chat_complete_stream_midvars():
        """demo,主要用于展示返回的中间变量
        """
        # global chat
        global conv
        try:
            param = accept_param()
            result = conv.general_yield_result(sys_prompt=param.get('prompt'), 
                                               mid_vars=[], 
                                               use_sys_prompt=True, 
                                               **param)
        except Exception as err:
            logger.exception(err)
            result = make_result(head=600, msg=repr(err), items=param)
        finally:
            return Response(decorate_chat_complete(result, 
                                                   ret_mid=True,
                                                   ret_his=True), mimetype='text/event-stream')

    @app.route('/intent/query', methods=['post'])
    def intent_query():
        global chat
        try:
            param = accept_param()
            item = chat.intent_query(param.get('history',[]), task=param.get('task',
                ''), prompt=param.get('prompt', ''), userInfo=param.get('promptParam', ''))
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
    logger.success(f"serve at {args.ip}:{args.port}")
    server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="local", help='env: local, dev, test, prod')
    parser.add_argument('--ip', type=str, default="0.0.0.0", help='ip')
    parser.add_argument('--port', type=int, default=6500, help='port')
    args = parser.parse_args()

    chat = Chat(args.env)
    conv = Conv(args.env)
    app = create_app()
    server_forever(args)
