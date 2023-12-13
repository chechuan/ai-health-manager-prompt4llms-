# -*- encoding: utf-8 -*-
'''
@Time    :   2023-11-14 17:38:45
@desc    :   server
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import json
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import asyncio
import time

from flask import Flask, Response, request
from gevent import pywsgi

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
        # for yield_item in generator:
            item = {**yield_item['data']}
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

    @app.route('/test/sync', methods=['post'])
    def _test_sync():
        """获取意图代码
        """
        t1 = curr_time()
        time.sleep(1)
        ret = {"start":t1, "end": curr_time()}
        return Response(dumpJS(ret), content_type='application/json')
    
    async def async_sleep():
        await asyncio.sleep(1)
        return "async"

    @app.route('/test/async', methods=['post'])
    async def _test_async():
        """获取意图代码
        """
        t1 = curr_time()
        await async_sleep()
        ret = {"start":t1, "end": curr_time()}
        return Response(dumpJS(ret), content_type='application/json')
    
    return app

def prepare_for_all():
    global global_share_resource
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="local", help='env: local, dev, test, prod')
    parser.add_argument('--ip', type=str, default="0.0.0.0", help='ip')
    parser.add_argument('--port', type=int, default=6500, help='port')
    parser.add_argument('--special_prompt_version', 
                        action="store_true",
                        help='是否使用指定的prompt版本, Default为False,都使用lastest')
    args = parser.parse_args()

    logger.info(f"Initialize args: {args}")
    
def server_forever():
    global app
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    logger.success(f"serve at {args.ip}:{args.port}")
    server.serve_forever()

if __name__ == '__main__':
    prepare_for_all()
    app = create_app()
    server_forever()
