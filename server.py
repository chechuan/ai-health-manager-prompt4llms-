#encoding='utf-8'
import argparse
import json
import traceback

from flask import Flask, Response, request, stream_with_context
# from flask_cors import CORS
from gevent import pywsgi

from chat.qwen_chat import Chat
from src.utils.Logger import logger
from src.utils.module import NpEncoder, clock

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="local", help='env: local, dev, test, prod')
parser.add_argument('--ip', type=str, default="0.0.0.0", help='ip')
parser.add_argument('--port', type=int, default=6500, help='port')
args = parser.parse_args()

app = Flask(__name__)
chat = Chat(args.env)

def accept_param():
    p = json.loads(request.data.decode("utf-8"))
    logger.info(f"=============Input Param===========\n{p}")
    return p

def make_result(param, head=200, msg=None, items=None, cls=False):
    if not items and head == 200:
        head = 600
    res = {"head":head,"msg":msg,"items":items}
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    return res

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    

def decorate(generator):
    for item in generator:
        yield format_sse(json.dumps(item, ensure_ascii=False), 'delta')

@app.route('/chat', methods=['post'])
def get_chat_reponse():
    global chat
    try:
        param = accept_param()
        task = param.get('task', 'chat')
        customId = param.get('customId', '')
        orgCode = param.get('orgCode', '')
        if task == 'chat':
            result = chat.run_prediction(param.get('history',[]),
                                         param.get('prompt',''),
                                         param.get('intentCode','default_code'), 
                                         customId=customId,
                                         orgCode=orgCode, 
                                         streaming=param.get('streaming', True)
                                         )
    except AssertionError as err:
        logger.error(traceback.format_exc())
        result = make_result(param, head=601, msg=repr(err))
    except Exception as err:
        logger.error(traceback.format_exc())
        result = make_result(param, msg=repr(err))
    finally:
        return Response(decorate(result), mimetype='text/event-stream')

@app.route('/reload_prompt', methods=['get'])
def _reload_prompt():
    """重启chat实例
    """
    global chat
    try:
        chat.reload_prompt()
        ret = {"code": 200, "success": True, "msg": "restart success"}
    except Exception as err:
        ret = {"code": 500, "success": False, "msg": repr(err)}
    finally:
        return ret

def server_forever(args):
    global app
    server = pywsgi.WSGIServer((args.ip, args.port), app)
    logger.debug(f"serve at {args.ip}:{args.port}")
    server.serve_forever()


if __name__ == '__main__':
    server_forever(args)
    #uvicorn.run(app, host='0.0.0.0', port=6500, log_level="info")
