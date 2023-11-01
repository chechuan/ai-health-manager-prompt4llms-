#encoding='utf-8'
import json
import traceback

from flask import Flask, Response, request, stream_with_context
# from flask_cors import CORS
from gevent import pywsgi
# from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
#                           BitsAndBytesConfig)

from chat.qwen_chat import Chat
from utils.Logger import logger
from utils.module import NpEncoder, clock

app = Flask(__name__)

##qwen
# model_dir = 'qwen/Qwen-14B-Chat'
# model_dir = '/root/.cache/modelscope/hub/qwen/Qwen-7B-Chat'
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# print('loading model')
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
# print('model ready')
# chat = Chat(tokenizer, model)
chat = Chat()

def accept_param():
    p = json.loads(request.data.decode("utf-8"))
    logger.debug(p)
    return p

def check_param(param, key=None):
    """检查入参合法性
    """
    if isinstance(key, str):
        assert param.get(key) ,InterruptedError(f"入参缺少{key}")        
    elif isinstance(key, list):
        for k in key:
            assert param.get(k) ,InterruptedError(f"入参缺少{k}")

def make_result(param, head=200, msg=None, items=None, cls=False):
    if not items and head == 200:
        head = 600
    res = {
            "head":head,
            "msg":msg,
            "items":items
        }
    if cls:
        res = json.dumps(res, cls=NpEncoder)
    return res

def format_sse(data: str, event=None) -> str:
    msg = 'data: {}\n\n'.format(data)
    if event is not None:
        msg = 'event: {}\n{}'.format(event, msg)
    return msg    
    

@app.route('/chat', methods=['post'])
@clock
def get_chat_reponse():
    def decorate(generator):
        for item in generator:
            yield format_sse(json.dumps(item, ensure_ascii=False), 'delta')
    try:
        param = accept_param()
        task = param.get('task', 'chat')
        if task == 'chat':
            print('prompt: ' + param.get('prompt', ''))
            result = chat.run_prediction(
                param['history'],
                param.get('prompt', ''), 
                param.get('intentCode', 'default_code')
                )
    except AssertionError as err:
        logger.error(traceback.format_exc())
        result = make_result(param, head=601, msg=repr(err))
    except Exception as err:
        logger.error(traceback.format_exc())
        result = make_result(param, msg=repr(err))
    finally:
        return result


if __name__ == '__main__':
    ip, port = '0.0.0.0', 6500
    server = pywsgi.WSGIServer((ip, port), app)
    logger.debug(f"serve at {ip}:{port}")
    server.serve_forever()
    #uvicorn.run(app, host='0.0.0.0', port=6500, log_level="info")
