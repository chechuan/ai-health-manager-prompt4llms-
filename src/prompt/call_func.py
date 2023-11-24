# -*- encoding: utf-8 -*-
'''
@Time    :   2023-10-20 13:45:09
@desc    :   call function script
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

import sys
from pathlib import Path
from typing import Any, AnyStr, Dict, List, Optional

import yaml
from requests import Session

sys.path.append(str(Path.cwd()))

from config.constrant import ParamServer
from src.utils.Logger import logger
from src.utils.module import req_prompt_data_from_mysql


class funcCall:
    headers: Dict = {"content-type": "application/json"}
    session = Session()
    api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
    param_server: object = ParamServer()

    def __init__(self, prompt_meta_data=None, env="local"):
        self.prompt_meta_data = prompt_meta_data if prompt_meta_data else req_prompt_data_from_mysql(env)
        self.funcmap = {}
        self.funcname_map = {i['name']: i['code'] for i in self.prompt_meta_data['tool'].values()}
        self.register_func("searchKnowledge", self.call_search_knowledge, "/chat/knowledge_base_chat")

    def register_func(self, func_name: AnyStr, func_call: Any, method: AnyStr) -> None:
        """
        """
        self.funcmap[func_name] = {"func": func_call, "method": method}
        logger.info(f"register {func_name} success")

    def call_search_knowledge(self, 
                              *args, 
                              knowledge_base_name="内科学",
                              local_doc_url=False,
                              model_name="Qwen-14B-Chat",
                              stream=False, 
                              score_threshold=0.5,
                              temperature=0.7,
                              top_k=3,
                              top_p=0.8,
                              prompt_name="default",
                              **kwargs) -> AnyStr:
        """
        """
        called_method = self.funcmap['searchKnowledge']['method']
        payload = {}
        payload["knowledge_base_name"] = knowledge_base_name
        payload["local_doc_url"] = local_doc_url
        payload["model_name"] = model_name
        payload["score_threshold"] = score_threshold
        payload["stream"] = stream
        payload["temperature"] = temperature
        payload["top_k"] = top_k
        payload["top_p"] = top_p
        payload["prompt_name"] = prompt_name

        payload['query'] = args[0]
        url = self.api_config['langchain']+called_method
        response = self.session.post(url, json=payload, headers=self.headers)
        msg = eval(response.text)
        if kwargs.get("verbose"):
            print(msg)
        ret = msg['answer']
        return ret

    def call_llm_with_search_engine(self, *args, **kwargs) -> AnyStr:
        """
        """
        called_method = "/chat/search_engine_chat"
        payload = self.param_server.llm_with_search_engine
        payload['query'] = args[0]['query']
        response = self.session.post(self.api_config['langchain']+called_method,
                                     json=payload,
                                     headers=self.headers)
        res_js = eval(response.text)
        if kwargs.get("verbose"):
            print(res_js)
        ret = res_js['answer']
        return ret
    
    def call_llm_with_graph(self, *args, **kwargs) -> AnyStr:
        """
        """
        called_method = "/v1/subgraph_qa"
        payload = self.param_server.llm_with_graph
        payload['messages'][0]['content'] = args[0]['query']
        response = self.session.post(self.api_config['graph']+called_method,
                                     json=payload,
                                     headers=self.headers)
        res_js = eval(response.text)
        if kwargs.get("verbose"):
            print(res_js)
        ret = res_js['result']
        return ret

    def _call(self, verbose=False, **kwargs):
        """"""
        function_call = kwargs['function_call']
        func_name = function_call["name"]
        arguments = function_call["arguments"]
        assert self.funcmap.get(func_name) is not None, "Unregistered function name"
        ret = self.funcmap[func_name]['func'](arguments, verbose=verbose)
        return ret

if __name__ == "__main__":
    funcall = funcCall()
    function_call = {'name': 'searchKnowledge', 'arguments': '后脑勺持续一个月的头疼'}
    funcall._call(function_call=function_call, verbose=True)
