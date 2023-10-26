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

from config.constrant import PLAN_MAP, ParamServer


class funcCall:
    headers: Dict = {"content-type": "application/json"}
    session = Session()
    api_config: Dict = yaml.load(open(Path("config","api_config.yaml"), "r"),Loader=yaml.FullLoader)['local']
    param_server: object = ParamServer()

    def __init__(self) -> None:
        self.funcmap: Dict[AnyStr, ] = {
            "chat_with_user": self.call_chat_with_user,
            "get_plan": self.call_get_plan,
            "llm_with_graph": self.call_llm_with_graph,
            "llm_with_documents": self.call_llm_with_documents,
            "llm_with_search_engine": self.call_llm_with_search_engine
        }
    
    def call_get_plan(self, *args, **kwargs) -> AnyStr:
        """
        """
        plan_key = args[0]['query']
        ret = PLAN_MAP.get(plan_key)
        return ret

    def call_chat_with_user(self, *args, **kwargs) -> AnyStr:
        """
        """
        query = args[0]['query']
        user_input = input(f"Question:{query}\nUser: ")
        return user_input

    def call_llm_with_documents(self, *args, **kwargs) -> AnyStr:
        """
        """
        called_method = "/chat/knowledge_base_chat"
        payload = self.param_server.llm_with_documents
        payload['query'] = args[0]['query']
        response = self.session.post(self.api_config['langchain']+called_method,
                                     json=payload,
                                     headers=self.headers)
        res_js = eval(response.text)
        if kwargs.get("verbose"):
            print(res_js)
        ret = res_js['answer']
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

    def _call(self, kwargs, verbose=False):
        """"""
        assert kwargs.get("name")
        assert kwargs.get("arguments")
        assert isinstance(eval(kwargs["arguments"]), dict)
        func_name = kwargs.get("name")
        arguments = eval(kwargs.get("arguments"))
        ret = self.funcmap[func_name](arguments, verbose=verbose)
        return ret

if __name__ == "__main__":
    funcall = funcCall()
    gen_args = {"name":"llm_with_graph", "arguments": "{'query': '继发性高血压的病因有哪些'}"}
    funcall._call(gen_args, verbose=True)
