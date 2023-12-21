# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-21 10:34:32
@desc    :   角色扮演llm
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from src.utils.module import initAllResource


class rolePlayModel:
    def __init__(self, gsr: initAllResource):
        self.gsr = gsr
        self.__extract_role_message__()
        
    def __extract_role_message__(self, ):
        """提取mysql中预定义的角色信息"""
        role_base_data = self.gsr.prompt_meta_data['role_play']
        for role_name, role_info in role_base_data.items():
            for key, value in role_info.items():
                if key == "variables" and value:
                    role_info[key] = json.loads(value)
                if key == "executor":
                    role_info[key] = [i for i in re.split(",|，", value, re.S) if i != role_name] if value else []
        self.role_base_data = role_base_data
    
    def __get_play_roles__(self, ):
        """提供支持的角色   /get方法
        """
        return list(self.role_base_data.keys())
    
    def __get_user_can_play_roles__(self, param):
        """获取用户可以扮演的角色   /post

        Args:
            role [Str]: 已选的要聊天的角色

            example:
                ```json
                {"role": "白展堂"}
                ```
        return:
            user_can_play_roles [List]: 用户可以选择的角色
        """
        user_can_play_roles = self.role_base_data.get(param.get('role', '白展堂'), [])
        return user_can_play_roles

    def __chat__(self, ):
        """角色聊天, 流式yield生成结果
        - Args:

        

        """
        ...

if __name__ == "__main__":
    rolellm = rolePlayModel(initAllResource())
    param = {
        "character": "白展堂"
    }