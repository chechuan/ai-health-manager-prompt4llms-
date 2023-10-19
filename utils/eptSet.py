# -*- encoding: utf-8 -*-
'''
@Time    :   2023-03-27 11:14:32
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

class CustomError(Exception):
    """自定义Exception
    """
    def __init__(self, msg, code=12150):
        super().__init__(self)
        self.msg = msg
        self.code = code