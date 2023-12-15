# -*- encoding: utf-8 -*-
'''
@Time    :   2023-12-15 11:37:35
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ihmHealthData(BaseModel):
    date: str
    value: float
    
class healthBloodPressureTrendAnalysis(BaseModel):
    ihm_health_sbp: List[ihmHealthData]
    ihm_health_dbp: List[ihmHealthData]
    ihm_health_hr: List[ihmHealthData]