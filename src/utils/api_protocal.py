# -*- encoding: utf-8 -*-
"""
@Time    :   2023-12-15 11:37:35
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

from httpx import options
from pydantic import BaseModel, Field


class ihmHealthData(BaseModel):
    date: str
    value: float


class healthBloodPressureTrendAnalysis(BaseModel):
    ihm_health_sbp: List[ihmHealthData]
    ihm_health_dbp: List[ihmHealthData]
    ihm_health_hr: List[ihmHealthData]


class RolePlayChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None
    schedule: Optional[List] = None


class RolePlayRequest(BaseModel):
    model: str
    messages: List[RolePlayChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class AigcFunctionsRequest(BaseModel):
    url: Optional[str] = None
    intentCode: str
    prompt: Optional[str] = None
    options: Optional[List[str]] = None
