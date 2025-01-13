# -*- encoding: utf-8 -*-
"""
@Time    :   2025-1-9 16:48:19
@desc    :   Langfuse 提示词管理功能
@Author  :   车川
@Contact :   1163317515@qq.com
"""

import re
from typing import Dict, Any
from langfuse import Langfuse
from src.utils.Logger import logger

class LangfusePromptManager:
    """
    工具类：用于从 Langfuse 获取 Prompt，失败时回退到服务启动时预加载的 prompt_meta_data。
    """
    def __init__(self, langfuse_client: Langfuse, prompt_meta_data: Dict[str, Any]) -> None:
        """
        初始化 LangfusePromptManager。

        :param langfuse_client: Langfuse 客户端实例
        :param prompt_meta_data: 服务启动时预加载的 prompt 数据
        """
        self.langfuse = langfuse_client
        self.prompt_meta_data = prompt_meta_data

    async def get_formatted_prompt(self, event_code: str, prompt_vars: Dict[str, Any]) -> str:
        """
        获取并格式化提示词（优先从 Langfuse 获取，失败则回退到预加载数据）。

        :param event_code: 事件代码，用于标识 prompt
        :param prompt_vars: 动态替换变量
        :return: 格式化后的提示词
        """
        try:
            # 从 Langfuse 获取 Prompt 对象并进行插值编译
            prompt_obj = self.langfuse.get_prompt(event_code)

            # 提取 Langfuse 原始提示词内容
            prompt_template = prompt_obj.prompt

            # 替换动态占位符 {{variable}}
            compiled_prompt = prompt_obj.compile(**prompt_vars)

            # 还原静态占位符 {{{variable}}} 为 {variable}
            return self._restore_static_placeholders(prompt_template, compiled_prompt)

        except Exception as e:
            # 如果 Langfuse 调用失败，回退到预加载数据
            logger.info(f"[LangfusePromptManager] Langfuse 获取失败，改用预加载 prompt_meta_data: {e}")
            return self._get_formatted_prompt_from_preloaded(event_code, prompt_vars)

    def _get_formatted_prompt_from_preloaded(self, event_code: str, prompt_vars: Dict[str, Any]) -> str:
        """
        从预加载的 prompt_meta_data 获取并格式化提示词。

        :param event_code: 事件代码
        :param prompt_vars: 动态替换变量
        :return: 格式化后的提示词
        """
        event_item = self.prompt_meta_data.get("event", {}).get(event_code, {})
        base_prompt = event_item.get("description", f"[No preloaded prompt for {event_code}]")
        try:
            return base_prompt.format(**prompt_vars)  # 使用 format 替换变量
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt formatting.") from e

    def _restore_static_placeholders(self, template: str, compiled_prompt: str) -> str:
        """
        还原静态占位符 {{{variable}}} 为 {variable}。

        :param template: 原始提示词模板（含 {{{variable}}}）
        :param compiled_prompt: Langfuse 替换后的提示词
        :return: 最终提示词
        """
        # 匹配 {{{variable}}} 的静态占位符
        static_placeholder_pattern = re.compile(r"{{{(.*?)}}}")
        static_placeholders = static_placeholder_pattern.findall(template)

        # 按顺序将静态占位符替换回原内容
        for placeholder in static_placeholders:
            compiled_prompt = compiled_prompt.replace(f"{{{{{placeholder}}}}}", f"{{{placeholder}}}")
        return compiled_prompt

