# -*- encoding: utf-8 -*-
"""
@Time    :   2025-1-9 16:48:19
@desc    :   Langfuse 提示词管理功能
@Author  :   车川
@Contact :   1163317515@qq.com
"""

import re
import json
from typing import Dict, Any
from langfuse import Langfuse
from src.utils.Logger import logger

class LangfusePromptManager:
    """
    工具类：用于从 Langfuse 获取 Prompt，失败时回退到服务启动时预加载的 prompt_meta_data。
    """
    def __init__(self, langfuse_client: Langfuse, prompt_meta_data: Dict[str, Any] = None) -> None:
        """
        初始化 LangfusePromptManager。

        :param langfuse_client: Langfuse 客户端实例
        :param prompt_meta_data: 服务启动时预加载的 prompt 数据
        """
        self.langfuse = langfuse_client
        self.prompt_meta_data = prompt_meta_data

    async def get_formatted_prompt(self, event_code: str, prompt_vars: Dict[str, Any], default_prompt: str = None) -> str:
        """
        获取并格式化提示词（优先从 Langfuse 获取，失败则回退到默认提示词，若未提供默认则回退到预加载数据）。

        :param event_code: 事件代码，用于标识 prompt
        :param prompt_vars: 动态替换变量
        :param default_prompt: 如果 Langfuse 获取失败则使用的默认提示词（可选）
        :return: 格式化后的提示词
        """
        try:
            # 从 Langfuse 获取 Prompt 对象并进行插值编译
            prompt_obj = self.langfuse.get_prompt(event_code)

            # 提取 Langfuse 原始提示词内容
            prompt_template = prompt_obj.prompt

            # 替换动态占位符 {{variable}} 并将 prompt_vars 变量直接插入
            compiled_prompt = prompt_obj.compile(**prompt_vars)

            logger.info(f"[LangfusePromptManager] 提示词成功从 Langfuse 获取，prompt name: {event_code} prompt version {prompt_obj.version}")

            # 还原静态占位符 {{{variable}}} 为 {variable}
            return self._restore_static_placeholders(prompt_template, compiled_prompt)

        except Exception as e:
            # 如果 Langfuse 获取失败，尝试使用默认提示词（若提供）
            if default_prompt:
                logger.info(f"[LangfusePromptManager] 获取 Langfuse 提示词失败，使用提供的默认提示词: {e}")
                try:
                    return default_prompt.format(**prompt_vars)  # 使用 format 替换变量
                except KeyError as e:
                    raise ValueError(f"Missing variable {e} for prompt formatting.") from e
            else:
                # 如果没有提供默认提示词，则回退到预加载数据
                if self.prompt_meta_data:
                    logger.info(f"[LangfusePromptManager] Langfuse 获取失败，回退到预加载 prompt_meta_data: {e}")
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

        # 如果 compiled_prompt 是一个 JSON 字符串，确保它只有一个层次的花括号
        try:
            # 尝试解析编译后的提示词作为一个 JSON 对象
            json_obj = json.loads(compiled_prompt)
            # 返回正确格式化的 JSON 字符串
            return json.dumps(json_obj, ensure_ascii=False)
        except json.JSONDecodeError:
            # 如果不是一个有效的 JSON 字符串，则返回原始字符串
            return compiled_prompt

