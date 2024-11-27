# -*- encoding: utf-8 -*-
"""
@Time    :   2023-03-27 10:23:48
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""
import os
import sys
from functools import partial
from pathlib import Path

from loguru import logger

logger.remove()


def filter_function(appid, record):
    record["file"].path = (
        record["file"].path.split(f"{appid}/")[1]
        if f"{appid}/" in record["file"].path
        else record["file"].path
    )
    return True  # 返回True，表示所有的日志都应被记录

def monitor_filter(record):
    """过滤监控日志"""
    return record["extra"].get("source", "default") == "monitor"


class Logging:
    def __init__(self, appid="0", console_level="TRACE", file_level="DEBUG", monitor_level="INFO"):
        """日志记录模块
        :param appid: 应用id
        :param file_level: 保存文件日志logger级别
        """
        # global logger
        self.LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        self.MONITOR_FORMAT = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<green>[{extra[source]}]</green> "
            "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        LOG_PATH = Path(__file__).parents[2] / "logs"
        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)

        logger.add(
            sink=sys.stderr,
            level=console_level,
            format=self._get_console_format,
            backtrace=False,
            diagnose=False,
            filter=partial(filter_function, appid),
        )

        ENV = os.environ.get("ENV", "")
        logger.add(
            sink=LOG_PATH / f"{appid}.{ENV}.log",
            level=file_level,
            format=self.LOG_FORMAT,
            rotation="50 MB",
            colorize=False,
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )

        # 添加监控日志（独立文件）
        logger.add(
            sink=LOG_PATH / f"{appid}.monitor.log",
            level=monitor_level,
            format=self.MONITOR_FORMAT,
            rotation="10 MB",
            encoding="utf-8",
            filter=monitor_filter,  # 仅处理监控日志
        )

        self.logger = logger

    def _get_console_format(self, record):
        """根据记录类型返回不同的格式"""
        if record["extra"].get("source", "default") == "monitor":
            return self.MONITOR_FORMAT
        else:
            return self.LOG_FORMAT


logger = Logging(appid="ai-health-manager-prompt4llms").logger

if __name__ == "__main__":
    logger.trace("log level trace.")
    logger.debug("log level debug.")
    logger.info("log level info.")
    logger.success("log level success.")
    logger.warning("log level warning.")
    logger.error("log level error.")
    logger.critical("log level critical.")
