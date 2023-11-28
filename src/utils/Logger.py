# -*- encoding: utf-8 -*-
'''
@Time    :   2023-03-27 10:23:48
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
'''
import socket
import sys
from pathlib import Path

from loguru import logger

logger.remove()

class Logging:
    def __init__(self, appid = "0", console_level='TRACE', file_level = "DEBUG"):
        """日志记录模块
        :param appid: 应用id
        :param file_level: 保存文件日志logger级别
        """
        logger.add(sink=sys.stderr, level=console_level, backtrace=True, diagnose=True)
        hostname = socket.gethostname()
        try:
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True, parents=True)
        except Exception as error:
            logger.exception(error)
        
        log_path = log_dir.joinpath(f"{appid}.log")
        log_path.touch(exist_ok=True)
        
        format = "[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{level}] [{process}] {module}:{name}:{line} {message}"
        logger.add(
            sink=log_path,
            format=format, 
            level=file_level, 
            encoding="utf-8", 
            rotation="00:00", 
            retention="10 days", 
            compression="gz",
            backtrace=True, 
            diagnose=True
            )
        
        self.logger = logger

logger = Logging(appid="aimp-algo-health-manager-model").logger

if __name__ == "__main__":
    logger.trace("log level trace.")
    logger.debug("log level debug.")
    logger.info("log level info.")
    logger.success("log level success.")
    logger.warning("log level warning.")
    logger.error("log level error.")
    logger.critical("log level critical.")
