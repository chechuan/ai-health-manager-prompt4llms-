# -*- encoding: utf-8 -*-
"""
@Time    :   2024-05-09 10:41:27
@desc    :   知识工程应用方案验证1期,数据资源
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
import sys
from pathlib import Path

sys.path.append(Path.cwd().as_posix())
from loguru import logger

from utils import *


def main():
    if not obj.args.kb_name in obj.get_kbs():  # 指定知识库不存在P
        obj.create_kb()
    else:
        logger.debug(f"KB named {obj.args.kb_name} already exists.")
    if not obj.save_path.exists():
        obj.upload_docs()
        
        ds = obj.load_valid_dataset()
        ds_matched = obj.match_with_knowledge(ds)
        logger.info(f"Save to {obj.save_path}.")
        ds_matched.to_excel(obj.save_path)
    else:
        obj.process_excel_at_end()


if __name__ == "__main__":
    obj = ShareResource()
    main()
