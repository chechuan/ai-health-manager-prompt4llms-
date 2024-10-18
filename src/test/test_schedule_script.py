# -*- encoding: utf-8 -*-
"""
@Time    :   2024-01-05 09:53:59
@desc    :   XXX
@Author  :   宋昊阳
@Contact :   1627635056@qq.com
"""


import argparse
import copy_table
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path.cwd()))
import pandas as pd
import requests
from loguru import logger
from sseclient import SSEClient

from src.utils.Logger import logger
from src.utils.module import curr_time


def dumpJS(obj, ensure_ascii=True):
    return json.dumps(obj, ensure_ascii=ensure_ascii)


class ScheduleManagerText:
    payload: dict = {
        "orgCode": "sf",
        "customId": "test_songhaoyang",
        "intentCode": "schedule_manager",
        "history": [{"msgId": "1908745280", "role": "1", "content": "3分钟后提醒我喝牛奶"}],
        "backend_history": [],
        "debug": True,
    }
    headers: dict = {"Content-Type": "application/json"}
    session = requests.Session()
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=15)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.load_data()

    def load_data(self):
        """
        Load data from csv/xlsx file.
        """
        self.save_path = Path(self.args.file_path).parent / f"{Path(self.args.file_path).stem}-{self.args.sheet_name}-{curr_time()}.xlsx"
        if self.args.file_path.endswith(".csv"):
            self.df_raw = pd.read_csv(self.args.file_path, encoding="gbk")
        elif self.args.file_path.endswith(".xlsx"):
            if self.save_path.exists():
                self.df_raw = pd.read_excel(self.save_path, dtype=str)
            else:
                if not self.args.sheet_name:
                    self.df_raw = pd.read_excel(self.args.file_path, dtype=str)
                else:
                    self.df_raw = pd.read_excel(self.args.file_path, sheet_name=self.args.sheet_name, dtype=str)
        else:
            raise ValueError(f"Unsupported file type: {self.args.file_path}")
        logger.info(f"DataFrame Columns: {list(self.df_raw.columns)}")
        logger.info(f"DataFrame Shape: {self.df_raw.shape}")

    def get_intent(self, query) -> str:
        """
        Get the intent of the query.
        """
        payload = {"history": [{"role": 1, "content": query}]}
        try:
            response = self.session.post(self.args.url + "/intent/query", headers=self.headers, data=dumpJS(payload)).json()
            intent_code = response["items"]["intentCode"]
        except Exception as e:
            try:
                response = self.session.post(self.args.url + "/intent/query", headers=self.headers, data=dumpJS(payload)).json()
                intent_code = response["items"]["intentCode"]
            except Exception as e:
                intent_code = "schedule_manager"
        return intent_code

    # @clock
    def get_response(self, index, query):
        payload = copy.deepcopy(self.payload)
        intent_code = self.get_intent(query)
        payload["intentCode"] = intent_code
        payload["history"][0]["content"] = query
        response = self.session.post(
            self.args.url + "/chat/complete", headers=self.headers, data=dumpJS(payload), stream=True
        )
        client = SSEClient(response)
        time_desc, reply = [], ""
        for event in client.events():
            if event.data == "[DONE]":
                continue
            else:
                js = json.loads(event.data)
            if js["end"]:
                for i in js["mid_vars"]:
                    if i["key"] == "parse_time_desc":  # 创建日程字段
                        time_desc = i["output_text"]["except_result"]
                        if time_desc:
                            logger.debug(f"Time desc: {time_desc}")
                            self.df_raw.loc[index, "_event"] = time_desc[0][0]
                            self.df_raw.loc[index, "_date"] = time_desc[0][2][:10]
                            self.df_raw.loc[index, "_remind_time"] = time_desc[0][2][11:]
                            payload = {
                                "customId": payload["customId"],
                                "orgCode": payload["orgCode"],
                                "taskName": time_desc[0][0],
                                "cronDate": time_desc[0][2],
                                "taskType": "reminder",
                                "intentCode": "CANCEL",
                            }
                            response = self.session.post(
                                self.args.ai_backend_url + "/manage", headers=self.headers, data=json.dumps(payload)
                            ).json()
                            logger.success(f"Cancel task response: {response}")
                    if i["key"] == "confirm_query_time_range":  # 查询日程 字段
                        time_range = i["output_text"]
                        self.df_raw.loc[index, "_date"] = " ~ ".join(set([i[:10] for i in time_range.values()]))
                        self.df_raw.loc[index, "_start_time"] = time_range["startTime"][-8:]
                        self.df_raw.loc[index, "_end_time"] = time_range["endTime"][-8:]
                    if i["key"] == "调用查询日程接口":  # 删除日程 字段
                        time_range = [i["input_text"]["startTime"], i["input_text"]["endTime"]]
                        self.df_raw.loc[index, "_date"] = " ~ ".join(set([i[:10] for i in time_range]))
                        self.df_raw.loc[index, "_start_time"] = i["input_text"]["startTime"][-8:]
                        self.df_raw.loc[index, "_end_time"] = i["input_text"]["endTime"][-8:]
                reply = js["message"]
        self.df_raw.loc[index, "_intent_code"] = intent_code
        self.df_raw.loc[index, "_reply"] = reply
        logger.debug(f"Query: {query}, Intent: {intent_code}, Reply: {reply}, Index: {index}")
        return index, reply, time_desc

    def run(self):
        """
        Run the test cases.
        """        
        all_task = []
        for index, row in self.df_raw.iterrows():
            if not pd.isnull(self.df_raw.loc[index, "_intent_code"]): 
                logger.trace(f"Skip index: {index}, intent code: {self.df_raw.loc[index, '_intent_code']}")
                continue
            query = row["测试用例"]
            if self.args.debug:
                _, reply, time_desc = self.get_response(index, query)
            else:
                all_task.append(self.executor.submit(self.get_response, index, query))
                if index % 25 == 0:
                    for task in as_completed(all_task):
                        index, reply, time_desc = task.result()
                    self.df_raw.to_excel(self.save_path, index=False)
                    all_task = []
                    logger.info(f"save file index up to {index}.")
        if not self.args.debug:
            for task in as_completed(all_task):
                index, reply, time_desc = task.result()
            self.df_raw.to_excel(self.save_path, index=False)
            logger.info(f"save file index up to {index}.")
        self.df_raw.to_excel(self.save_path, index=False)
        logger.success(f"save file index up to {index}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        type=str,
        default=Path(".cache/日程管理测试集v2-20240105.csv"),
        help="file path of the test cases",
    )
    parser.add_argument("--sheet-name", type=str, default="日程创建", help="sheet name of the test cases")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:26928/chat/complete", help="url of the chatbot")
    parser.add_argument(
        "--ai-backend-url",
        type=str,
        default="https://gate-dev.op.laikang.com/aihealthmanager-dev/alg-api/schedule",
        help="url of the ai-backend",
    )
    parser.add_argument("--debug", type=bool, default=False, help="debug mode")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    app = ScheduleManagerText(args)
    app.run()

    # sheet_name = "日程查询" / "日程创建" / "日程取消"

    """
    python src/test/test_schedule_script.py \
        --file-path /home/tico/workspace/ai-health-manager-prompt4llms/.cache/日程管理测试用例20240219.xlsx \
        --url http://127.0.0.1:26928 \
        --sheet-name 日程删除 \
        --ai-backend-url https://gate-dev.op.laikang.com/aihealthmanager-dev/alg-api/schedule
    """
