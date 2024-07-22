# -*- encoding: utf-8 -*-
"""
@Time    :   2024-05-10 14:42:16
@desc    :   实现验证相关方法、资源
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""
import copy
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import AnyStr, BinaryIO, Dict, List, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from pydantic import Field
from requests import Session
from rich import print
from tqdm import tqdm

sys.path.append(Path.cwd().as_posix())

from loguru import logger

from protocal import CreateKnowledgeBaseRequest, SearchDocsRequest, UploadDocsRequest

load_dotenv()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--kb-name", type=str, default="valid", help="验证知识库名称")
    parser.add_argument(
        "--vector-store-type", type=str, default="faiss", help="向量库类型"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="bce-embedding-base-v1",
        help="默认的embedding模型",
    )
    parser.add_argument(
        "--data-dirs",
        type=Union[List, str],
        default=[
            ".cache/知识验证1期/验证1知识资源",
            ".cache/知识验证1期/验证2知识资源",
        ],
        help="要处理及向量化的知识原始文件目录",
    )
    parser.add_argument(
        "--valid-data-path",
        type=str,
        default=".cache/知识验证1期/大模型1期验证测试问题(1).xlsx",
        help="测试集文件",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=".cache/kb_valid.xlsx",
        help="最终输出文件的存储路径",
    )
    args = parser.parse_args()
    return args


class ShareResource:
    base_kb_url: str = os.getenv("BASE_KB_URL", "http://10.228.67.99:26925")
    methods_map: Dict = {
        "get_kb_list": "/knowledge_base/list_knowledge_bases",
        "create_kb": "/knowledge_base/create_knowledge_base",
        "upload_docs": "/knowledge_base/upload_docs",
        "search_docs": "/knowledge_base/search_docs",
        "list_files": "/knowledge_base/list_files",
    }
    args: ArgumentParser = parse_args()
    save_path = Path(args.save_path)

    def get_session(
        self,
    ):
        return Session()

    @classmethod
    def get_method_url(cls, key: str):
        assert key in cls.methods_map, "method not found"
        return cls.base_kb_url + cls.methods_map[key]

    def get_kbs(self) -> List:
        """获取知识库列表"""
        with self.get_session() as session:
            response = session.get(self.get_method_url("get_kb_list")).json()
        return response["data"]

    def list_files(self) -> List:
        """获取知识库中的文件列表"""
        with self.get_session() as session:
            response = session.get(
                self.get_method_url("list_files"),
                params={"knowledge_base_name": self.args.kb_name},
            ).json()
        return response["data"]

    def create_kb(self):
        """创建知识库"""
        data = CreateKnowledgeBaseRequest(
            knowledge_base_name=self.args.kb_name,
            vector_store_type=self.args.vector_store_type,
            embed_model=self.args.embed_model,
        )
        with self.get_session() as session:
            response = session.post(
                self.get_method_url("create_kb"),
                json=data.dict(),
            ).json()
        if response.get("code") != 200:
            print(response)
            return
        else:
            logger.info(f"Create KB {self.args.kb_name} success.")
        return response

    def __compose_files__(
        self, file_path: Path
    ) -> List[Tuple[AnyStr, Tuple[AnyStr, BinaryIO, AnyStr]]]:
        match file_path.suffix:
            case ".docx":
                suffix = "vnd.openxmlformats-officedocument.wordprocessingml.document"
            case ".xls":
                suffix = "vnd.ms-excel"
            case ".xlsx":
                suffix = "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            case ".pdf":
                suffix = "pdf"
            case ".txt":
                suffix = "plain"
            case ".json":
                suffix = "plain"
            case _:
                logger.warning(f"{file_path.name} suffix not support.")
                return []
        files = [
            (
                "files",
                (
                    file_path.name,
                    open(file_path, "rb"),
                    f"application/{suffix}",
                ),
            )
        ]
        return files

    def __preprocess_file_format__(self, all_file_paths: List[Path]) -> List[Path]:
        """预处理文件 格式转换, 名称调整"""
        _all_file_paths = []
        for file_path in all_file_paths:
            _file_name = (
                file_path.name.replace(" ", "-").replace("(", "（").replace(")", "）")
            )
            if _file_name != file_path.name:
                _file_path = file_path.parent / _file_name
                file_path.rename(_file_path)
                logger.debug(f"rename file from {file_path.name} to {_file_name}")
                file_path = _file_path
            _all_file_paths.append(file_path)
        return _all_file_paths

    def upload_docs(self):
        if isinstance(self.args.data_dirs, str):
            all_file_paths = self.get_all_file_paths(self.args.data_dirs)
        elif isinstance(self.args.data_dirs, list):
            all_file_paths = []
            for dir_path in self.args.data_dirs:
                all_file_paths.extend(self.get_all_file_paths(dir_path))
        else:
            raise ValueError("upload_dir must be str or list")

        all_file_paths = self.__preprocess_file_format__(all_file_paths)[:5]

        exist_files = self.list_files()
        for file_path in tqdm(
            all_file_paths,
            desc="upload files && vectorizing...",
            total=len(all_file_paths),
        ):
            if file_path.suffix == ".doc":
                # 使用libreoffice将doc转为docx
                try:
                    command = (
                        f"libreoffice --headless --convert-to docx {file_path.resolve().as_posix()} --outdir {file_path.parent.resolve().as_posix()}"
                        f" && rm {file_path.resolve().as_posix()}"
                    )
                    subprocess.run(command, shell=True)
                    logger.info(f"transfer {file_path.name} to docx")
                    file_path = file_path.parent / file_path.stem / ".docx"
                except Exception as e:
                    logger.error(e)
            if file_path.name in exist_files:
                logger.debug(f"{file_path.name} already exist.")
                continue
            files = self.__compose_files__(file_path)
            if not files:
                continue
            data = UploadDocsRequest(
                knowledge_base_name=self.args.kb_name,
                to_vector_store=True,
                chunk_size=500,
                chunk_overlap=50,
                zh_title_enhance=True,
                override=True,
            )
            with self.get_session() as session:
                response = session.post(
                    self.get_method_url("upload_docs"),
                    data=data.dict(),
                    files=files,
                    timeout=36000,
                ).json()
                if response["code"] == 200:
                    logger.debug(f"upload {file_path.name} success.")

    @staticmethod
    def get_all_file_paths(dir_path: Union[str, Path]) -> List[Path]:
        file_paths = [
            file_path
            for file_path in Path(dir_path).glob("**/*")
            if file_path.is_file()
            and file_path.suffix not in [".json", ".xls", ".xlsx"]
        ]
        return file_paths

    def load_valid_dataset(self) -> pd.DataFrame:
        if Path(self.args.valid_data_path).exists():
            ds = pd.read_excel(self.args.valid_data_path, sheet_name=None)
            columns = [i.columns for i in ds.values()]
            if all([all(x == columns[0]) for x in columns]):
                df = pd.concat(list(ds.values()), axis=0)
                logger.info(f"load valid dataset, total nums: {len(df)}.")
        return df

    def match_with_knowledge(self, ds: pd.DataFrame) -> List[Dict]:
        """
        使用知识库对数据集中的每一行进行匹配
        :param ds: DataFrame, 包含要匹配的数据
        :param knowledge_base_name: str, 知识库的名称
        :return: DataFrame, 匹配结果
        """
        _result = []
        for _, row in tqdm(ds.iterrows()):
            query = row["问题"]  # 替换为实际需要匹配的列名
            data = SearchDocsRequest(
                query=query,
                knowledge_base_name=self.args.kb_name,
                top_k=20,
                score_threshold=0.8,
                metadata={},
                use_reranker=True,
                rerank_threshold=0.35,
                rerank_top_k=20,
            )
            with self.get_session() as session:
                response = session.post(
                    self.get_method_url("search_docs"), json=data.dict()
                ).json()
            if response:
                row["匹配结果"] = json.dumps(response, ensure_ascii=False)
            _result.append(row.to_dict())
        return pd.DataFrame(_result)

    def process_excel_at_end(self, precision: int = 3):
        """最终处理excel格式 -> 交付"""
        df = pd.read_excel(self.save_path, index_col=0)
        _final_result = []
        for idx, row in df.iterrows():
            if not pd.isnull(row["匹配结果"]):
                matched = json.loads(row["匹配结果"])
                matched = sorted(matched, key=lambda x: x["rerank_score"], reverse=True)
            for _idx, i in enumerate(matched):
                _row = copy.deepcopy(row).to_dict()
                _row["匹配结果"] = i["page_content"]
                _row["是否相关(0/1)"] = ""
                _row["相关程度(满分1)"] = ""
                _row["rerank_score(满分1)"] = round(i["rerank_score"], precision)
                _row["score(满分1)"] = round(i["score"], precision)
                if idx == 0 and _idx == 0:
                    logger.trace("Record example")
                    print(_row)
                _final_result.append(_row)
        final_save_path = self.save_path.parent / (
            self.save_path.stem + "_final" + self.save_path.suffix
        )
        pd.DataFrame(_final_result).to_excel(final_save_path)
        logger.success(f"save result to {final_save_path}")
