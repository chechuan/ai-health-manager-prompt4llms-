from typing import Dict, List

from fastapi import Body, File, Query, UploadFile
from langchain.docstore.document import Document
from pydantic import BaseModel, Field, Json


class CreateKnowledgeBaseRequest(BaseModel):
    knowledge_base_name: str = Field(..., description="知识库名称")
    vector_store_type: str = Field("faiss", description="向量库类型")
    embed_model: str = Field("bce-embedding-base-v1", description="嵌入模型")


class UploadDocsRequest(BaseModel):
    knowledge_base_name: str = Field(..., description="知识库名称", examples=["samples"])
    override: bool = Field(False, description="覆盖已有文件")
    to_vector_store: bool = Field(True, description="上传文件后是否进行向量化")
    chunk_size: int = Field(250, description="知识库中单段文本最大长度")
    chunk_overlap: int = Field(50, description="知识库中相邻文本重合长度")
    zh_title_enhance: bool = Field(False, description="是否开启中文标题加强")
    metadata: Json = Body(
        {},
        description="查询可以根据metadata进行过滤，仅支持一级键, 避免`id`, `source`, `doc_ids`保留字",
    )
    docs: Json = Field(
        {},
        description="自定义的docs，需要转为json字符串",
        examples=[{"test.txt": [Document(page_content="custom doc")]}],
    )
    not_refresh_vs_cache: bool = Field(False, description="暂不保存向量库（用于FAISS）")


class SearchDocsRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    knowledge_base_name: str = Field(..., description="知识库名称", examples=["samples"])
    top_k: int = Field(10, description="返回结果数量")
    score_threshold: float = Field(0.5, description="返回结果分数阈值")
    metadata: Dict = Body(
        {},
        description="查询可以根据metadata进行过滤，仅支持一级键, 避免`id`, `source`, `doc_ids`保留字",
    )
    use_reranker: bool = Field(True, description="是否使用reranker")
    rerank_threshold: float = Field(0.35, description="reranker分数阈值, 值越高，相关性越高")
    rerank_top_k: int = Field(20, description="reranker返回结果数量")
