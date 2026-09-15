'''向量化与向量库访问的共享封装，供 ingest / query 复用。'''
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config

# 用余弦距离，便于把检索距离转换成直观的相似度
_COLLECTION_METADATA = {"hnsw:space": "cosine"}

# 缓存 embedding 模型实例：同一 (模型, 是否本地) 只加载一次，
# ingest 和 query 复用同一个实例，避免重复加载模型（省时间省内存）。
# 切换模型 / 本地模式时，key 变 -> 自动重建，不会误用旧模型。
_EMBEDDING_CACHE: dict = {}


def create_embeddings() -> HuggingFaceEmbeddings:
    '''创建（并缓存）embedding 模型。

    normalize_embeddings=True 让向量归一化，配合余弦相似度更稳定。
    '''
    key = (config.EMBEDDING_MODEL, config.EMBEDDING_LOCAL_ONLY)
    if key not in _EMBEDDING_CACHE:
        model_kwargs = dict(config.EMBEDDING_MODEL_KWARGS)
        if config.EMBEDDING_LOCAL_ONLY:
            model_kwargs["local_files_only"] = True
        _EMBEDDING_CACHE[key] = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDING_CACHE[key]


def get_vectorstore() -> Chroma:
    '''打开（或创建）Chroma 向量库。'''
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embedding_function=create_embeddings(),
        collection_metadata=_COLLECTION_METADATA,
    )


def is_similarity(score: float) -> float:
    '''把 Chroma 余弦距离转换为 [0, 1] 的相似度。'''
    return max(0.0, min(1.0, 1.0 - score))
