"""向量模型 + Chroma 向量库的统一入口。

全项目只有这里直接碰向量库和 embedding 模型，其它模块都用本文件暴露的函数，
这样「换模型 / 换库」只需要改一处。
"""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# 用余弦距离，便于把检索距离转成直观的相似度。全项目只在这里定义一次。
COLLECTION_METADATA = {"hnsw:space": "cosine"}

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# 模型 / 客户端都缓存起来：重复加载一次模型要好几秒，还会多吃一份显存
_EMBEDDING_CACHE: dict[tuple[str, bool, str], Any] = {}
_CLIENT_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------
# 向量模型
# --------------------------------------------------------------------------
def resolve_device() -> str:
    """把 EMBEDDING_DEVICE=auto 解析成真实设备。"""
    if config.EMBEDDING_DEVICE != "auto":
        return config.EMBEDDING_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # torch 没装 / 驱动异常都不该让程序起不来
        logger.debug("未检测到可用的 torch 设备，回退 CPU", exc_info=True)
    return "cpu"


def create_embeddings(
    model_name: str | None = None,
    *,
    local_only: bool | None = None,
    device: str | None = None,
):
    """创建（并缓存）embedding 模型。

    `normalize_embeddings=True` 让向量归一化，配合余弦距离更稳定。
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # 延迟导入，避免拖慢 CLI 启动

    model_name = model_name or config.EMBEDDING_MODEL
    local_only = config.EMBEDDING_LOCAL_ONLY if local_only is None else local_only
    device = device or resolve_device()
    key = (model_name, local_only, device)

    if key not in _EMBEDDING_CACHE:
        model_kwargs: dict[str, Any] = {
            "device": device,
            "trust_remote_code": config.EMBEDDING_TRUST_REMOTE_CODE,
        }
        if local_only:
            model_kwargs["local_files_only"] = True
        logger.info("加载 embedding 模型：%s（设备 %s）", model_name, device)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": config.EMBEDDING_BATCH_SIZE,
            },
        )
        # max_seq_length 不是 SentenceTransformer 的构造参数，只能在实例上设置。
        # 限制长度有两个好处：省显存、避免长文档把关键句「稀释」掉。
        client = getattr(embeddings, "_client", None)
        if client is not None and hasattr(client, "max_seq_length"):
            client.max_seq_length = config.EMBEDDING_MAX_LENGTH
        _EMBEDDING_CACHE[key] = embeddings
    return _EMBEDDING_CACHE[key]


def clear_embedding_cache() -> None:
    """换模型 / 换设备后调用，释放旧模型。"""
    _EMBEDDING_CACHE.clear()


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return create_embeddings().embed_documents(list(texts))


def embed_query(text: str) -> list[float]:
    """查询侧向量化。

    注意：只给"查询"加前缀，库里文档不加 —— 非对称检索里这是标准做法
    （Qwen3-Embedding 等模型的 instruct 用法就依赖它）。
    """
    prefix = config.EMBEDDING_QUERY_PREFIX
    return create_embeddings().embed_query(f"{prefix}{text}" if prefix else text)


# --------------------------------------------------------------------------
# Chroma 存取
# --------------------------------------------------------------------------
def get_client(path: Path | None = None):
    """按目录缓存 PersistentClient（重复创建会互相锁文件）。"""
    import chromadb

    target = Path(path or config.CHROMA_DIR)
    target.mkdir(parents=True, exist_ok=True)
    key = str(target.resolve())
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = chromadb.PersistentClient(path=key)
    return _CLIENT_CACHE[key]


def get_collection(*, create: bool = False, name: str | None = None, client=None):
    """拿到原生 chromadb collection（embedding 一律由我们显式传入）。"""
    client = client or get_client()
    collection_name = name or config.COLLECTION_NAME
    if create:
        return client.get_or_create_collection(
            collection_name,
            metadata=COLLECTION_METADATA,
            embedding_function=None,
        )
    return client.get_collection(collection_name, embedding_function=None)


def collection_exists(name: str | None = None) -> bool:
    try:
        get_collection(name=name)
        return True
    except Exception:
        return False


def get_vectorstore():
    """LangChain 版 Chroma（需要和 LangChain 生态对接时用）。"""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embedding_function=create_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def distance_to_similarity(distance: float) -> float:
    """余弦距离 -> [0, 1] 相似度。"""
    return max(0.0, min(1.0, 1.0 - float(distance)))


# 旧名字，保持向后兼容
is_similarity = distance_to_similarity


# --------------------------------------------------------------------------
# 管理与统计
# --------------------------------------------------------------------------
def collection_stats(name: str | None = None, directory: Path | None = None) -> dict[str, Any]:
    """返回块数、集合名、库目录等，供界面展示。"""
    stats: dict[str, Any] = {
        "collection": name or config.COLLECTION_NAME,
        "directory": str(directory or config.CHROMA_DIR),
        "count": 0,
        "exists": False,
    }
    try:
        collection = get_collection(name=name)
        stats["count"] = collection.count()
        stats["exists"] = True
        stats["metadata"] = dict(collection.metadata or {})
    except Exception as exc:
        stats["error"] = str(exc)
    return stats


def collection_documents(name: str | None = None) -> tuple[list[str], list[str], list[dict]]:
    """取出全部块的 (ids, 文本, 元数据)，BM25 建索引要用。"""
    collection = get_collection(name=name)
    result = collection.get(include=["documents", "metadatas"])
    ids = list(result.get("ids") or [])
    documents = list(result.get("documents") or [])
    metadatas = [dict(item or {}) for item in (result.get("metadatas") or [])]
    return ids, documents, metadatas


def list_collections() -> list[dict[str, Any]]:
    """列出库里所有 collection 及块数，方便发现"建错集合"这种坑。"""
    client = get_client()
    items: list[dict[str, Any]] = []
    for collection in client.list_collections():
        name = getattr(collection, "name", str(collection))
        try:
            handle = get_collection(name=name)
            items.append({"name": name, "count": handle.count()})
        except Exception as exc:
            items.append({"name": name, "count": None, "error": str(exc)})
    return sorted(items, key=lambda item: item["name"])


def delete_collection(name: str | None = None) -> None:
    """删除整个 collection（不存在也不报错）。"""
    try:
        get_client().delete_collection(name or config.COLLECTION_NAME)
    except Exception:
        logger.debug("collection 不存在或删除失败：%s", name, exc_info=True)


def delete_source(source: str, name: str | None = None) -> int:
    """按来源文件名删块，用于「只重跑一个文档」的增量入库。"""
    collection = get_collection(name=name)
    existing = collection.get(where={"source": source}, include=[])
    ids = list(existing.get("ids") or [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def upsert_chunks(
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    *,
    name: str | None = None,
    batch_size: int = 256,
) -> int:
    """批量写入。用 upsert 而不是 add：重复入库不会产生重复块。"""
    collection = get_collection(create=True, name=name)
    embeddings = create_embeddings().embed_documents(list(documents))
    total = 0
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=list(ids[start:end]),
            documents=list(documents[start:end]),
            metadatas=list(metadatas[start:end]),
            embeddings=embeddings[start:end],
        )
        total += len(ids[start:end])
    return total


def prune_orphan_segments(dry_run: bool = True) -> list[str]:
    """清理向量库里没有登记在 sqlite 的残留目录。

    Chroma 删除/重建 collection 时会留下旧的 HNSW 段目录，久了能占几百 MB。
    这里只删除"形如 UUID 且不在 segments 表里"的目录，绝不碰其它文件。
    """
    directory = Path(config.CHROMA_DIR)
    database = directory / "chroma.sqlite3"
    if not database.exists():
        return []

    connection = sqlite3.connect(str(database))
    try:
        known = {row[0] for row in connection.execute("SELECT id FROM segments")}
    finally:
        connection.close()

    orphans: list[str] = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or not _UUID_RE.match(child.name):
            continue
        if child.name in known:
            continue
        orphans.append(child.name)
        if not dry_run:
            shutil.rmtree(child, ignore_errors=True)
            logger.info("已删除孤儿段目录：%s", child.name)
    return orphans
