"""入库编排：来源解析 -> 加载 -> 切章节 -> 分块 -> 向量化 -> 写 Chroma。

支持三种入库方式：
- 单文档：`ingest_documents()`（沿用 DOC_PATH）
- 多来源：`ingest_sources("./docs, 论文.pdf")`，目录会递归扫描
- 增量：重复入库同一文件会先删掉它的旧块再写新块（按 `source` 元数据），
  所以改了文档重新跑一次即可，不会出现新旧内容混在一起。

加新格式只需要在 `loaders.py` 里加一个 loader，这里不用改。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document as LCDocument

import config
import vectorstore
from chunking import build_chunks
from headings import sections_from_blocks
from loaders import LoaderError, load_blocks, resolve_sources

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """入库失败（找不到文件、格式不支持、文档没有文字等）。"""


@dataclass
class FileReport:
    path: str
    chunks: int = 0
    removed: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class IngestReport:
    collection: str = ""
    directory: str = ""
    files: list[FileReport] = field(default_factory=list)
    seconds: float = 0.0
    reset: bool = False

    @property
    def chunks(self) -> int:
        return sum(item.chunks for item in self.files)

    @property
    def failed(self) -> list[FileReport]:
        return [item for item in self.files if not item.ok]

    def summary(self) -> str:
        head = (
            f"{len(self.files)} 个文件 → {self.chunks} 个文本块，"
            f"耗时 {self.seconds:.1f}s，集合 {self.collection}"
        )
        if self.failed:
            head += f"，其中 {len(self.failed)} 个失败"
        return head


def build_documents(path: Path, doc_type: str = "") -> list[LCDocument]:
    """单个文件 -> 文本块列表。"""
    path = Path(path)
    blocks = list(load_blocks(path, doc_type))
    if not blocks:
        raise IngestError(f"{path.name} 里没有读到任何文字。")
    sections = list(sections_from_blocks(blocks))
    return build_chunks(sections, source_name=path.name)


def ingest_sources(specs: str | Path | Sequence[str | Path] | None = None, *,
                   reset: bool = True, doc_type: str = "") -> IngestReport:
    """批量入库。`reset=True` 先清空集合，做成"全量重建"。

    `reset=False` 时按文件增量更新：只替换这些文件对应的块，其它文档不动。
    """
    started = time.perf_counter()
    raw_spec = specs if specs is not None else (config.SOURCES or config.DOC_PATH)
    paths = resolve_sources(raw_spec)

    report = IngestReport(
        collection=config.COLLECTION_NAME,
        directory=str(config.CHROMA_DIR),
        reset=reset,
    )
    if not paths:
        raise IngestError(f"没有找到可入库的文件：{raw_spec!r}")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        vectorstore.delete_collection()
        logger.info("已清空集合 %s", config.COLLECTION_NAME)

    for path in paths:
        item = FileReport(path=str(path))
        file_started = time.perf_counter()
        try:
            # 单文件模式下才使用显式 DOC_TYPE，多文件时按各自扩展名判断
            explicit = doc_type if (doc_type and len(paths) == 1) else ""
            chunks = build_documents(path, explicit)
            if not chunks:
                raise IngestError("分块后没有内容（可能全是空行或短于 MIN_CHUNK_CHARS）。")
            if not reset:
                item.removed = vectorstore.delete_source(path.name)
            vectorstore.upsert_chunks(
                ids=[chunk.metadata["chunk_id"] for chunk in chunks],
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )
            item.chunks = len(chunks)
        except (LoaderError, IngestError) as exc:
            item.error = str(exc)
            logger.error("入库失败 %s：%s", path.name, exc)
        except Exception as exc:  # 单文件失败不该拖垮整批
            item.error = f"{type(exc).__name__}: {exc}"
            logger.exception("入库异常 %s", path.name)
        item.seconds = time.perf_counter() - file_started
        report.files.append(item)

    report.seconds = time.perf_counter() - started
    return report


def ingest_documents(doc_path: Path | str | None = None) -> int:
    """兼容旧接口：重建单文档索引，返回文本块数量。"""
    report = ingest_sources(doc_path or config.DOC_PATH, reset=True,
                            doc_type=config.DOC_TYPE)
    if report.failed and report.chunks == 0:
        raise IngestError(report.failed[0].error)
    return report.chunks


def iter_chunks(chunks: Iterable[LCDocument]) -> Iterable[tuple[str, str]]:
    """小工具：打印/调试时遍历 (章节, 正文)。"""
    for chunk in chunks:
        yield chunk.metadata.get("section", ""), chunk.page_content
