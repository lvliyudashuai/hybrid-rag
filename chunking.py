"""分块：把 Section 切成带元数据的文本块。

两个关键设计：
1. **章节前缀** —— 每个块前面拼上它的章节路径再向量化。中文长文档里，
   光看块本身常常判断不出它在讲谁（"他道：'……'"），加上前缀后召回明显变好。
   前缀用 `[路径]` 包裹，展示时可以再剥掉。
2. **稳定 ID** —— 块 ID 由 来源+章节+序号+正文 哈希得到。同一份文档重复入库
   会命中同一个 ID（upsert 幂等），不会越建越多的重复块。
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from docmodel import Section

PREFIX_OPEN = "["
PREFIX_CLOSE = "] "


def make_splitter(chunk_size: int | None = None,
                  chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    """中文友好的递归分块器：优先在段落/句号处断开。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "\u3000", " ", ""],
        keep_separator=True,
    )


def chunk_id(source: str, section: str, index: int, text: str) -> str:
    """稳定的块 ID：内容相同 -> ID 相同。"""
    payload = f"{source}\x00{section}\x00{index}\x00{text}".encode()
    return hashlib.sha1(payload).hexdigest()[:24]


def strip_context_prefix(text: str) -> str:
    """把展示用的章节前缀剥掉，得到纯正文。"""
    if text.startswith(PREFIX_OPEN):
        end = text.find(PREFIX_CLOSE)
        if end != -1:
            return text[end + len(PREFIX_CLOSE):].lstrip()
    return text


def build_chunks(sections: Iterable[Section], source_name: str) -> list[LCDocument]:
    """Section 序列 -> LangChain Document 列表（含完整元数据）。"""
    splitter = make_splitter()
    chunks: list[LCDocument] = []
    global_index = 0

    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        pieces = [piece.strip() for piece in splitter.split_text(text)]
        page = section.page_label()
        section_index = 0
        for piece in pieces:
            if len(piece) < config.MIN_CHUNK_CHARS:
                continue
            body = (
                f"{PREFIX_OPEN}{section.path}{PREFIX_CLOSE}{piece}"
                if config.CHUNK_CONTEXT_PREFIX
                else piece
            )
            identifier = chunk_id(source_name, section.path, section_index, piece)
            chunks.append(
                LCDocument(
                    page_content=body,
                    metadata={
                        "chunk_id": identifier,
                        "source": source_name,
                        "section": section.path,
                        "page": page,
                        "chunk_index": global_index,
                        "section_chunk_index": section_index,
                        "chars": len(piece),
                    },
                )
            )
            global_index += 1
            section_index += 1
    return chunks
