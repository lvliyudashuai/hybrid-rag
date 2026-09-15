"""测试公共装置。

关键点：全程不联网、不下载模型。用一个确定性的假 embedding 顶替真实模型，
这样 CI 里几秒就能跑完，而且结果稳定（真实模型的分数会随版本漂移）。
"""
from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import pytest

# 让 tests/ 能 import 到项目根目录下的模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


class FakeEmbeddings:
    """字符二元组哈希向量：文本越像，向量越接近。

    不是真模型，但保留了"语义相近 -> 向量相近"的基本性质，
    足以验证检索链路是否接对。
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        cleaned = "".join(text.split())
        if not cleaned:
            return vector
        grams = [cleaned[i:i + 2] for i in range(max(1, len(cleaned) - 1))]
        for gram in grams:
            # 用 crc32 而不是内置 hash：内置 hash 每个进程都会加盐，跨运行不稳定
            index = zlib.crc32(gram.encode("utf-8")) % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def isolated_store(tmp_path, monkeypatch, fake_embeddings):
    """把向量库指到临时目录，并用假 embedding 顶替真实模型。"""
    import bm25
    import vectorstore

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "COLLECTION_NAME", "test_kb")
    monkeypatch.setattr(config, "CHUNK_SIZE", 120)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 20)
    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 5)
    monkeypatch.setattr(config, "USE_MMR", False)
    monkeypatch.setattr(config, "USE_RERANK", False)
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(vectorstore, "create_embeddings", lambda *a, **k: fake_embeddings)
    vectorstore.clear_embedding_cache()
    bm25.clear_cache()
    yield tmp_path
    vectorstore.clear_embedding_cache()
    bm25.clear_cache()


@pytest.fixture
def sample_doc(tmp_path) -> Path:
    """一份结构清晰的小文档：两级编号标题 + 一个固定词标题。"""
    path = tmp_path / "sample.md"
    path.write_text(
        "# 1. 总则\n"
        "本文件用于说明退货流程，适用于所有线上订单。\n"
        "## 1.1 适用范围\n"
        "仅限通过官方渠道购买的商品。\n"
        "## 1.2 生效时间\n"
        "自 2026 年 1 月 1 日起生效。\n"
        "# 2. 退货条件\n"
        "商品需保持完好，且在三日内提出申请。\n"
        "## 2.1 不支持退货\n"
        "定制品与拆封的电子设备不支持退货。\n"
        "# 违约责任\n"
        "任何一方违约需赔偿对方由此产生的直接损失。\n",
        encoding="utf-8",
    )
    return path
