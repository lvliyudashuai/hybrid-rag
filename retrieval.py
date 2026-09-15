"""检索层：向量 + BM25 混合召回 -> RRF 融合 -> MMR 去重 -> 可选重排。

这是整个 RAG 里最影响效果的一层。四种检索模式：

- `vector`  ：纯语义检索，问法灵活但容易漏掉专有名词。
- `keyword` ：纯 BM25，专有名词/编号命中准，但同义改写就失效。
- `hybrid`  ：两路都召回，用 RRF 按「排名」而不是「分数」融合（推荐默认）。
- 任何模式都可以再叠加 `USE_RERANK=1`，用 CrossEncoder 精排。

融合用 RRF（Reciprocal Rank Fusion）而不是直接加权分数，是因为向量相似度
和 BM25 分数**量纲完全不同**，直接相加会被 BM25 的大数值主导。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LCDocument

import bm25
import config
import vectorstore

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[tuple[str, str], Any] = {}


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Hit:
    """一条检索结果。字段保留每一路的原始分，便于排查"为什么把它排上来"。"""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    source: str = "vector"                 # vector / keyword / hybrid
    similarity: float | None = None        # 向量相似度（0~1）
    keyword_score: float | None = None     # BM25 原始分
    rerank_score: float | None = None      # 重排分（logits，越大越相关）
    rank: int = 0

    @property
    def section(self) -> str:
        return str(self.metadata.get("section") or "未知章节")

    @property
    def page(self) -> str:
        return str(self.metadata.get("page") or "")

    @property
    def source_file(self) -> str:
        return str(self.metadata.get("source") or "")

    @property
    def locator(self) -> str:
        """"来源 · 章节 · 页码" 的一句话描述，直接用在下拉框标题上。"""
        parts = [part for part in (self.source_file, self.section, self.page) if part]
        return " ｜ ".join(parts)

    def as_document(self) -> LCDocument:
        return LCDocument(page_content=self.text, metadata=dict(self.metadata))

    def preview(self, limit: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[:limit] + "…"


@dataclass
class Trace:
    """检索过程的耗时与中间结果，排查效果问题全靠它。"""

    query: str = ""
    mode: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, milliseconds: float, detail: str = "") -> None:
        self.steps.append({"step": name, "ms": round(milliseconds, 1), "detail": detail})

    @property
    def total_ms(self) -> float:
        return round(sum(step["ms"] for step in self.steps), 1)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "mode": self.mode,
                "total_ms": self.total_ms, "steps": list(self.steps)}


class _Timer:
    """小工具：with 块里计时并写进 Trace。"""

    def __init__(self, trace: Trace | None, name: str, detail: str = "") -> None:
        self.trace = trace
        self.name = name
        self.detail = detail

    def __enter__(self) -> _Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        if self.trace is not None:
            elapsed = (time.perf_counter() - self.start) * 1000
            self.trace.add(self.name, elapsed, self.detail)


# --------------------------------------------------------------------------
# 召回
# --------------------------------------------------------------------------
def _vector_candidates(query: str, collection, limit: int,
                       filters: dict | None) -> list[dict[str, Any]]:
    """向量路召回：一次 query 同时拿到文本、元数据、距离和向量。"""
    total = collection.count()
    if total == 0:
        return []
    vector = vectorstore.embed_query(query)
    result = collection.query(
        query_embeddings=[vector],
        n_results=max(1, min(limit, total)),
        where=filters or None,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    ids = result.get("ids", [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    embeddings = (result.get("embeddings") or [[]])[0]

    items: list[dict[str, Any]] = []
    for index, doc_id in enumerate(ids):
        items.append({
            "id": doc_id,
            "text": documents[index] if index < len(documents) else "",
            "metadata": dict(metadatas[index] or {}) if index < len(metadatas) else {},
            "similarity": vectorstore.distance_to_similarity(distances[index]),
            "embedding": list(embeddings[index]) if index < len(embeddings) else None,
        })
    return items


def _keyword_candidates(query: str, limit: int,
                        name: str | None) -> list[dict[str, Any]]:
    """BM25 关键词路召回。"""
    index = bm25.get_index(name)
    results = index.search(query, k=limit)
    items: list[dict[str, Any]] = []
    for hit in results:
        items.append({
            "id": index.ids[hit.index],
            "text": index.documents[hit.index],
            "metadata": dict(index.metadatas[hit.index] or {}),
            "keyword_score": hit.score,
        })
    return items


# --------------------------------------------------------------------------
# 融合与重排
# --------------------------------------------------------------------------
def _fuse(vector_items: Sequence[dict], keyword_items: Sequence[dict],
          alpha: float, rrf_k: int) -> list[dict[str, Any]]:
    """加权 RRF：按排名融合两路结果。

    分数 = alpha * 1/(k + 向量排名) + (1-alpha) * 1/(k + 关键词排名)
    """
    merged: dict[str, dict[str, Any]] = {}

    def take(items: Sequence[dict], weight: float, tag: str) -> None:
        for rank, item in enumerate(items, start=1):
            entry = merged.setdefault(item["id"], dict(item))
            entry["fused"] = entry.get("fused", 0.0) + weight / (rrf_k + rank)
            entry.setdefault("channels", set()).add(tag)
            # 补全另一路缺失的字段（相似度 / 关键词分 / 向量）
            for key in ("similarity", "keyword_score", "embedding", "text", "metadata"):
                if entry.get(key) is None and item.get(key) is not None:
                    entry[key] = item[key]

    take(vector_items, alpha, "vector")
    take(keyword_items, 1.0 - alpha, "keyword")

    ordered = sorted(merged.values(), key=lambda item: item.get("fused", 0.0), reverse=True)
    for entry in ordered:
        channels = entry.get("channels") or set()
        entry["source"] = "hybrid"
        entry["channels"] = sorted(channels)
        entry["score"] = entry.get("fused", 0.0)
    return ordered


def _ensure_embeddings(collection, items: Sequence[dict]) -> None:
    """给缺向量的候选补上向量（关键词路召回的块没有向量）。"""
    missing = [item["id"] for item in items if item.get("embedding") is None]
    if not missing:
        return
    try:
        result = collection.get(ids=missing, include=["embeddings"])
        fetched = result.get("embeddings")
        if fetched is None:
            return
        lookup = {
            doc_id: list(vector)
            for doc_id, vector in zip(result.get("ids") or [], fetched, strict=False)
        }
        for item in items:
            if item.get("embedding") is None:
                item["embedding"] = lookup.get(item["id"])
    except Exception:
        logger.debug("补齐候选向量失败，MMR 将退化为按相关性排序", exc_info=True)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    # embedding 已归一化，点积即余弦相似度
    return sum(a * b for a, b in zip(left, right, strict=False))


def _mmr_select(items: Sequence[dict], k: int, lam: float) -> list[dict[str, Any]]:
    """最大边际相关：在"相关"和"不重复"之间取平衡。"""
    candidates = list(items)
    if len(candidates) <= k:
        return candidates

    peak = max((item.get("score", 0.0) for item in candidates), default=1.0) or 1.0
    selected: list[dict[str, Any]] = []
    while candidates and len(selected) < k:
        best_index, best_value = 0, float("-inf")
        for index, item in enumerate(candidates):
            relevance = item.get("score", 0.0) / peak
            vector = item.get("embedding")
            penalty = 0.0
            if selected and vector is not None:
                similarities = [
                    _cosine(vector, chosen["embedding"])
                    for chosen in selected if chosen.get("embedding") is not None
                ]
                penalty = max(similarities) if similarities else 0.0
            value = lam * relevance - (1 - lam) * penalty
            if value > best_value:
                best_index, best_value = index, value
        selected.append(candidates.pop(best_index))
    return selected


def get_reranker(model_name: str | None = None):
    """CrossEncoder 重排模型（首次使用会自动下载，约 1GB）。"""
    from sentence_transformers import CrossEncoder  # 延迟导入

    name = model_name or config.RERANK_MODEL
    device = vectorstore.resolve_device()
    key = (name, device)
    if key not in _RERANKER_CACHE:
        logger.info("加载重排模型：%s（设备 %s）", name, device)
        _RERANKER_CACHE[key] = CrossEncoder(name, device=device)
    return _RERANKER_CACHE[key]


def _apply_rerank(query: str, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    pool = items[:top_k]
    if not pool:
        return items
    model = get_reranker()
    scores = model.predict([(query, item["text"]) for item in pool])
    for item, score in zip(pool, scores, strict=False):
        item["rerank_score"] = float(score)
    pool.sort(key=lambda item: item["rerank_score"], reverse=True)
    return pool + items[top_k:]


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def search(
    query: str,
    k: int | None = None,
    *,
    mode: str | None = None,
    name: str | None = None,
    filters: dict | None = None,
    use_mmr: bool | None = None,
    use_rerank: bool | None = None,
    alpha: float | None = None,
    trace: Trace | None = None,
) -> list[Hit]:
    """检索入口。返回按相关性排序的 Hit 列表。

    所有参数留空时都用 config 里的值，方便界面覆盖单个参数做对比实验。
    """
    query = (query or "").strip()
    mode = (mode or config.RETRIEVAL_MODE).lower()
    k = k or config.TOP_K
    alpha = config.HYBRID_ALPHA if alpha is None else alpha
    use_mmr = config.USE_MMR if use_mmr is None else use_mmr
    use_rerank = config.USE_RERANK if use_rerank is None else use_rerank
    if trace is not None:
        trace.query, trace.mode = query, mode

    if not query:
        return []
    try:
        collection = vectorstore.get_collection(name=name)
    except Exception as exc:
        if trace is not None:
            trace.add("打开向量库", 0.0, f"失败：{exc}")
        return []

    candidate_k = max(config.CANDIDATE_K, k)
    vector_items: list[dict[str, Any]] = []
    keyword_items: list[dict[str, Any]] = []

    if mode in {"vector", "hybrid"}:
        with _Timer(trace, "向量召回", f"top {candidate_k}") as timer:
            vector_items = _vector_candidates(query, collection, candidate_k, filters)
            timer.detail = f"{len(vector_items)} 个候选"
    if mode in {"keyword", "hybrid"}:
        with _Timer(trace, "BM25 关键词召回", f"top {candidate_k}") as timer:
            keyword_items = _keyword_candidates(query, candidate_k, name)
            timer.detail = f"{len(keyword_items)} 个候选"

    if mode == "vector":
        items = vector_items
        for item in items:
            item["score"] = item.get("similarity", 0.0)
            item["source"] = "vector"
    elif mode == "keyword":
        items = keyword_items
        for item in items:
            item["score"] = item.get("keyword_score", 0.0)
            item["source"] = "keyword"
    else:
        with _Timer(trace, "RRF 融合", f"alpha={alpha}") as timer:
            items = _fuse(vector_items, keyword_items, alpha, config.RRF_K)
            timer.detail = f"{len(items)} 个候选"

    # 相似度阈值过滤：词法路的候选没有相似度，一律保留
    if config.SCORE_THRESHOLD > 0:
        items = [
            item for item in items
            if item.get("similarity") is None or item["similarity"] >= config.SCORE_THRESHOLD
        ]

    pool_size = max(k * 2, 10)
    if use_mmr and len(items) > k:
        with _Timer(trace, "MMR 去重", f"lambda={config.MMR_LAMBDA}") as timer:
            _ensure_embeddings(collection, items[:pool_size])
            items = _mmr_select(items, pool_size, config.MMR_LAMBDA)
            timer.detail = f"{len(items)} 个候选"

    if use_rerank and items:
        with _Timer(trace, "CrossEncoder 重排", config.RERANK_MODEL) as timer:
            items = _apply_rerank(query, items, config.RERANK_TOP_K)
            timer.detail = f"{len(items)} 个候选"

    hits: list[Hit] = []
    for rank, item in enumerate(items[:k], start=1):
        hits.append(
            Hit(
                doc_id=item["id"],
                text=item.get("text", ""),
                metadata=dict(item.get("metadata") or {}),
                score=float(item.get("score", 0.0)),
                source=item.get("source", "vector"),
                similarity=item.get("similarity"),
                keyword_score=item.get("keyword_score"),
                rerank_score=item.get("rerank_score"),
                rank=rank,
            )
        )
    return hits
