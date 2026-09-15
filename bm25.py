"""BM25 关键词检索：向量检索的"另一半"。

为什么需要它？向量检索擅长语义，但遇到**专有名词、编号、人名、法条号**
这类精确词反而容易漏（它们在小模型里几乎没学到语义）。BM25 是纯粹的词频
统计，正好补上这块。两路一起召回再用 RRF 融合，就是工业界常说的混合检索。

分词默认用 jieba；没装 jieba 时退化成「中文二元组 + 英文单词」，
保证任何环境都能跑起来。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_HAS_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]")

# 查询侧停用词：这些词在几乎每个块里都出现，IDF 低但会在短查询里
# 抢走大量权重，把它们从"查询"里去掉能明显提升关键词路的精度。
# 注意只在查询侧过滤，文档侧保留（否则会改变语料的长度分布）。
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "被", "把", "对",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "这个",
    "那个", "什么", "怎么", "怎样", "为什么", "哪个", "哪里", "多少", "几",
    "吗", "呢", "吧", "请问", "请", "介绍", "说明", "一下", "可以", "能否",
    "a", "an", "the", "of", "is", "are", "was", "to", "in", "on", "for",
    "what", "which", "how", "why", "and", "or",
}

K1 = 1.5   # 词频饱和系数
B = 0.75   # 文档长度归一化强度

_JIEBA = None
_JIEBA_TRIED = False


def _load_jieba():
    global _JIEBA, _JIEBA_TRIED
    if not _JIEBA_TRIED:
        _JIEBA_TRIED = True
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            _JIEBA = jieba
        except Exception:
            logger.warning("未安装 jieba，中文改用二元组分词（效果略降，但可用）")
    return _JIEBA


def tokenize(text: str) -> list[str]:
    """中英文混合分词，统一小写。"""
    if not text:
        return []
    lowered = text.lower()
    jieba = _load_jieba()
    if jieba is not None:
        tokens = [token.strip() for token in jieba.lcut(lowered)]
        return [token for token in tokens if token and not token.isspace()]

    tokens: list[str] = []
    for chunk in _CJK_RE.findall(lowered):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    tokens.extend(_ASCII_WORD_RE.findall(lowered))
    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：在通用分词基础上丢掉标点和停用词。"""
    return [
        token for token in tokenize(text)
        if _HAS_WORD_RE.search(token) and token not in _STOPWORDS
    ]


@dataclass(slots=True)
class KeywordHit:
    index: int      # 在文档列表里的下标
    score: float


class BM25Index:
    """标准 Okapi BM25。语料不大（几万块以内）时，纯 Python 足够快。"""

    def __init__(self, documents: Sequence[str], ids: Sequence[str] | None = None,
                 metadatas: Sequence[dict] | None = None) -> None:
        self.documents = list(documents)
        # ids / metadatas 与 documents 同序：命中下标可以直接回溯到向量库里的块
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(documents))]
        self.metadatas = (
            list(metadatas) if metadatas is not None else [{} for _ in documents]
        )
        self.doc_count = len(self.documents)
        self.term_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        document_freq: Counter = Counter()

        for document in self.documents:
            tokens = tokenize(document)
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            self.doc_lengths.append(len(tokens))
            document_freq.update(freqs.keys())

        self.avg_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def search(self, query: str, k: int = 20) -> list[KeywordHit]:
        if not self.doc_count:
            return []
        terms = tokenize_query(query) or tokenize(query)
        if not terms:
            return []

        scores: list[float] = [0.0] * self.doc_count
        for term in set(terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                length = self.doc_lengths[index] or 1
                denominator = tf + K1 * (1 - B + B * length / (self.avg_length or 1))
                scores[index] += idf * tf * (K1 + 1) / denominator

        ranked = sorted(
            ((index, score) for index, score in enumerate(scores) if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        return [KeywordHit(index=index, score=score) for index, score in ranked]


# --------------------------------------------------------------------------
# 索引缓存：向量库内容没变就不重建
# --------------------------------------------------------------------------
_INDEX_CACHE: dict[tuple, BM25Index] = {}


def index_fingerprint(name: str, count: int, documents: Iterable[str]) -> tuple:
    """用「块数 + 总字符数」当指纹：重建过库就一定会变。"""
    return (name, count, sum(len(item) for item in documents))


def get_index(name: str | None = None, *, force: bool = False) -> BM25Index:
    """从向量库取全部文本，建（或复用）BM25 索引。"""
    import config
    import vectorstore

    collection_name = name or config.COLLECTION_NAME
    ids, documents, metadatas = vectorstore.collection_documents(collection_name)
    key = index_fingerprint(collection_name, len(documents), documents)

    if force or key not in _INDEX_CACHE:
        logger.info("构建 BM25 索引：%s 共 %d 块", collection_name, len(documents))
        # 只留最近一份，避免反复重建时把内存撑爆
        for stale in [item for item in _INDEX_CACHE if item[0] == collection_name]:
            _INDEX_CACHE.pop(stale, None)
        _INDEX_CACHE[key] = BM25Index(documents, ids, metadatas)
    return _INDEX_CACHE[key]


def index_documents(name: str | None = None) -> list[str]:
    """索引里保存的文本（下标与 KeywordHit.index 对齐）。"""
    index = get_index(name)
    return index.documents


def clear_cache() -> None:
    _INDEX_CACHE.clear()
