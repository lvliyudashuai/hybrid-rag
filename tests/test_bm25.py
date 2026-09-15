"""BM25：中文分词与排序是否符合直觉。"""
from __future__ import annotations

from bm25 import BM25Index, tokenize

CORPUS = [
    "第一回 灵根育孕源流出 心性修持大道生。孙悟空从石头里蹦出来。",
    "第三回 四海千山皆拱伏 九幽十类尽除名。如意金箍棒重一万三千五百斤。",
    "第八九回 黄狮精虚设钉钯宴。猪八戒的九齿钉钯重八百斤。",
]


def test_tokenize_returns_something_for_chinese():
    tokens = tokenize("猪八戒的武器")
    assert tokens
    assert all(isinstance(token, str) for token in tokens)


def test_tokenize_handles_mixed_language():
    tokens = tokenize("RAG 系统使用 BM25 算法")
    assert any("bm25" in token for token in tokens)
    assert any("算" in token for token in tokens)


def test_tokenize_empty():
    assert tokenize("") == []


def test_exact_term_wins():
    """关键词检索的看家本领：专有名词必须排第一。"""
    index = BM25Index(CORPUS)
    results = index.search("金箍棒有多重", k=3)
    assert results, "应该至少命中一条"
    assert index.documents[results[0].index].startswith("第三回")


def test_scores_are_descending():
    index = BM25Index(CORPUS)
    results = index.search("猪八戒 武器", k=3)
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_query_returns_nothing():
    index = BM25Index(CORPUS)
    assert index.search("量子纠缠退相干", k=3) == []


def test_empty_corpus_is_safe():
    index = BM25Index([])
    assert index.search("任何问题", k=3) == []


def test_k_limits_results():
    index = BM25Index(CORPUS)
    assert len(index.search("回", k=2)) <= 2


def test_index_keeps_ids_and_metadatas_aligned():
    index = BM25Index(CORPUS, ids=["a", "b", "c"], metadatas=[{"s": 1}, {"s": 2}, {"s": 3}])
    results = index.search("金箍棒", k=1)
    assert index.ids[results[0].index] == "b"
    assert index.metadatas[results[0].index]["s"] == 2
