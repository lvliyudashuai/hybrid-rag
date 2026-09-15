"""检索层：融合、阈值、MMR、退化路径。"""
from __future__ import annotations

import config
import retrieval


def _item(doc_id: str, score: float, **extra):
    payload = {"id": doc_id, "text": f"正文 {doc_id}", "metadata": {"section": doc_id},
               "score": score}
    payload.update(extra)
    return payload


def test_rrf_fuses_both_channels():
    """只出现在关键词路的结果也应该被融合进来。"""
    vector_items = [_item("a", 0.9), _item("b", 0.8)]
    keyword_items = [{**_item("c", 0.0), "keyword_score": 12.0}]
    fused = retrieval._fuse(vector_items, keyword_items, alpha=0.6, rrf_k=60)
    ids = [item["id"] for item in fused]
    assert set(ids) == {"a", "b", "c"}
    assert ids[0] == "a"  # 排第一的仍然是两路都靠前的那个
    entry_c = next(item for item in fused if item["id"] == "c")
    assert entry_c["channels"] == ["keyword"]
    assert entry_c["source"] == "hybrid"


def test_rrf_alpha_controls_influence():
    vector_items = [_item("v", 0.9)]
    keyword_items = [_item("k", 0.1)]
    keyword_heavy = retrieval._fuse(vector_items, keyword_items, alpha=0.1, rrf_k=60)
    vector_heavy = retrieval._fuse(vector_items, keyword_items, alpha=0.9, rrf_k=60)
    assert keyword_heavy[0]["id"] == "k"
    assert vector_heavy[0]["id"] == "v"


def test_rrf_bonus_for_appearing_in_both_channels():
    vector_items = [_item("both", 0.5), _item("only_vector", 0.5)]
    keyword_items = [_item("both", 0.5)]
    fused = retrieval._fuse(vector_items, keyword_items, alpha=0.5, rrf_k=60)
    assert fused[0]["id"] == "both"


def test_fuse_keeps_missing_fields_from_either_side():
    vector_items = [_item("x", 0.7, similarity=0.7, embedding=[1.0, 0.0])]
    keyword_items = [{**_item("x", 0.0), "keyword_score": 5.0}]
    fused = retrieval._fuse(vector_items, keyword_items, alpha=0.5, rrf_k=60)
    assert fused[0]["similarity"] == 0.7
    assert fused[0]["keyword_score"] == 5.0
    assert fused[0]["embedding"] == [1.0, 0.0]


def test_mmr_prefers_diverse_candidates():
    items = [
        {**_item("a", 1.0), "embedding": [1.0, 0.0]},
        {**_item("b", 0.99), "embedding": [1.0, 0.0]},   # 和 a 几乎一样
        {**_item("c", 0.5), "embedding": [0.0, 1.0]},    # 完全不同
    ]
    selected = retrieval._mmr_select(items, k=2, lam=0.5)
    assert [item["id"] for item in selected] == ["a", "c"]


def test_mmr_returns_everything_when_pool_is_small():
    items = [{**_item("a", 1.0), "embedding": [1.0, 0.0]}]
    assert len(retrieval._mmr_select(items, k=5, lam=0.7)) == 1


def test_cosine_matches_normalized_dot_product():
    assert retrieval._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert retrieval._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_distance_to_similarity_is_clamped():
    from vectorstore import distance_to_similarity

    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(1.25) == 0.0
    assert distance_to_similarity(-0.5) == 1.0


def test_search_on_missing_collection_returns_empty(isolated_store):
    assert retrieval.search("任何问题", 3) == []


def test_search_records_trace(isolated_store, sample_doc):
    import ingest

    ingest.ingest_sources(sample_doc, reset=True)
    trace = retrieval.Trace()
    retrieval.search("退货条件", 3, trace=trace)
    names = [step["step"] for step in trace.steps]
    assert "向量召回" in names and "BM25 关键词召回" in names and "RRF 融合" in names


def test_empty_query_returns_nothing(isolated_store):
    assert retrieval.search("   ", 3) == []


def test_modes_return_hits(isolated_store, sample_doc):
    import ingest

    ingest.ingest_sources(sample_doc, reset=True)
    for mode in ("vector", "keyword", "hybrid"):
        hits = retrieval.search("退货条件是什么", 3, mode=mode)
        assert hits, mode
        assert all(hit.rank == index for index, hit in enumerate(hits, 1))


def test_score_threshold_filters_weak_vector_hits(isolated_store, sample_doc, monkeypatch):
    import ingest

    ingest.ingest_sources(sample_doc, reset=True)
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.999)
    hits = retrieval.search("退货条件", 5, mode="vector")
    assert all(hit.similarity >= 0.999 for hit in hits)


def test_hit_locator_includes_source_and_section(isolated_store, sample_doc):
    import ingest

    ingest.ingest_sources(sample_doc, reset=True)
    hit = retrieval.search("退货条件", 1)[0]
    assert "sample.md" in hit.locator
    assert hit.section
