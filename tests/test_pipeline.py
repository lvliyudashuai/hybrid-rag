"""端到端：入库 -> 检索 -> 生成（用离线模拟，不联网）。"""
from __future__ import annotations

import config
import ingest
import query


def test_ingest_then_search(isolated_store, sample_doc):
    report = ingest.ingest_sources(sample_doc, reset=True)
    assert report.chunks > 0
    assert not report.failed
    hits = query.retrieve("退货需要几天内提出", 3)
    assert hits
    assert any("退货" in hit.text or "退货" in hit.section for hit in hits)


def test_ingest_is_idempotent(isolated_store, sample_doc):
    """同一份文档重复入库，块数不应该翻倍（稳定 ID + upsert 的功劳）。"""
    import vectorstore

    first = ingest.ingest_sources(sample_doc, reset=True).chunks
    after_first = vectorstore.collection_stats()["count"]
    ingest.ingest_sources(sample_doc, reset=False)
    after_second = vectorstore.collection_stats()["count"]
    assert first == after_first == after_second


def test_incremental_ingest_replaces_only_that_source(isolated_store, tmp_path):
    import vectorstore

    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_text("1. 甲文档\n甲公司内容\n", encoding="utf-8")
    beta.write_text("1. 乙文档\n乙公司内容\n", encoding="utf-8")
    ingest.ingest_sources([alpha, beta], reset=True)

    alpha.write_text("1. 甲文档\n甲公司内容已更新\n", encoding="utf-8")
    ingest.ingest_sources(alpha, reset=False)

    _, documents, metadatas = vectorstore.collection_documents()
    assert sum(1 for meta in metadatas if meta["source"] == "beta.txt") > 0
    assert any("已更新" in text for text in documents)


def test_ingest_without_documents_raises(isolated_store, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    report = ingest.ingest_sources(empty, reset=True)
    assert report.chunks == 0
    assert report.failed


def test_mock_answer_is_used_without_api_key(isolated_store, sample_doc, monkeypatch):
    ingest.ingest_sources(sample_doc, reset=True)
    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config.set_api_key(None)
    try:
        result = query.ask("退货条件是什么")
        assert result.provider == "mock"
        assert result.used_llm is False
        assert "离线模拟" in result.text
        assert result.hits
    finally:
        config.set_api_key(None)


def test_ask_reports_no_answer_when_nothing_retrieved(isolated_store, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "mock")
    result = query.ask("文档里根本没有的问题")
    assert result.hits == []
    assert "没有检索到" in result.text


def test_context_budget_truncates(isolated_store, sample_doc, monkeypatch):
    ingest.ingest_sources(sample_doc, reset=True)
    monkeypatch.setattr(config, "MAX_CONTEXT_CHARS", 120)
    hits = query.retrieve("退货", 5)
    context = query.format_context(hits)
    assert "已达上下文预算" in context


def test_messages_include_system_history_and_sources(isolated_store, sample_doc):
    ingest.ingest_sources(sample_doc, reset=True)
    hits = query.retrieve("退货", 2)
    messages = query.build_messages("退货条件？", hits,
                                    history=[{"role": "user", "content": "之前问过什么"},
                                             {"role": "assistant", "content": "之前的回答"}])
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "之前问过什么"
    assert "【参考资料】" in messages[-1]["content"]
    assert "[1]" in messages[-1]["content"]


def test_citations_markdown_lists_sources(isolated_store, sample_doc):
    ingest.ingest_sources(sample_doc, reset=True)
    hits = query.retrieve("退货", 2)
    text = query.citations_markdown(hits)
    assert "参考来源" in text
    assert "sample.md" in text
