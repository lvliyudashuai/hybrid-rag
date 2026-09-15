"""分块：元数据、稳定 ID、章节前缀。"""
from __future__ import annotations

import config
from chunking import build_chunks, chunk_id, strip_context_prefix
from docmodel import Block, Section


def _sections() -> list[Section]:
    return [
        Section("第一回 开篇", [Block("第一回 开篇"), Block("正文" * 60)]),
        Section("第二回 发展", [Block("第二回 发展"), Block("内容" * 60)]),
    ]


def test_metadata_is_complete():
    chunks = build_chunks(_sections(), source_name="demo.docx")
    first = chunks[0]
    for key in ("chunk_id", "source", "section", "page", "chunk_index",
                "section_chunk_index", "chars"):
        assert key in first.metadata, key
    assert first.metadata["source"] == "demo.docx"
    assert first.metadata["section"] == "第一回 开篇"
    assert first.metadata["chunk_index"] == 0


def test_chunk_ids_are_stable_and_unique():
    first = build_chunks(_sections(), source_name="demo.docx")
    second = build_chunks(_sections(), source_name="demo.docx")
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]
    ids = [chunk.metadata["chunk_id"] for chunk in first]
    assert len(set(ids)) == len(ids)


def test_chunk_id_changes_when_content_changes():
    assert chunk_id("a", "s", 0, "内容一") != chunk_id("a", "s", 0, "内容二")
    assert chunk_id("a", "s", 0, "同一段") == chunk_id("a", "s", 0, "同一段")


def test_section_prefix_can_be_stripped_for_display():
    chunks = build_chunks(_sections(), source_name="demo.docx")
    text = chunks[0].page_content
    assert text.startswith("[第一回 开篇]")
    assert strip_context_prefix(text).startswith("第一回 开篇")
    assert "[" not in strip_context_prefix(text)[:1]


def test_prefix_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_CONTEXT_PREFIX", False)
    chunks = build_chunks(_sections(), source_name="demo.docx")
    assert not chunks[0].page_content.startswith("[")


def test_short_pieces_are_dropped(monkeypatch):
    monkeypatch.setattr(config, "MIN_CHUNK_CHARS", 1000)
    assert build_chunks(_sections(), source_name="demo.docx") == []


def test_sections_longer_than_chunk_size_are_split(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 100)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 10)
    sections = [Section("长章节", [Block("很长的一段话。" * 60)])]
    chunks = build_chunks(sections, source_name="long.docx")
    assert len(chunks) > 1
    assert all(chunk.metadata["section"] == "长章节" for chunk in chunks)
    assert [chunk.metadata["section_chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_empty_section_is_skipped():
    chunks = build_chunks([Section("空的", [])], source_name="x.docx")
    assert chunks == []
