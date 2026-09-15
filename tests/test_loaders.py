"""多格式加载：通用性的核心，格式多一个，能用的文档就多一类。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from loaders import (
    LoaderError,
    detect_type,
    load_blocks,
    read_text,
    resolve_sources,
)


def _texts(path: Path, doc_type: str = "") -> list[str]:
    return [block.text for block in load_blocks(path, doc_type)]


def test_txt_and_markdown_are_read_line_by_line(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\n第一行\n第二行\n", encoding="utf-8")
    assert _texts(path) == ["# 标题", "第一行", "第二行"]


def test_gbk_text_does_not_explode(tmp_path):
    """中文用户手上大量 GBK 老文件，读不出来会直接报 UnicodeDecodeError。"""
    path = tmp_path / "gbk.txt"
    path.write_bytes("合同标的：一批钢材\n".encode("gb18030"))
    assert "一批钢材" in read_text(path)


def test_csv_rows_become_key_value_lines(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("产品,价格\n笔记本,5000\n鼠标,100\n", encoding="utf-8")
    lines = _texts(path)
    assert lines[0] == "产品 | 价格"
    assert lines[1] == "产品：笔记本；价格：5000"


def test_json_is_flattened(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"合同": {"甲方": "甲公司", "金额": 100}},
                               ensure_ascii=False), encoding="utf-8")
    lines = _texts(path)
    assert any("甲方：甲公司" in line for line in lines)
    assert any("金额：100" in line for line in lines)


def test_html_strips_tags_and_scripts(tmp_path):
    path = tmp_path / "t.html"
    path.write_text(
        "<html><head><style>.a{color:red}</style></head>"
        "<body><h1>标题</h1><script>var x=1;</script><p>正文内容</p></body></html>",
        encoding="utf-8",
    )
    lines = _texts(path)
    assert "标题" in lines
    assert "正文内容" in lines
    assert not any("var x" in line for line in lines)
    assert not any("color:red" in line for line in lines)


def test_read_text_handles_utf8(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('{"a": "b"}', encoding="utf-8")
    assert read_text(path).startswith("{")


def test_detect_type_prefers_explicit_argument(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("x", encoding="utf-8")
    assert detect_type(path) == "text"
    assert detect_type(path, "csv") == "csv"
    with pytest.raises(LoaderError):
        detect_type(path, "exe")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "t.exe"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(LoaderError):
        list(load_blocks(path))


def test_missing_file_raises(tmp_path):
    with pytest.raises(LoaderError):
        list(load_blocks(tmp_path / "nope.txt"))


def test_resolve_sources_expands_directory_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "sub" / "skip.exe").write_bytes(b"\x00")
    found = resolve_sources(tmp_path)
    assert [path.name for path in found] == ["a.txt", "b.md"]


def test_resolve_sources_accepts_comma_separated_and_globs(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "c.md").write_text("c", encoding="utf-8")
    names = [path.name for path in resolve_sources(f"{tmp_path / 'a.txt'},{tmp_path / 'b.txt'}")]
    assert names == ["a.txt", "b.txt"]
    globbed = [path.name for path in resolve_sources(str(tmp_path / "*.txt"))]
    assert globbed == ["a.txt", "b.txt"]


def test_resolve_sources_deduplicates(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a", encoding="utf-8")
    assert len(resolve_sources([str(path), str(path)])) == 1
