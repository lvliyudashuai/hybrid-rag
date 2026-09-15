"""标题识别：能不能把各种文体的标题认出来，且不误伤正文。"""
from __future__ import annotations

from docmodel import Block
from headings import num_of, parent_num, parse_heading, sections_from_blocks, sections_from_lines


def test_numbered_headings_get_level_from_dots():
    assert parse_heading("1. 总则") == (1, "1. 总则")
    assert parse_heading("2.3.1 模型结构")[0] == 3
    assert num_of("2.3.1 模型结构") == "2.3.1"
    assert parent_num("2.3.1") == "2.3"
    assert parent_num("2") is None


def test_chinese_chapter_headings():
    for text in ("第一回 灵根育孕源流出", "第三章 方法", "第 5 条 保密义务".replace(" ", ""),
                 "第一百二十回 结尾"):
        parsed = parse_heading(text)
        assert parsed is not None and parsed[0] == 1, text


def test_fixed_headings_with_and_without_colon():
    assert parse_heading("参考文献") == (1, "参考文献")
    assert parse_heading("Abstract:") == (1, "Abstract")
    assert parse_heading("违约责任：") == (1, "违约责任")


def test_markdown_headings_use_hash_depth():
    assert parse_heading("# 一级") == (1, "一级")
    assert parse_heading("### 三级") == (3, "三级")


def test_plain_sentence_is_not_a_heading():
    assert parse_heading("这是一句普通正文，不应该被当成标题。") is None
    assert parse_heading("") is None
    # 编号后面跟的是一大段话时，不能误判
    assert parse_heading("1. " + "很长的正文" * 40) is None


def test_sections_are_split_and_nested():
    lines = ["1. 总则", "正文A", "1.1 范围", "正文B", "2. 附则", "正文C"]
    sections = list(sections_from_lines(lines))
    assert [item.path for item in sections] == ["1. 总则", "1. 总则 > 1.1 范围", "2. 附则"]
    assert "正文A" in sections[0].text
    assert "正文B" in sections[1].text


def test_missing_parent_does_not_create_wrong_nesting():
    """文档从 2.1 直接开始（没有 2），不应被挂到上一章的 1 下面。"""
    lines = ["1. 总则", "正文", "2.1 子条款", "内容"]
    sections = list(sections_from_lines(lines))
    assert sections[1].path == "2.1 子条款"


def test_content_before_any_heading_goes_to_document_info():
    sections = list(sections_from_lines(["前言内容", "还有一行"]))
    assert sections[0].path == "文档信息"
    assert "前言内容" in sections[0].text


def test_page_numbers_are_carried_through():
    blocks = [Block(text="第一回 开篇", page=1), Block(text="正文", page=2)]
    sections = list(sections_from_blocks(blocks))
    assert sections[0].pages == [1, 2]
    assert sections[0].page_label() == "p.1-2"


def test_empty_input_yields_nothing():
    assert list(sections_from_lines([])) == []
    assert list(sections_from_lines(["", "   "])) == []
