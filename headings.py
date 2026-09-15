"""标题识别引擎：把一串文本行按「标题栈」切成若干 Section。

docx / pdf / txt / markdown / html / 表格 全都走这一份逻辑，
区别只在「行从哪里来」。识别不出标题也不会报错——所有内容会落到
「文档信息」一节，照样能被检索到（通用性优先，不强制文档必须有结构）。

想适配新文体（法律、合同、病历……）时，只需往 `_FIXED_HEADINGS` 里加词，
或者再加一条正则。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from docmodel import Block, Section

# --------------------------------------------------------------------------
# 规则 1：无编号的固定标题词（论文 / 法律 / 合同 / 常见公文）
# --------------------------------------------------------------------------
_FIXED_HEADINGS: set[str] = {
    # 论文 / 技术报告
    "abstract", "keywords", "msc", "introduction", "related work",
    "methodology", "methods", "experiments", "results", "discussion",
    "conclusion", "conclusions", "references", "acknowledgments",
    "acknowledgements", "author contributions", "appendix",
    "背景", "引言", "方法", "实验", "结论", "参考文献", "致谢", "摘要", "目录",
    # 法律 / 诉讼文书
    "诉讼请求", "事实与理由", "本院认为", "本院查明", "审理查明", "判决如下",
    "裁定如下", "上诉请求", "答辩意见", "争议焦点", "证据清单", "法律依据",
    # 合同
    "鉴于", "定义与解释", "合同标的", "价款与支付", "违约责任", "争议解决",
    "保密条款", "通知与送达", "合同生效", "签署页", "权利义务", "免责条款",
    # 公文 / 通用
    "总则", "分则", "附则", "概述", "总结", "附录",
}

# 规则 2：中文编号标题 —— 第X回/章/节/条/部分/篇
_CHAPTER_RE = re.compile(r"^第[0-9零〇一二三四五六七八九十百千万两]+[回章节条篇部]")

# 规则 3：阿拉伯数字编号标题 —— 1 / 1.2 / 1.2.3
_NUMBERED_RE = re.compile(
    r"""^
    (\d+(?:\.\d+)*)      # 编号
    [\.、\s：:]+          # 分隔符
    (.+?)                # 标题内容（非贪婪，避免吞掉后面的正文）
    $""",
    re.VERBOSE,
)

# 规则 4：Markdown 标题 —— # / ## / ###
_MARKDOWN_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def parse_heading(text: str) -> tuple[int, str] | None:
    """把一行文本解析成 (层级, 标题)，不像标题就返回 None。

    层级从 1 开始，用于决定它在标题栈里的位置。
    """
    stripped = text.strip()
    if not stripped:
        return None
    lowered = stripped.lower()

    # 固定词：允许结尾带中英文冒号
    bare = lowered.rstrip(":：").strip()
    if bare in _FIXED_HEADINGS:
        return 1, stripped.rstrip(":：").strip()

    if _CHAPTER_RE.match(stripped):
        return 1, stripped

    markdown = _MARKDOWN_RE.match(stripped)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    numbered = _NUMBERED_RE.match(stripped)
    if numbered:
        num, title = numbered.group(1), numbered.group(2).strip()
        # 限制长度，避免把「1. 这行其实是一句正文」误判成标题
        if title and len(title) <= 120 and len(stripped) <= 160:
            # 直接用原文当标题，不要重新拼 "1.1. 标题"——多出来的点号会污染章节路径
            return num.count(".") + 1, stripped
    return None


def num_of(title: str) -> str | None:
    """取标题里的编号，如 `2.1.3 方法` -> `2.1.3`。"""
    match = re.match(r"^(\d+(?:\.\d+)*)", title)
    return match.group(1) if match else None


def parent_num(num: str) -> str | None:
    """取上级编号：`2.1.3` -> `2.1`；一级标题返回 None。"""
    parts = num.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def sections_from_blocks(blocks: Iterable[Block]) -> Iterator[Section]:
    """按标题栈把 Block 序列切成 Section。

    对编号标题会校验父级是否存在：若文档跳过了上级标题，
    就把当前标题独立成章，避免出现「2.1 挂在 1.3 下面」这种错误嵌套。
    """
    stack: list[str] = []
    buffer: list[Block] = []

    def flush() -> Section:
        path = " > ".join(stack) if stack else "文档信息"
        return Section(path=path, blocks=list(buffer))

    for block in blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        parsed = parse_heading(text)
        if parsed is None:
            buffer.append(block)
            continue

        if buffer:
            yield flush()

        level, title = parsed
        num = num_of(title)
        if num is None or level == 1:
            stack = [title]
        else:
            parent = parent_num(num)
            matched = (
                parent is not None
                and len(stack) >= level - 1
                and num_of(stack[level - 2]) == parent
            )
            stack = stack[: level - 1] + [title] if matched else [title]
        buffer = [Block(text=title, page=block.page, locator=block.locator)]

    if buffer:
        yield flush()


def sections_from_lines(lines: Iterable[str]) -> Iterator[Section]:
    """纯文本行的便捷入口（docx / txt 用）。"""
    yield from sections_from_blocks(Block(text=line) for line in lines)


def heading_keywords() -> set[str]:
    """暴露当前固定标题词表，方便测试或界面展示。"""
    return set(_FIXED_HEADINGS)
