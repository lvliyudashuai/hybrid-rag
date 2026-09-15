"""文档数据模型：Block（一行文本 + 位置）与 Section（章节路径 + 内容）。

所有格式的文档先被读成 Block 序列，再由标题引擎聚成 Section，
这样「读文档」和「识别章节」两件事互不干扰，加新格式只要写一个 loader。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Block:
    """最小文本单位：一行（或表格一行/PDF 的一行）。"""

    text: str
    page: int | None = None      # PDF 的页码，从 1 开始；其它格式为 None
    locator: str = ""            # 更细的位置描述，如 "第 3 张表" / "sheet1 第 5 行"

    def with_page(self, page: int | None) -> Block:
        return Block(text=self.text, page=page if page is not None else self.page,
                     locator=self.locator)


@dataclass(slots=True)
class Section:
    """一个章节：路径形如 `第三回 > 3.1 数据集`，含它的全部文本行。"""

    path: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)

    @property
    def pages(self) -> list[int]:
        seen: list[int] = []
        for block in self.blocks:
            if block.page is not None and block.page not in seen:
                seen.append(block.page)
        return seen

    def page_label(self) -> str:
        """给界面/引用用的页码描述，如 `p.12-13`；没有页码则返回空串。"""
        pages = self.pages
        if not pages:
            return ""
        if len(pages) == 1:
            return f"p.{pages[0]}"
        return f"p.{pages[0]}-{pages[-1]}"
