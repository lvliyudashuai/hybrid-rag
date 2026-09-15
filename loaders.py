"""通用文档加载：把 8 种常见格式读成 Block 序列。

支持的格式：docx / pdf / txt / md / html / csv / tsv / xlsx / json
目录、通配符、多个路径都能一次给进来（批量入库）。

设计要点：
- 所有 loader 都产出 Block，位置信息（页码 / 行号）尽量保留，便于做引用溯源。
- 重依赖（pypdf、openpyxl、python-docx）都在函数内部导入，
  这样只用 txt 时不必装全套依赖，CI 里也能跑得动。
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator, Sequence
from html.parser import HTMLParser
from pathlib import Path

from docmodel import Block

# 支持的扩展名 -> 内部类型名
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".docx": "docx",
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".tsv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
}

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1")


class LoaderError(RuntimeError):
    """文档读取失败（格式不支持 / 文件损坏 / 扫描件无文字层）。"""


def detect_type(path: Path, doc_type: str = "") -> str:
    """决定用哪个 loader：显式指定优先，否则按扩展名。"""
    if doc_type:
        explicit = doc_type.strip().lower().lstrip(".")
        if explicit in {"docx", "pdf", "text", "html", "csv", "xlsx", "json"}:
            return explicit
        raise LoaderError(f"不支持的 DOC_TYPE：{doc_type!r}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        raise LoaderError(f"不支持的格式 {ext or '(无扩展名)'}；支持：{supported}")
    return SUPPORTED_EXTENSIONS[ext]


def read_text(path: Path) -> str:
    """按常见中文编码依次尝试读取文本。"""
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# 各格式 loader
# --------------------------------------------------------------------------
def load_docx(path: Path) -> Iterator[Block]:
    """Word：段落 + 表格。表格按「列名：值」重排，合同/论文里的表格才检索得到。"""
    from docx import Document  # 延迟导入

    document = Document(str(path))
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            yield Block(text=text)

    for table_index, table in enumerate(document.tables, 1):
        header: list[str] = []
        for row_index, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if not any(cells):
                continue
            if not header:
                header = cells
                yield Block(text=" | ".join(cells), locator=f"表{table_index} 表头")
                continue
            pairs = [
                f"{header[i]}：{value}"
                for i, value in enumerate(cells)
                if value and i < len(header)
            ]
            text = "；".join(pairs) if pairs else " | ".join(cells)
            yield Block(text=text, locator=f"表{table_index} 第{row_index}行")


def load_pdf(path: Path) -> Iterator[Block]:
    """PDF：逐页提取文字，保留页码。

    扫描件（图片型 PDF）没有文字层，pypdf 提不出内容，这里会明确报错，
    而不是悄悄返回一个空库——那种问题在演示时最难排查。
    """
    from pypdf import PdfReader  # 延迟导入

    reader = PdfReader(str(path))
    produced = 0
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # 个别页损坏不该拖垮整篇文档
            yield Block(text=f"[第 {number} 页解析失败：{exc}]", page=number)
            continue
        for line in text.splitlines():
            if line.strip():
                yield Block(text=line.strip(), page=number)
                produced += 1
    if produced == 0:
        raise LoaderError(
            f"PDF 里没有可提取的文字：{path.name}。它可能是扫描件，需要先做 OCR。"
        )


def load_text(path: Path) -> Iterator[Block]:
    """txt / markdown：逐行。"""
    for line in read_text(path).splitlines():
        if line.strip():
            yield Block(text=line.rstrip())


class _TextExtractor(HTMLParser):
    """标准库 HTML 解析：丢掉脚本样式，块级标签当换行。"""

    _BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "table", "ul", "ol", "blockquote",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)

    def lines(self) -> list[str]:
        text = "".join(self.parts)
        return [line.strip() for line in text.splitlines() if line.strip()]


def load_html(path: Path) -> Iterator[Block]:
    """网页：正文提出来，标签丢掉。"""
    parser = _TextExtractor()
    parser.feed(read_text(path))
    for line in parser.lines():
        yield Block(text=line)


def load_csv(path: Path) -> Iterator[Block]:
    """CSV/TSV：每一行转成「列名：值；列名：值」，否则一行数据检索不到列含义。"""
    raw = read_text(path)
    if not raw.strip():
        return
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        pass  # 嗅探失败就用默认分隔符

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return
    header = [cell.strip() for cell in rows[0]]
    yield Block(text=" | ".join(header), locator="表头")
    for index, row in enumerate(rows[1:], 2):
        pairs = [
            f"{header[i]}：{value.strip()}"
            for i, value in enumerate(row)
            if value.strip() and i < len(header)
        ]
        text = "；".join(pairs) if pairs else " | ".join(row)
        if text.strip():
            yield Block(text=text, locator=f"第{index}行")


def load_xlsx(path: Path) -> Iterator[Block]:
    """Excel：每个工作表按「列名：值」展开。"""
    from openpyxl import load_workbook  # 延迟导入

    workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            header: list[str] = []
            for index, row in enumerate(sheet.iter_rows(values_only=True), 1):
                cells = ["" if cell is None else str(cell).strip() for cell in row]
                if not any(cells):
                    continue
                if not header:
                    header = cells
                    yield Block(text=f"[{sheet.title}] " + " | ".join(cells),
                                locator=f"{sheet.title} 表头")
                    continue
                pairs = [
                    f"{header[i]}：{value}"
                    for i, value in enumerate(cells)
                    if value and i < len(header)
                ]
                text = "；".join(pairs) if pairs else " | ".join(cells)
                yield Block(text=text, locator=f"{sheet.title} 第{index}行")
    finally:
        workbook.close()


def _flatten_json(value, prefix: str = "") -> Iterator[tuple[str, str]]:
    """把嵌套 JSON 拍平成 (路径, 值)，便于逐行检索。"""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_json(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            yield prefix, "、".join(str(item) for item in value)
        else:
            for index, item in enumerate(value):
                yield from _flatten_json(item, f"{prefix}[{index}]")
    else:
        yield prefix, "" if value is None else str(value)


def load_json(path: Path) -> Iterator[Block]:
    """JSON：拍平成 `路径：值` 逐行输出。"""
    data = json.loads(read_text(path))
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values()))
    for key, value in _flatten_json(data):
        if value:
            yield Block(text=f"{key}：{value}")


_LOADERS = {
    "docx": load_docx,
    "pdf": load_pdf,
    "text": load_text,
    "html": load_html,
    "csv": load_csv,
    "xlsx": load_xlsx,
    "json": load_json,
}


def load_blocks(path: Path, doc_type: str = "") -> Iterator[Block]:
    """按类型选 loader，统一返回 Block 序列。"""
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"找不到文件：{path}")
    kind = detect_type(path, doc_type)
    loader = _LOADERS[kind]
    try:
        yield from loader(path)
    except LoaderError:
        raise
    except Exception as exc:
        raise LoaderError(f"读取 {path.name} 失败：{exc}") from exc


# --------------------------------------------------------------------------
# 批量来源解析
# --------------------------------------------------------------------------
def resolve_sources(specs: str | Path | Sequence[str | Path]) -> list[Path]:
    """把「文件 / 目录 / 通配符 / 逗号分隔的列表」统一展开成文件列表。

    目录会递归扫描所有支持的格式；结果去重并排序，保证多次运行顺序一致。
    """
    if isinstance(specs, (str, Path)):
        raw_items = [specs]
    else:
        raw_items = list(specs)

    candidates: list[str] = []
    for item in raw_items:
        text = str(item)
        candidates.extend(part.strip() for part in re.split(r"[,\n;]", text) if part.strip())

    base = Path.cwd()
    found: list[Path] = []
    for item in candidates:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (base / path)
        if path.is_dir():
            found.extend(
                child for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.exists():
            found.append(path)
        else:
            # 支持 ./docs/*.pdf 这类通配符
            parent = path.parent if str(path.parent) != "." else base
            pattern = path.name
            if parent.exists():
                found.extend(sorted(child for child in parent.glob(pattern) if child.is_file()))

    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(str(path.resolve()), path)
    return [unique[key] for key in sorted(unique)]
