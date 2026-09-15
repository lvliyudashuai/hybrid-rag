'''文档入库（docx / pdf）：识别章节 → 注入元数据 → 分块 → 向量化 → 写入 Chroma 。'''
import re #re 模块能帮你实现“按语义边界切分”
from pathlib import Path  
from typing import Iterable, Iterator #管理超大文档集

from docx import Document
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from vectorstore import create_embeddings

# 用余弦距离，便于把检索距离转换成直观的相似度
_COLLECTION_METADATA = {"hnsw:space": "cosine"}

# 常见一级标题（非编号）：论文 + 法律文书 + 合同
_FIXED_HEADINGS = {
    "abstract", "keywords", "msc", "introduction", "related work",
    "methodology", "experiments", "results", "discussion", "conclusion",
    "references", "acknowledgments", "acknowledgements",
    "author contributions", "appendix", "背景", "引言", "方法",
    "实验", "结论", "参考文献", "致谢",
    # 法律/诉讼类
    "诉讼请求", "事实与理由", "本院认为", "本院查明", "审理查明",
    "判决如下", "裁定如下", "上诉请求", "答辩意见",
    # 合同类
    "鉴于", "定义与解释", "合同标的", "价款与支付", "违约责任",
    "争议解决", "保密条款", "通知与送达", "合同生效", "签署页",
}

_CHAPTER_RE = re.compile(r"^第[0-9零〇一二三四五六七八九十百两]+[章条节回]")
_HEADING_RE = re.compile(
    r"^"                 # 行开头
    r"(\d+(?:\.\d+)*)"   # 编号，如 1 或 2.3.4
    r"[\.、\s：]+"        # 分隔符：英文句点 / 顿号 / 空白 / 中文冒号
    r"(.+)"              # 标题内容
    r"$",                # 行结尾
    re.VERBOSE,
)


def parse_heading(text: str):
    '''把形似论文章节标题的段落解析为 (level, title)，否则返回 None。'''
    low = text.strip().lower()
    if low in _FIXED_HEADINGS:
        return 1, text.strip()
    if low.endswith(":") and low[:-1] in _FIXED_HEADINGS:
        return 1, text.strip()
    if low.endswith("：") and low[:-1] in _FIXED_HEADINGS:
            return 1, text.strip()
    if _CHAPTER_RE.match(text):
        return 1, text.strip()
    m = _HEADING_RE.match(text)
    if not m:
        return None
    num, title = m.group(1), m.group(2).strip()
    # 编号章节：长度要合理，避免把 "1." 这种列表项误判
    if len(title) <= 120 and len(text) <= 160:
        return num.count(".") + 1, f"{num}. {title}"
    return None


def _num_of(title: str) -> str | None:
    m = re.match(r"^(\d+(?:\.\d+)*)", title)
    return m.group(1) if m else None


def _parent_num(num: str) -> str | None:
    parts = num.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


# ---------- 公共的标题栈识别引擎 ----------
def sections_from_lines(lines: Iterable[str]) -> Iterator[tuple[str, list[str]]]:
    '''把一行行文本按标题栈切分成 (章节路径, 段落文本列表)。

    docx 和 pdf 都走这里：只是"喂进来的行"来源不同。
    未落入任何标题的段落会被归入 "文档信息"。对编号章节会校验父级是否存在，
    若文档缺了上一级标题则独立成章，避免出现错误的嵌套路径。
    '''
    heading_stack: list[str] = []
    buffer: list[str] = []

    for raw in lines:
        text = (raw or "").strip()
        if not text:
            continue
        parsed = parse_heading(text)
        if parsed:
            if buffer:
                path = " > ".join(heading_stack) if heading_stack else "文档信息"
                yield path, list(buffer)
            level, title = parsed
            num = _num_of(title)
            if num is None or level == 1:
                heading_stack = [title]
            else:
                parent = _parent_num(num)
                parent_match = (
                    parent is not None
                    and len(heading_stack) >= level - 1
                    and _num_of(heading_stack[level - 2]) == parent
                )
                if parent_match:
                    heading_stack = heading_stack[: level - 1] + [title]
                else:
                    # 文档缺父级标题，独立成章，避免错误嵌套
                    heading_stack = [title]
            buffer = [text]
        else:
            buffer.append(text)

    if buffer:
        path = " > ".join(heading_stack) if heading_stack else "文档信息"
        yield path, list(buffer)


# ---------- docx 加载器 ----------
def iter_sections(doc: Document) -> Iterator[tuple[str, list[str]]]:
    '''从 python-docx 的段落对象里取行，交给公共识别引擎。'''
    yield from sections_from_lines(p.text for p in doc.paragraphs)


# ---------- pdf 加载器 ----------
def iter_pdf_sections(pdf_path: Path) -> Iterator[tuple[str, list[str]]]:
    '''从 PDF 提取文本行，交给公共识别引擎。

    pypdf 逐页提取，再把所有行一次性喂给 sections_from_lines，
    这样跨页的章节栈能连续，不打断标题层级。
    '''
    from pypdf import PdfReader  # 延迟导入：docx 路径不需要 pypdf

    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(text.splitlines())
    yield from sections_from_lines(lines)


def _iter_sections_by_type(path: Path) -> Iterator[tuple[str, list[str]]]:
    '''按 DOC_TYPE / 扩展名选择加载器，统一返回 (章节路径, 文本列表)。'''
    ext = (config.DOC_TYPE or path.suffix.lstrip(".")).lower()
    if ext == "pdf":
        return iter_pdf_sections(path)
    if ext in ("docx", "doc", ""):
        return iter_sections(Document(str(path)))
    raise ValueError(f"不支持的文档类型：{ext!r}（请用 docx 或 pdf）")


def build_chunks(
    sections: Iterable[tuple[str, list[str]]], source_name: str
) -> list[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "\u3000", " ", ""],
    )
    chunks: list[LCDocument] = []
    global_idx = 0
    for path, texts in sections:
        text = "\n".join(texts)
        pieces = splitter.split_text(text)
        for i, seg in enumerate(pieces):
            if not seg.strip():
                continue
            metadata = {
                "source": source_name,
                "section": path,
                "global_chunk_index": global_idx,
                "section_chunk_index": i,
                "chars": len(seg),
            }
            chunks.append(LCDocument(page_content=seg, metadata=metadata))
            global_idx += 1
    return chunks


def _reset_collection():
    '''删除同名 collection，确保每次入库都是干净重建。'''
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass


def ingest_documents(doc_path: Path | None = None) -> int:
    doc_path = Path(doc_path or config.DOC_PATH)
    if not doc_path.exists():
        raise FileNotFoundError(f"找不到文档：{doc_path}")

    print(f"📄 加载文档：{doc_path.name}")
    sections = _iter_sections_by_type(doc_path)
    chunks = build_chunks(sections, source_name=doc_path.name)
    print(f"✂️  共分块 {len(chunks)} 个")

    if not chunks:
        raise ValueError("没有可用的文本块，请检查文档内容。")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _reset_collection()

    embeddings = create_embeddings()
    from langchain_chroma import Chroma
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        collection_metadata=_COLLECTION_METADATA,
    )
    print(f"✅ 向量库已写入：{config.CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    ingest_documents()
