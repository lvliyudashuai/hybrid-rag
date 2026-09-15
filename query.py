"""RAG 的「生成」环节：检索 -> 构造提示词 -> 调用模型 -> 带引用的回答。

提示词里刻意要求模型标注 [编号]，这样界面上能把每句话对回原文，
这是 RAG 相对"直接问模型"的核心价值：**可溯源**。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import config
import llm
from retrieval import Hit, Trace, search

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "资料中未找到相关内容。"

SYSTEM_PROMPT = (
    "你是一名严谨的文档问答助手，回答必须遵守以下规则：\n"
    "1. 只依据【参考资料】作答，不要使用资料之外的知识，也不要凭印象补充。\n"
    "2. 每个结论后面用方括号标注来源编号，例如：[1][3]。\n"
    "3. 如果资料不足以回答，直接回答「资料中未找到相关内容」，不要编造。\n"
    "4. 用中文回答；先给结论，再给依据；不要整段复述资料。"
)


@dataclass
class Answer:
    """一次问答的完整结果，界面和 CLI 都直接用它。"""

    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    provider: str = "mock"
    model: str = ""
    used_llm: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    trace: Trace | None = None

    @property
    def sources(self) -> list[str]:
        return [hit.locator for hit in self.hits]


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
def retrieve(query: str, k: int | None = None, **kwargs: Any) -> list[Hit]:
    """检索相关片段。语义与 2.0 之前不同：现在返回 Hit（含来源、页码、分数）。"""
    return search(query, k, **kwargs)


# --------------------------------------------------------------------------
# 提示词构造
# --------------------------------------------------------------------------
def format_context(hits: Sequence[Hit], max_chars: int | None = None) -> str:
    """把检索结果拼成带编号的参考资料，并按预算截断。"""
    budget = max_chars or config.MAX_CONTEXT_CHARS
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, 1):
        header = f"[{index}] 来源：{hit.locator}"
        body = hit.text.strip()
        block = f"{header}\n{body}"
        if used + len(block) > budget and blocks:
            blocks.append(f"[{index}]（省略：已达上下文预算 {budget} 字符）")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(question: str, hits: Sequence[Hit],
                   history: Sequence[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """构造发送给大模型的消息列表。"""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    context = format_context(hits) if hits else "（没有检索到任何资料）"
    messages.append({
        "role": "user",
        "content": (
            f"【参考资料】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "请依据参考资料回答，并在结论后标注来源编号。"
        ),
    })
    return messages


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------
def mock_answer(question: str, hits: Sequence[Hit]) -> str:
    """没有 API Key 时的离线演示：直接给出最相关片段，不做任何编造。"""
    if not hits:
        return "⚠️ 没有检索到相关片段。请先确认已建库（`python rag.py --rebuild`）。"
    lines = [
        "🔌 **离线模拟回答**（未调用真实模型）",
        "",
        f"与「{question}」最相关的内容来自 **{hits[0].locator}**：",
        "",
        f"> {hits[0].preview(220)}",
        "",
        f"共命中 {len(hits)} 个片段，展开下面「检索到的片段」可以看到原文。",
        "想得到真正的自然语言回答，请在设置里填入 DeepSeek / OpenAI / Claude 的 API Key。",
    ]
    return "\n".join(lines)


def _should_use_mock(provider: str) -> bool:
    if provider == "mock":
        return True
    return provider != "ollama" and not config.api_key(provider)


def generate(question: str, hits: Sequence[Hit], *,
             history: Sequence[dict[str, str]] | None = None,
             provider: str | None = None, model: str | None = None,
             api_key: str | None = None) -> Answer:
    """检索结果 -> 最终回答（一次性返回）。"""
    provider = (provider or config.LLM_PROVIDER or "mock").lower()
    started = time.perf_counter()
    answer = Answer(question=question, text="", hits=list(hits), provider=provider)

    if _should_use_mock(provider):
        answer.text = mock_answer(question, hits)
        answer.provider = "mock"
        answer.elapsed_ms = (time.perf_counter() - started) * 1000
        return answer

    messages = build_messages(question, hits, history)
    try:
        reply = llm.complete(messages, provider=provider, model=model, api_key=api_key)
        answer.text = reply.text or NO_ANSWER_TEXT
        answer.model = reply.model
        answer.used_llm = True
    except llm.LLMError as exc:
        answer.error = str(exc)
        answer.text = f"⚠️ {exc}"
        logger.warning("生成失败：%s", exc)
    answer.elapsed_ms = (time.perf_counter() - started) * 1000
    return answer


def stream_generate(question: str, hits: Sequence[Hit], *,
                    history: Sequence[dict[str, str]] | None = None,
                    provider: str | None = None, model: str | None = None,
                    api_key: str | None = None) -> Iterator[str]:
    """流式版：逐段吐出文本，界面用它做打字机效果。"""
    provider = (provider or config.LLM_PROVIDER or "mock").lower()
    if _should_use_mock(provider):
        yield mock_answer(question, hits)
        return
    messages = build_messages(question, hits, history)
    yield from llm.stream(messages, provider=provider, model=model, api_key=api_key)


def ask(question: str, k: int | None = None, *, history=None,
        provider: str | None = None, model: str | None = None,
        trace: Trace | None = None, **search_kwargs: Any) -> Answer:
    """一站式问答：检索 + 生成，并记录耗时。"""
    started = time.perf_counter()
    hits = retrieve(question, k, trace=trace, **search_kwargs)
    answer = generate(question, hits, history=history, provider=provider, model=model)
    answer.trace = trace
    answer.elapsed_ms = (time.perf_counter() - started) * 1000
    return answer


def citations_markdown(hits: Sequence[Hit]) -> str:
    """把来源列成 Markdown 清单，可贴在回答后面。"""
    if not hits:
        return ""
    lines = ["**参考来源**", ""]
    for index, hit in enumerate(hits, 1):
        score = (
            f"{hit.similarity:.1%}" if hit.similarity is not None
            else f"BM25 {hit.keyword_score:.1f}" if hit.keyword_score is not None
            else f"{hit.score:.3f}"
        )
        lines.append(f"- [{index}] {hit.locator}（相关度 {score}）")
    return "\n".join(lines)
