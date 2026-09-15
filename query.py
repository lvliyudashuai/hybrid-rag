"""检索 + 生成闭环：从 Chroma 召回相关文档块，再交给 LLM 生成答案。"""

from langchain_core.documents import Document

import config
from vectorstore import get_vectorstore

SYSTEM_PROMPT = (
    "你是一个严谨的中文知识问答助手。请仅依据给定的参考资料回答用户问题。"
    "如果资料不足以回答，请直接说明“资料中未找到相关内容”，不要编造。"
    "回答应条理清晰、简洁、使用中文。"
)

# 各 provider 的默认地址 / 模型；若配置了 .env 则以 .env 为准
_PROVIDER_DEFAULTS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "claude": {"base_url": "", "model": "claude-3-5-sonnet-latest"},
}


def retrieve(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """余弦相似度检索，返回 [(Document, score)]，score 为余弦距离，越小越相关。"""
    vs = get_vectorstore()
    k = k or config.TOP_K
    return vs.similarity_search_with_score(query, k=k)


def _format_context(docs: list[Document]) -> str:
    """把检索到的文档块拼成带章节来源的上下文。"""
    blocks = []
    for i, doc in enumerate(docs, 1):
        section = doc.metadata.get("section", "未知章节")
        blocks.append(f"[{i}] 来源章节：{section}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_messages(query: str, docs: list[Document]) -> list[dict[str, str]]:
    context = _format_context(docs)
    user = (
        f"请根据下面的参考资料回答问题。\n\n"
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _mock_generate(query: str, docs: list[Document]) -> str:
    """无 API Key 时用于离线演示，返回基于检索结果的模拟回答。"""
    if not docs:
        return "⚠️ 未检索到相关文档，无法回答。"
    top = docs[0]
    section = top.metadata.get("section", "未知章节")
    snippet = top.page_content.strip().replace("\n", " ")[:180]
    return (
        f"（离线模拟回答，未调用真实模型）\n\n"
        f"与你问题最相关的内容来自「{section}」：\n\n> {snippet}\n\n"
        f"共命中 {len(docs)} 个片段，可在下方“检索到的片段”中查看详情。\n"
        f"如需真实生成，请在 .env 或界面填入 DeepSeek/OpenAI API Key。"
    )


def generate(query: str, docs: list[Document], provider: str | None = None,
             api_key: str | None = None) -> str:
    """按 provider 调用 LLM 生成答案；provider 缺省读 config。"""
    provider = provider or config.LLM_PROVIDER
    api_key = api_key or config.LLM_API_KEY
    messages = build_messages(query, docs)

    if provider == "mock":
        return _mock_generate(query, docs)
    if not api_key:
        return _mock_generate(query, docs)

    if provider in {"deepseek", "openai"}:
        return _call_openai_compatible(messages, api_key, provider)
    if provider == "claude":
        return _call_claude(messages, api_key)
    return f"❓ 未知的 LLM_PROVIDER：{provider}，请检查配置。"


def _call_openai_compatible(messages: list[dict[str, str]], api_key: str,
                            provider: str) -> str:
    from openai import OpenAI

    defaults = _PROVIDER_DEFAULTS.get(provider, {})
    base_url = config.LLM_BASE_URL if provider == config.LLM_PROVIDER else defaults.get("base_url")
    model = config.LLM_MODEL if provider == config.LLM_PROVIDER else defaults.get("model")
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _call_claude(messages: list[dict[str, str]], api_key: str) -> str:
    try:
        from anthropic import Anthropic

        defaults = _PROVIDER_DEFAULTS.get("claude", {})
        model = config.LLM_MODEL if config.LLM_PROVIDER == "claude" else defaults.get("model")
        client = Anthropic(api_key=api_key)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        msg = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )
        return msg.content[0].text if msg.content else ""
    except ImportError:
        return "未安装 anthropic 包，请先执行 pip install anthropic。"


def provider_options() -> dict[str, str]:
    """Streamlit 下拉框可用的 provider 及展示文案。"""
    return {
        "mock": "离线模拟（无需 Key）",
        "deepseek": "DeepSeek（OpenAI 兼容）",
        "openai": "OpenAI",
        "claude": "Claude（需 anthropic）",
    }

