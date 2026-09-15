'''Streamlit 前端：检索 + 生成的可交互演示。

侧边栏暴露 .env 里的全部可配置项（侧边栏内可调，点「保存到 .env」写回 .env）：
- 生成 / 检索相关项：改动后立即生效。
- 文档 / 分块 / 向量模型相关项：仅在下一次「重新建库」时生效。
'''
from pathlib import Path

import streamlit as st

import config
import ingest
from query import generate, provider_options, retrieve
from vectorstore import is_similarity

st.set_page_config(page_title="通用 RAG 文档问答助手", page_icon="🔍", layout="wide")


@st.cache_resource
def _get_providers() -> dict[str, str]:
    return provider_options()


def _save_env(overrides: dict) -> None:
    # 把侧边栏设置写回 .env（保留注释与未知项，如 HF_ENDPOINT）
    env_path = config.BASE_DIR / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    result: list = []
    seen: set = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            result.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            if key not in seen:
                result.append(f"{key}={overrides[key]}")
                seen.add(key)
        else:
            result.append(line)
    for key, val in overrides.items():
        if key not in seen:
            result.append(f"{key}={val}")
            seen.add(key)
    env_path.write_text(chr(10).join(result) + chr(10), encoding="utf-8")


def main():
    st.title("🔍 通用 RAG 文档问答助手")
    st.caption("基于本地文档的 全文检索 + 向量召回 + LLM 生成 端到端演示。")

    # ---------- 侧边栏：暴露出 .env 里全部可配置项 ----------
    with st.sidebar:
        st.header("⚙️ 设置")
        st.caption("改动立即生效；点「保存到 .env」可写回 .env，下次启动保留。")

        providers = _get_providers()
        provider = st.selectbox(
            "生成模型",
            list(providers.keys()),
            format_func=lambda k: providers[k],
            index=list(providers.keys()).index(config.LLM_PROVIDER)
            if config.LLM_PROVIDER in providers else 0,
        )

        with st.expander("🛠️ 生成参数", expanded=True):
            api_key = st.text_input(
                "API Key（留空则离线模拟）",
                value="", type="password",
            )
            st.caption("Key 只在本次会话内存中临时使用、不会明文显示；只有你点「保存到 .env」才会把当前值写入本地 .env。")
            llm_model = st.text_input("模型名", value=config.LLM_MODEL)
            base_url = st.text_input("Base URL", value=config.LLM_BASE_URL)
            temperature = st.slider(
                "生成温度", 0.0, 1.0, float(config.LLM_TEMPERATURE), 0.05,
            )
            max_tokens = st.number_input(
                "最大输出 tokens", 64, 8192, int(config.LLM_MAX_TOKENS), 64,
            )

        top_k = st.slider(
            "召回片段数 Top-K", 1, 15,
            int(config.TOP_K) if config.TOP_K <= 15 else 5,
        )

        st.divider()
        st.caption("以下项在点击「重新建库」后才会生效：")

        with st.expander("📄 文档 / 向量库"):
            doc_path = st.text_input("文档路径", value=str(config.DOC_PATH))
            doc_type_map = {"自动": "", "docx": "docx", "pdf": "pdf"}
            doc_type_cur = next(
                (k for k, v in doc_type_map.items() if v == config.DOC_TYPE),
                "自动",
            )
            doc_type = doc_type_map[
                st.selectbox("文档类型", list(doc_type_map),
                             index=list(doc_type_map).index(doc_type_cur))
            ]
            chroma_dir = st.text_input("向量库目录", value=str(config.CHROMA_DIR))
            collection_name = st.text_input("集合名", value=config.COLLECTION_NAME)

        with st.expander("✂️ 分块参数"):
            chunk_size = st.number_input(
                "chunk_size（字符）", 100, 4000, int(config.CHUNK_SIZE), 50,
            )
            chunk_overlap = st.number_input(
                "chunk_overlap（重叠）", 0, 1000, int(config.CHUNK_OVERLAP), 10,
            )

        with st.expander("🧠 向量模型"):
            embed_model = st.text_input("Embedding 模型", value=config.EMBEDDING_MODEL)
            embed_local = st.checkbox(
                "仅本地模式（不联网下载）", value=config.EMBEDDING_LOCAL_ONLY,
            )
            st.caption("改动 embedding 模型后需点击「重新建库」，否则检索会错位。")

        rebuild = st.button("♻️ 重新建库", type="secondary")
        save_env = st.button("💾 保存到 .env", type="secondary")

    # ---------- 把侧边栏选择写回 config（不修改 .env）----------
    config.DOC_PATH = Path(doc_path) if doc_path else config.DOC_PATH
    config.DOC_TYPE = doc_type
    config.CHROMA_DIR = Path(chroma_dir) if chroma_dir else config.CHROMA_DIR
    config.COLLECTION_NAME = collection_name or config.COLLECTION_NAME
    config.CHUNK_SIZE = int(chunk_size)
    config.CHUNK_OVERLAP = int(chunk_overlap)
    config.EMBEDDING_MODEL = embed_model or config.EMBEDDING_MODEL
    config.EMBEDDING_LOCAL_ONLY = bool(embed_local)
    config.LLM_PROVIDER = provider
    config.LLM_MODEL = llm_model or config.LLM_MODEL
    config.LLM_BASE_URL = base_url or config.LLM_BASE_URL
    config.LLM_TEMPERATURE = float(temperature)
    config.LLM_MAX_TOKENS = int(max_tokens)
    # API Key：留空则强制离线模拟（即使 .env 里有 Key 也覆盖）
    config.LLM_API_KEY = api_key or ""
    # ---------- 保存到 .env ----------
    if save_env:
        _save_env({
            "DOC_PATH": str(config.DOC_PATH),
            "DOC_TYPE": config.DOC_TYPE,
            "CHROMA_DIR": str(config.CHROMA_DIR),
            "COLLECTION_NAME": config.COLLECTION_NAME,
            "CHUNK_SIZE": str(config.CHUNK_SIZE),
            "CHUNK_OVERLAP": str(config.CHUNK_OVERLAP),
            "TOP_K": str(config.TOP_K),
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "EMBEDDING_LOCAL_ONLY": "1" if config.EMBEDDING_LOCAL_ONLY else "0",
            "LLM_PROVIDER": config.LLM_PROVIDER,
            "LLM_MODEL": config.LLM_MODEL,
            "LLM_BASE_URL": config.LLM_BASE_URL,
            "LLM_TEMPERATURE": str(config.LLM_TEMPERATURE),
            "LLM_MAX_TOKENS": str(config.LLM_MAX_TOKENS),
            "LLM_API_KEY": config.LLM_API_KEY or "",
            "DEEPSEEK_API_KEY": config.LLM_API_KEY or "",
            "OPENAI_API_KEY": config.LLM_API_KEY or "",
            "ANTHROPIC_API_KEY": config.LLM_API_KEY or "",
        })
        st.sidebar.success("已保存到 .env")


    # ---------- 重新建库 ----------
    if rebuild:
        try:
            with st.spinner("正在重建向量库（读取、分块、向量化），请稍候…"):
                n = ingest.ingest_documents()
            st.success(f"✅ 重建完成，共 {n} 个文本块。现在可以提问了。")
        except Exception as e:
            st.error(f"重建失败：{e}")

    # ---------- 提问 ----------
    st.subheader("📥 提问")
    question = st.text_area("输入你的问题", placeholder="例如：文档主要在讲什么？")
    if st.button("🚀 检索并生成", type="primary"):
        if not question.strip():
            st.warning("请先输入问题。")
            return

        with st.spinner("正在检索相关文档…"):
            docs = retrieve(question, k=top_k)

        if not docs:
            st.error("未检索到相关文档，请确认已先执行 `python ingest.py` 建库。")
            return

        st.markdown("### ✅ 检索到的片段")
        for i, (doc, score) in enumerate(docs, 1):
            section = doc.metadata.get("section", "未知章节")
            sim = is_similarity(score)
            with st.expander(f"片段 {i}｜{section}｜相似度 {sim:.1%}"):
                st.write(doc.page_content)

        with st.spinner("正在生成回答…"):
            answer = generate(
                question, [d for d, _ in docs],
                provider=provider, api_key=api_key or None,
            )

        st.markdown("### 💬 回答")
        st.markdown(answer)


if __name__ == "__main__":
    main()