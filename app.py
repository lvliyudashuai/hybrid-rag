"""Streamlit 前端：对话 / 知识库 / 检索调试 / 设置 / 关于。

界面只负责「收集参数 + 展示结果」，真正的逻辑都在 retrieval / query / ingest 里，
所以同一套能力命令行和 REST API 也能用。

想双击就用：跑 start.bat（或桌面快捷方式），它等于「起服务 + 开窗口」，见 launch.py。
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import config
import ingest
import llm
from loaders import SUPPORTED_EXTENSIONS, resolve_sources
from query import citations_markdown, stream_generate
from retrieval import Trace, search
from vectorstore import collection_documents, collection_stats, list_collections

st.set_page_config(page_title="通用 RAG 文档问答助手", page_icon="🔍", layout="wide")

MODE_LABELS = {
    "hybrid": "混合检索（向量 + BM25，推荐）",
    "vector": "纯向量（语义）",
    "keyword": "纯关键词（BM25）",
}


# --------------------------------------------------------------------------
# 状态与工具
# --------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_trace", None)


def apply_settings_to_config(settings: dict) -> None:
    """把界面值写进 config（只在本次会话生效，不落盘）。

    密钥走 `set_api_key()` 单独一条路：它不属于 `apply_runtime` 的可覆盖字段
    （那种"顺手塞进去"的写法会让整页报 ConfigError）。
    """
    config.apply_runtime(**settings["runtime"])
    config.set_api_key(settings["api_key"])


@st.cache_data(show_spinner=False, ttl=30)
def _source_overview(collection: str) -> list[dict]:
    """每个来源文件有多少块 —— 用于知识库页的清单。"""
    try:
        _, _, metadatas = collection_documents(collection)
    except Exception:
        return []
    tally: dict[str, int] = {}
    for meta in metadatas:
        name = str(meta.get("source") or "(未知来源)")
        tally[name] = tally.get(name, 0) + 1
    return [{"来源文件": name, "文本块": count}
            for name, count in sorted(tally.items(), key=lambda item: -item[1])]


def render_hits(hits, *, expanded_first: bool = False) -> None:
    """把检索结果显示成可展开的片段列表。"""
    if not hits:
        st.info("没有检索到相关片段。")
        return
    for hit in hits:
        if hit.rerank_score is not None:
            score = f"重排 {hit.rerank_score:.3f}"
        elif hit.similarity is not None:
            score = f"相似度 {hit.similarity:.1%}"
        elif hit.keyword_score is not None:
            score = f"BM25 {hit.keyword_score:.1f}"
        else:
            score = f"{hit.score:.4f}"
        with st.expander(f"[{hit.rank}] {score} ｜ {hit.locator}",
                         expanded=expanded_first and hit.rank == 1):
            st.caption(f"命中通道：{hit.source}")
            st.write(hit.text)


def render_trace(trace_payload: dict | None) -> None:
    if not trace_payload:
        return
    with st.expander("⏱️ 检索链路耗时"):
        for step in trace_payload["steps"]:
            st.write(f"`{step['step']}` — {step['ms']} ms — {step['detail']}")
        st.caption(f"合计 {trace_payload['total_ms']} ms")


# --------------------------------------------------------------------------
# 侧边栏
# --------------------------------------------------------------------------
def sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ 本次会话设置")
        st.caption("改动立即生效，只影响当前会话；要永久保存去「设置」页。")

        providers = llm.provider_options()
        keys = list(providers)
        current = config.LLM_PROVIDER if config.LLM_PROVIDER in keys else keys[0]
        provider = st.selectbox(
            "生成模型", keys, index=keys.index(current),
            format_func=lambda key: providers[key],
        )

        st.text_input(
            "API Key（留空则沿用 .env 里的 Key）",
            value="", type="password", key="api_key_input",
            help="这里输入的 Key 只存在内存中，不会写入 .env，也不会上传。",
        )
        typed_key = (st.session_state.get("api_key_input") or "").strip()
        effective_key = typed_key or config.env_api_key(provider)
        if provider == "mock":
            st.caption("当前使用离线模拟，不会调用任何模型。")
        elif effective_key:
            origin = "本次输入" if typed_key else "来自 .env"
            st.caption(f"🔑 生效中的 Key：`{config.mask_secret(effective_key)}`（{origin}）")
        else:
            st.caption("⚠️ 没有 Key，回答会退化成离线模拟。")

        with st.expander("🎛️ 检索参数", expanded=True):
            mode = st.radio(
                "检索模式", list(MODE_LABELS),
                index=list(MODE_LABELS).index(config.RETRIEVAL_MODE)
                if config.RETRIEVAL_MODE in MODE_LABELS else 0,
                format_func=lambda key: MODE_LABELS[key],
            )
            top_k = st.slider("召回片段数 Top-K", 1, 20, min(config.TOP_K, 20))
            alpha = st.slider(
                "向量 / 关键词 权重 α", 0.0, 1.0, float(config.HYBRID_ALPHA), 0.05,
                help="仅混合检索生效：α 越大越偏向语义，越小越偏向关键词。",
                disabled=mode != "hybrid",
            )
            use_mmr = st.checkbox("MMR 去重（避免重复片段）", value=config.USE_MMR)

        with st.expander("🤖 生成参数"):
            llm_model = st.text_input("模型名", value=config.LLM_MODEL)
            temperature = st.slider("温度", 0.0, 1.0, float(config.LLM_TEMPERATURE), 0.05)
            max_tokens = st.number_input("最大输出 tokens", 64, 8192,
                                         int(config.LLM_MAX_TOKENS), 64)

        st.divider()
        stats = collection_stats()
        st.metric("知识库块数", stats["count"] if stats["exists"] else 0,
                  help=f"集合：{stats['collection']}")
        if not stats["exists"]:
            st.warning("还没有建库，先去「知识库」页入库。")

    return {
        "runtime": {
            "LLM_PROVIDER": provider,
            "RETRIEVAL_MODE": mode,
            "TOP_K": top_k,
            "HYBRID_ALPHA": alpha,
            "USE_MMR": use_mmr,
            "LLM_MODEL": llm_model or config.LLM_MODEL,
            "LLM_TEMPERATURE": float(temperature),
            "LLM_MAX_TOKENS": int(max_tokens),
        },
        "api_key": typed_key or None,
    }


# --------------------------------------------------------------------------
# 各页面
# --------------------------------------------------------------------------
def page_chat() -> None:
    st.subheader("💬 文档问答")
    st.caption("回答只依据你的文档，并标注来源编号；资料里没有的内容会明确说明「未找到」。")

    if st.session_state.messages:
        _, clear_col = st.columns([8, 1])
        if clear_col.button("🧹 清空对话", key="clear_chat",
                            help="只清掉聊天记录，API Key 与检索参数保持不变"):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.rerun()

    for index, message in enumerate(list(st.session_state.messages)):
        with st.chat_message(message["role"]):
            body, tools = st.columns([20, 1])
            with tools:
                if st.button("🗑", key=f"delete_message_{index}", help="删除这一条"):
                    st.session_state.messages.pop(index)
                    st.rerun()
            with body:
                st.markdown(message["content"])
                if message.get("hits"):
                    render_hits(message["hits"])

    question = st.chat_input("基于文档提问，例如：这份文档主要讲了什么？")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        trace = Trace()
        with st.spinner("检索中…"):
            hits = search(question, config.TOP_K, trace=trace)
        if not hits:
            st.error("没有检索到内容。请先在「知识库」页入库文档，或换个问法。")
            return

        holder = st.empty()
        text = ""
        try:
            for piece in stream_generate(question, hits, history=history):
                text += piece
                holder.markdown(text + "▌")
        except llm.LLMError as exc:
            text = f"⚠️ {exc}"
        holder.markdown(text or "（模型没有返回内容）")

        render_hits(hits)
        st.markdown(citations_markdown(hits))
        render_trace(trace.to_dict())

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "hits": hits}
    )
    st.session_state.last_trace = trace.to_dict()


def page_knowledge() -> None:
    st.subheader("📚 知识库管理")
    stats = collection_stats()
    left, middle, right = st.columns(3)
    left.metric("集合", stats["collection"])
    middle.metric("文本块", stats["count"] if stats["exists"] else 0)
    right.metric("库目录", Path(stats["directory"]).name)

    st.divider()
    st.markdown("#### 1️⃣ 上传并入库")
    st.caption(
        "支持 " + "、".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        + "；可以一次传多个文件。"
    )
    uploaded = st.file_uploader(
        "选择文件", accept_multiple_files=True,
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
    )
    col_a, col_b = st.columns([1, 3])
    mode_reset = col_a.radio("入库方式", ["增量追加", "清空重建"], index=0,
                             help="增量只会替换同名文件的旧块，其它文档不受影响。")
    if col_b.button("开始入库", type="primary", disabled=not uploaded):
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for item in uploaded:
            target = config.UPLOAD_DIR / item.name
            target.write_bytes(item.getbuffer())
            saved.append(target)
        with st.spinner(f"正在处理 {len(saved)} 个文件…"):
            report = ingest.ingest_sources([str(path) for path in saved],
                                           reset=(mode_reset == "清空重建"))
        for entry in report.files:
            if entry.ok:
                st.success(f"{Path(entry.path).name}：{entry.chunks} 块（{entry.seconds:.1f}s）")
            else:
                st.error(f"{Path(entry.path).name}：{entry.error}")
        st.caption(report.summary())
        _source_overview.clear()

    st.divider()
    st.markdown("#### 2️⃣ 从路径入库")
    st.caption("填文件或目录（目录会递归扫描支持的格式），多个用逗号分隔。")
    spec = st.text_input("路径", value=str(config.DOC_PATH), label_visibility="collapsed")
    col_c, col_d, _ = st.columns([1, 1, 2])
    if col_c.button("解析路径", type="secondary"):
        found = resolve_sources(spec)
        if found:
            st.write(f"找到 {len(found)} 个文件：")
            for path in found:
                st.write(f"- `{path}`")
        else:
            st.warning("没有找到可入库的文件。")
    if col_d.button("入库这些文件", type="primary"):
        with st.spinner("正在入库…"):
            try:
                report = ingest.ingest_sources(spec, reset=False)
                for entry in report.files:
                    if entry.ok:
                        st.success(f"{Path(entry.path).name}：{entry.chunks} 块")
                    else:
                        st.error(f"{Path(entry.path).name}：{entry.error}")
                st.caption(report.summary())
                _source_overview.clear()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("#### 3️⃣ 已入库的文档")
    overview = _source_overview(config.COLLECTION_NAME)
    if overview:
        st.dataframe(overview, use_container_width=True, hide_index=True)
    else:
        st.info("还没有任何文档。")

    col_f, col_g, col_h = st.columns(3)
    if col_f.button("🧹 清理向量库残留目录"):
        from vectorstore import prune_orphan_segments

        orphans = prune_orphan_segments(dry_run=True)
        if orphans:
            prune_orphan_segments(dry_run=False)
            st.success(f"已清理 {len(orphans)} 个孤儿目录。")
        else:
            st.info("没有需要清理的目录。")
    if col_g.button("🗑️ 清空当前集合"):
        from vectorstore import delete_collection

        delete_collection()
        _source_overview.clear()
        st.success("已清空。")
    with col_h.expander("所有集合"):
        for item in list_collections():
            st.write(f"- `{item['name']}`：{item['count']} 块")


def page_debug() -> None:
    st.subheader("🔬 检索调试")
    st.caption("同一个问题用三种模式各跑一遍，直观看差别；调参后先在这里验证再改默认值。")
    question = st.text_input("测试问题", placeholder="例如：第十五回讲了什么？")
    top_k = st.slider("每种模式各取几条", 1, 10, 5)
    if st.button("跑一遍", type="primary", disabled=not question.strip()):
        columns = st.columns(3)
        for column, mode in zip(columns, ("vector", "keyword", "hybrid"), strict=False):
            with column:
                st.markdown(f"**{MODE_LABELS[mode]}**")
                trace = Trace()
                started = time.perf_counter()
                hits = search(question, top_k, mode=mode, trace=trace)
                elapsed = (time.perf_counter() - started) * 1000
                st.caption(f"{len(hits)} 条 ／ {elapsed:.0f} ms")
                for hit in hits:
                    if hit.similarity is not None:
                        score = f"{hit.similarity:.1%}"
                    elif hit.keyword_score is not None:
                        score = f"{hit.keyword_score:.1f}"
                    else:
                        score = f"{hit.score:.4f}"
                    st.markdown(f"`{hit.rank}.` {score} — {hit.section[:22]}")
                    with st.expander("看原文", expanded=False):
                        st.write(hit.text)


def write_env(updates: dict) -> None:
    """把配置写回 .env，保留原有注释和顺序。"""
    path = config.ENV_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    seen: set = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates and key not in seen:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen and value is not None:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def page_settings() -> None:
    st.subheader("⚙️ 设置")
    st.caption("这里的值决定 .env 与界面默认值；点「保存到 .env」才会写盘。")

    with st.form("settings"):
        st.markdown("**文档与向量库**")
        doc_path = st.text_input("默认文档路径 DOC_PATH", value=str(config.DOC_PATH))
        sources = st.text_input("批量来源 SOURCES（逗号分隔，可留空）", value=config.SOURCES)
        chroma_dir = st.text_input("向量库目录 CHROMA_DIR", value=str(config.CHROMA_DIR))
        collection = st.text_input("集合名 COLLECTION_NAME", value=config.COLLECTION_NAME)
        doc_type = st.selectbox(
            "文档类型 DOC_TYPE", ["自动", "docx", "pdf"],
            index=["", "docx", "pdf"].index(config.DOC_TYPE)
            if config.DOC_TYPE in {"", "docx", "pdf"} else 0,
        )

        st.markdown("**分块**")
        chunk_size = st.number_input("CHUNK_SIZE", 100, 4000, int(config.CHUNK_SIZE), 50)
        chunk_overlap = st.number_input("CHUNK_OVERLAP", 0, 1000, int(config.CHUNK_OVERLAP), 10)
        context_prefix = st.checkbox("块前拼章节路径（提升召回）", value=config.CHUNK_CONTEXT_PREFIX)

        st.markdown("**向量模型**")
        embed_model = st.text_input("EMBEDDING_MODEL", value=config.EMBEDDING_MODEL)
        embed_device = st.text_input("EMBEDDING_DEVICE（auto/cpu/cuda）", value=config.EMBEDDING_DEVICE)
        embed_local = st.checkbox("EMBEDDING_LOCAL_ONLY（只用本地缓存）",
                                  value=config.EMBEDDING_LOCAL_ONLY)

        st.markdown("**检索**")
        mode = st.selectbox(
            "RETRIEVAL_MODE", list(MODE_LABELS),
            index=list(MODE_LABELS).index(config.RETRIEVAL_MODE)
            if config.RETRIEVAL_MODE in MODE_LABELS else 0,
            format_func=lambda key: MODE_LABELS[key],
        )
        candidate_k = st.number_input("CANDIDATE_K（每路候选数）", 5, 200, int(config.CANDIDATE_K), 5)
        use_rerank = st.checkbox("USE_RERANK（CrossEncoder 重排，需额外下载模型）",
                                 value=config.USE_RERANK)
        rerank_model = st.text_input("RERANK_MODEL", value=config.RERANK_MODEL)
        max_context = st.number_input("MAX_CONTEXT_CHARS（上下文预算）", 500, 40000,
                                      int(config.MAX_CONTEXT_CHARS), 500)

        st.markdown("**生成**")
        provider_keys = list(llm.provider_options())
        provider = st.selectbox(
            "LLM_PROVIDER", provider_keys,
            index=provider_keys.index(config.LLM_PROVIDER)
            if config.LLM_PROVIDER in provider_keys else 0,
        )
        llm_model = st.text_input("LLM_MODEL", value=config.LLM_MODEL)
        base_url = st.text_input("LLM_BASE_URL", value=config.LLM_BASE_URL)
        timeout = st.number_input("LLM_TIMEOUT（秒）", 5, 600, int(config.LLM_TIMEOUT), 5)

        st.markdown("**密钥**")
        key_input = st.text_input("API Key（要写盘需勾选下面的确认）", value="", type="password")
        st.caption(f"当前生效的 Key：`{config.mask_secret(config.api_key())}`")

        st.markdown("**运维**")
        log_level = st.selectbox(
            "LOG_LEVEL", ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.LOG_LEVEL)
            if config.LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR"} else 1,
        )
        api_token = st.text_input("API_TOKEN（REST API 鉴权，留空不校验）",
                                  value=config.API_TOKEN, type="password")

        write_secret = st.checkbox("把 API Key 也写入 .env（明文保存，请谨慎）")
        submitted = st.form_submit_button("💾 保存到 .env", type="primary")

    if submitted:
        updates = {
            "DOC_PATH": doc_path,
            "SOURCES": sources,
            "CHROMA_DIR": chroma_dir,
            "COLLECTION_NAME": collection,
            "DOC_TYPE": "" if doc_type == "自动" else doc_type,
            "CHUNK_SIZE": str(chunk_size),
            "CHUNK_OVERLAP": str(chunk_overlap),
            "CHUNK_CONTEXT_PREFIX": "1" if context_prefix else "0",
            "EMBEDDING_MODEL": embed_model,
            "EMBEDDING_DEVICE": embed_device,
            "EMBEDDING_LOCAL_ONLY": "1" if embed_local else "0",
            "RETRIEVAL_MODE": mode,
            "CANDIDATE_K": str(candidate_k),
            "USE_RERANK": "1" if use_rerank else "0",
            "RERANK_MODEL": rerank_model,
            "MAX_CONTEXT_CHARS": str(max_context),
            "LLM_PROVIDER": provider,
            "LLM_MODEL": llm_model,
            "LLM_BASE_URL": base_url,
            "LLM_TIMEOUT": str(timeout),
            "LOG_LEVEL": log_level,
            "API_TOKEN": api_token,
        }
        if write_secret and key_input.strip():
            key_name = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
            }.get(provider, "LLM_API_KEY")
            updates[key_name] = key_input.strip()
        try:
            write_env(updates)
            st.success("已写入 .env，重启后依然生效。")
        except Exception as exc:
            st.error(f"写入失败：{exc}")

    with st.expander("📋 当前完整配置"):
        st.json({key: str(value) for key, value in config.snapshot().items()})


def page_about() -> None:
    st.subheader("ℹ️ 关于这个项目")
    st.markdown(
        f"""
**通用 RAG 文档问答** v{config.__version__}

一条完整的检索增强生成链路：

```
文档 docx / pdf / txt / md / html / csv / xlsx / json
   ↓  loaders.py      按格式取文本，保留页码
   ↓  headings.py     标题栈识别章节
   ↓  chunking.py     结构感知分块 + 章节前缀 + 稳定 ID
   ↓  vectorstore.py  向量化并写入 Chroma（余弦距离）
   ↓  retrieval.py    向量 + BM25 双路召回 → RRF 融合 → MMR → 可选重排
   ↓  query.py        拼提示词 → 调模型 → 带 [编号] 引用的回答
界面 app.py ／ 命令行 rag.py ／ REST 接口 api.py
```

| 想做什么 | 命令 |
| --- | --- |
| 环境自检 | `python rag.py --doctor` |
| 重建索引 | `python rag.py --rebuild` |
| 追加文档 | `python rag.py --ingest ./docs --append` |
| 命令行问答 | `python rag.py "你的问题"` |
| 检索评估 | `python evaluate.py --show-failures` |
| 启动接口 | `python api.py` |

配置项集中在 `.env`，也可以直接在「设置」页修改。
"""
    )


# --------------------------------------------------------------------------
def main() -> None:
    init_state()
    settings = sidebar()
    apply_settings_to_config(settings)

    st.title("🔍 通用 RAG 文档问答助手")
    st.caption("本地文档 → 混合检索 → 带引用的大模型回答。全流程可离线运行。")

    tabs = st.tabs(["💬 对话", "📚 知识库", "🔬 检索调试", "⚙️ 设置", "ℹ️ 关于"])
    with tabs[0]:
        page_chat()
    with tabs[1]:
        page_knowledge()
    with tabs[2]:
        page_debug()
    with tabs[3]:
        page_settings()
    with tabs[4]:
        page_about()


if __name__ == "__main__":
    main()
