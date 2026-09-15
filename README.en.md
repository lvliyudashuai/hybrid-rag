# RAG Document Q&A Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Chroma](https://img.shields.io/badge/Vector%20Store-Chroma-4B8BBE.svg)](https://www.trychroma.com/)

> 中文版: [README.md](README.md)

An end-to-end Retrieval-Augmented Generation (RAG) demo: point it at a local document (`.docx` or `.pdf`), and it runs **document parsing -> section metadata injection -> chunking -> embedding -> similarity retrieval -> LLM answer generation**, with both an interactive Streamlit UI and a CLI entry point.

> A good starting point for learning the full RAG pipeline or building a personal knowledge-base Q&A system.

## Features

- **Multiple formats**: `.docx` / `.pdf`, selected by file extension or `DOC_TYPE` in `.env`.
- **Section metadata injection**: recognises heading levels (e.g. `3.3 Section title`) and stores the section path in every chunk, so retrieved results are traceable.
- **Chinese embedding**: ships with the lightweight `BAAI/bge-small-zh-v1.5`; switch to `Qwen/Qwen3-Embedding-0.6B` for better Chinese semantics.
- **Retrieval + generation loop**: top-K passages are passed to DeepSeek / OpenAI / Claude; with no API key it degrades to an offline mock answer.
- **Similarity display**: converts Chroma cosine distance into an intuitive similarity percentage.
- **Interactive UI**: a Streamlit page to tune top-K, temperature and provider, and to expand each retrieved chunk.
- **Command line**: one-shot ingest and Q&A.

## Layout

```
rag_demo/
|-- config.py             # central config (overridable by .env)
|-- ingest.py             # parse + section detection + chunk + embed
|-- vectorstore.py        # embedding & Chroma wrapper (with model cache)
|-- query.py              # retrieval + generation
|-- app.py                # Streamlit UI
|-- rag.py                # CLI entry
|-- requirements.txt
|-- .env.example          # config template
`-- rag.docx              # sample document
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> In mainland China, use a mirror: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
>
> If HuggingFace is unreachable, set `HF_ENDPOINT=https://hf-mirror.com` in `.env`.

### 2. Configure (optional)

Copy the template:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

By default it uses a locally cached Chinese embedding model and offline answers, so **no API key is required for a demo**.

To get real generated answers, edit `.env`:

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-xxxx
```

OpenAI and Claude are also supported - see `.env.example`. `.env` is git-ignored; only `.env.example` is committed.

### 3. Build the vector store

```bash
python ingest.py
```

The embedding model is downloaded on first run (`BAAI/bge-small-zh-v1.5`, ~192MB); the store is written to `chroma_db_qwen/`.

### 4. Run

Streamlit UI:

```bash
streamlit run app.py
```

CLI:

```bash
python rag.py "What is this document about?"
```

On Windows you can also double-click `run_demo.bat` (it prefers a local `.venv`, otherwise uses `python` from PATH).

## FAQ

- **Nothing retrieved?** Run `python ingest.py` first and make sure ingest succeeded.
- **Switch embedding model?** Change `EMBEDDING_MODEL` in `.env` and re-run `python ingest.py` (different dimensions require rebuilding).
- **Can it run offline?** Yes. Without a key it uses the mock provider and answers from the retrieved context with sources.
- **Does it need network?** Only for the first model download; with a cache set `EMBEDDING_LOCAL_ONLY=1` to stay offline.

## Roadmap

- Asymmetric query/passage instructions for Qwen3-Embedding.
- Hybrid retrieval (BM25 + vector) and reranking.
- Document upload in the UI for a general-purpose Q&A app.

## License

MIT - see [LICENSE](LICENSE).