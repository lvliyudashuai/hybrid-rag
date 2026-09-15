"""REST API：把 RAG 能力暴露成 HTTP 接口，方便接到别的系统里。

启动：
    python api.py                       # 默认 127.0.0.1:8000
    uvicorn api:app --host 0.0.0.0 --port 8000

鉴权：`.env` 里设 `API_TOKEN=xxx` 后，所有 /v1 接口都要带 `X-API-Key: xxx`。
不设就是开放访问（只建议本机 / 内网使用）。

接口：
    GET  /health          健康检查
    GET  /v1/stats        向量库状态
    POST /v1/search       只检索，不生成（调试检索质量时最常用）
    POST /v1/query        检索 + 生成，返回答案和引用
    POST /v1/ingest       入库指定文件/目录
"""
from __future__ import annotations

import logging
import time
from typing import Any

import config
import logging_config

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少依赖：pip install fastapi uvicorn\n"
        f"（原始错误：{exc}）"
    ) from exc

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG 文档问答 API",
    description="基于 Chroma + 混合检索 + 大模型的通用文档问答服务。",
    version=config.__version__,
)


def require_token(x_api_key: str | None = Header(default=None)) -> None:
    """配置了 API_TOKEN 才校验；没配就放行，方便本机调试。"""
    if config.API_TOKEN and x_api_key != config.API_TOKEN:
        raise HTTPException(status_code=401, detail="X-API-Key 不正确")


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=config.TOP_K, ge=1, le=50)
    mode: str = Field(default=config.RETRIEVAL_MODE,
                      description="vector / keyword / hybrid")
    use_rerank: bool | None = None
    use_mmr: bool | None = None


class QueryRequest(SearchRequest):
    provider: str | None = Field(default=None, description="覆盖 LLM_PROVIDER")
    model: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class IngestRequest(BaseModel):
    sources: str = Field(..., description="文件/目录，逗号分隔")
    reset: bool = Field(default=True, description="True 清空重建，False 增量追加")


def _hit_payload(hit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "score": round(hit.score, 6),
        "similarity": None if hit.similarity is None else round(hit.similarity, 4),
        "keyword_score": None if hit.keyword_score is None else round(hit.keyword_score, 4),
        "rerank_score": hit.rerank_score,
        "source": hit.source,
        "section": hit.section,
        "page": hit.page,
        "file": hit.source_file,
        "text": hit.text,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    import vectorstore

    stats = vectorstore.collection_stats()
    return {
        "status": "ok",
        "version": config.__version__,
        "collection": stats["collection"],
        "chunks": stats["count"],
        "provider": config.LLM_PROVIDER,
        "has_api_key": bool(config.api_key()),
    }


@app.get("/v1/stats", dependencies=[Depends(require_token)])
def stats() -> dict[str, Any]:
    import vectorstore

    return {
        "stats": vectorstore.collection_stats(),
        "collections": vectorstore.list_collections(),
        "config": {
            "retrieval_mode": config.RETRIEVAL_MODE,
            "top_k": config.TOP_K,
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
        },
    }


@app.post("/v1/search", dependencies=[Depends(require_token)])
def search_endpoint(payload: SearchRequest) -> dict[str, Any]:
    from retrieval import Trace, search

    trace = Trace()
    started = time.perf_counter()
    hits = search(payload.question, payload.k, mode=payload.mode,
                  use_rerank=payload.use_rerank, use_mmr=payload.use_mmr,
                  trace=trace)
    return {
        "question": payload.question,
        "mode": payload.mode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "trace": trace.to_dict(),
        "hits": [_hit_payload(hit) for hit in hits],
    }


@app.post("/v1/query", dependencies=[Depends(require_token)])
def query_endpoint(payload: QueryRequest) -> dict[str, Any]:
    import query

    result = query.ask(payload.question, payload.k, history=payload.history,
                       provider=payload.provider, model=payload.model,
                       mode=payload.mode)
    return {
        "question": result.question,
        "answer": result.text,
        "used_llm": result.used_llm,
        "provider": result.provider,
        "model": result.model,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "error": result.error,
        "sources": [
            {"rank": hit.rank, "section": hit.section, "page": hit.page,
             "file": hit.source_file, "similarity": hit.similarity}
            for hit in result.hits
        ],
        "hits": [_hit_payload(hit) for hit in result.hits],
    }


@app.post("/v1/ingest", dependencies=[Depends(require_token)])
def ingest_endpoint(payload: IngestRequest) -> dict[str, Any]:
    import ingest

    try:
        report = ingest.ingest_sources(payload.sources, reset=payload.reset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "summary": report.summary(),
        "chunks": report.chunks,
        "seconds": round(report.seconds, 2),
        "files": [
            {"path": item.path, "chunks": item.chunks, "error": item.error}
            for item in report.files
        ],
    }


def main() -> None:
    logging_config.setup_logging()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("缺少依赖：pip install uvicorn") from exc
    print(f"启动 API：http://{config.API_HOST}:{config.API_PORT}/docs")
    if not config.API_TOKEN:
        print("提示：未设置 API_TOKEN，接口当前无鉴权，请不要暴露到公网。")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


if __name__ == "__main__":
    main()
