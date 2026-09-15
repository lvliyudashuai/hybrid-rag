"""集中配置：所有可调项都在这里，支持 `.env` 覆盖 + 运行时覆盖。

设计原则：
1. 单一真相来源 —— 其它模块只从这里读配置，不自己 `os.getenv`。
2. 解析失败要报错，不要静默用错值（例如把字符串 "0" 当成 True）。
3. API Key 只从环境变量 / 界面内存读取，绝不写进日志、代码或仓库。

运行时覆盖（Streamlit 侧边栏）用 `apply_runtime(**kwargs)` 改这些模块级变量；
`set_api_key()` 设置的 Key 只存在内存里，不会落盘。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__version__ = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# 显式指定 .env 位置：从任何工作目录启动都能读到同一份配置
load_dotenv(ENV_FILE, override=False)


class ConfigError(ValueError):
    """配置项非法。错误信息里直接给出怎么改。"""


# --------------------------------------------------------------------------
# 读取工具：带类型转换 + 校验
# --------------------------------------------------------------------------
def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name}={value!r} 不是合法布尔值，请写 1/0（或 true/false）。")


def _int(name: str, default: int, *, minimum: int | None = None,
         maximum: int | None = None) -> int:
    value = _raw(name)
    if value is None:
        number = default
    else:
        try:
            number = int(value)
        except ValueError as exc:
            raise ConfigError(f"{name}={value!r} 不是合法整数。") from exc
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name}={number} 太小，最小是 {minimum}。")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{name}={number} 太大，最大是 {maximum}。")
    return number


def _float(name: str, default: float, *, minimum: float | None = None,
           maximum: float | None = None) -> float:
    value = _raw(name)
    if value is None:
        number = default
    else:
        try:
            number = float(value)
        except ValueError as exc:
            raise ConfigError(f"{name}={value!r} 不是合法数字。") from exc
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name}={number} 太小，最小是 {minimum}。")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{name}={number} 太大，最大是 {maximum}。")
    return number


def _path(name: str, default: Path) -> Path:
    value = _raw(name)
    if value is None:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


# --------------------------------------------------------------------------
# 文档与向量库
# --------------------------------------------------------------------------
# DOC_PATH：单个文档；SOURCES：逗号分隔的多个文件/目录（批量入库用）
DOC_PATH = _path("DOC_PATH", BASE_DIR / "rag.docx")
DOC_TYPE = _str("DOC_TYPE", "")
SOURCES = _str("SOURCES", "")

CHROMA_DIR = _path("CHROMA_DIR", BASE_DIR / "chroma_db")
COLLECTION_NAME = _str("COLLECTION_NAME", "rag_kb")

# --------------------------------------------------------------------------
# 分块
# --------------------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 500, minimum=80)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 50, minimum=0)
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 20, minimum=1)
# 把「章节路径」拼在每个块前面再向量化：中文长文档里能明显提升召回
CHUNK_CONTEXT_PREFIX = _bool("CHUNK_CONTEXT_PREFIX", True)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigError(
        f"CHUNK_OVERLAP({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE({CHUNK_SIZE})，否则分块会死循环。"
    )

# --------------------------------------------------------------------------
# 向量模型
# --------------------------------------------------------------------------
# 中文推荐 Qwen/Qwen3-Embedding-0.6B（约 1.2GB，效果好很多）；
# 想省空间/加快启动可换 BAAI/bge-small-zh-v1.5（约 100MB，效果明显下降）
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# auto / cpu / cuda / cuda:0 / mps —— auto 优先用显卡
EMBEDDING_DEVICE = _str("EMBEDDING_DEVICE", "auto")
EMBEDDING_LOCAL_ONLY = _bool("EMBEDDING_LOCAL_ONLY", False)
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 32, minimum=1)
EMBEDDING_MAX_LENGTH = _int("EMBEDDING_MAX_LENGTH", 512, minimum=32)
EMBEDDING_TRUST_REMOTE_CODE = _bool("EMBEDDING_TRUST_REMOTE_CODE", True)
# 只加在"查询"侧的前缀。Qwen3-Embedding 官方推荐写成：
#   Instruct: 给定用户问题，检索能回答它的文档段落\nQuery:
# 向量库里的文档不加前缀，这是非对称检索的正确用法。
EMBEDDING_QUERY_PREFIX = _str("EMBEDDING_QUERY_PREFIX", "")

# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
# vector：纯向量；keyword：纯 BM25；hybrid：两路召回 + RRF 融合（推荐）
RETRIEVAL_MODE = _str("RETRIEVAL_MODE", "hybrid")
TOP_K = _int("TOP_K", 5, minimum=1, maximum=50)
# 融合前每路各取多少个候选
CANDIDATE_K = _int("CANDIDATE_K", 20, minimum=1, maximum=200)
# RRF 平滑常数：越大越弱化「排名第一」的优势，业界常用 60
RRF_K = _int("RRF_K", 60, minimum=1)
# 向量路权重（0~1），关键词路权重为 1-alpha。
# 默认 0.9 是在本仓库示例集上实测最优（见 README 的评估表）；
# 如果你的文档里大量出现编号、法条号、型号等精确词，应该调低到 0.5~0.7。
HYBRID_ALPHA = _float("HYBRID_ALPHA", 0.9, minimum=0.0, maximum=1.0)
# MMR 去重：1.0 = 纯相关性，越小越强调多样性
MMR_LAMBDA = _float("MMR_LAMBDA", 0.7, minimum=0.0, maximum=1.0)
# 默认关闭：实测在"查一个具体事实"的场景下 MMR 会把正确答案挤下去。
# 问"文档都讲了哪些方面"这种需要覆盖度的问题时再打开。
USE_MMR = _bool("USE_MMR", False)
# 低于这个相似度的片段直接丢掉（0 表示不过滤）
SCORE_THRESHOLD = _float("SCORE_THRESHOLD", 0.0, minimum=0.0, maximum=1.0)

# 重排（CrossEncoder）：效果最好，但需要额外下载模型，默认关闭
USE_RERANK = _bool("USE_RERANK", False)
RERANK_MODEL = _str("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TOP_K = _int("RERANK_TOP_K", 20, minimum=1, maximum=100)

# --------------------------------------------------------------------------
# 生成端
# --------------------------------------------------------------------------
LLM_PROVIDER = _str("LLM_PROVIDER", "deepseek")  # deepseek/openai/claude/ollama/mock
LLM_MODEL = _str("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = _str("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024, minimum=64, maximum=32768)
LLM_TIMEOUT = _float("LLM_TIMEOUT", 60.0, minimum=1.0)
LLM_STREAM = _bool("LLM_STREAM", True)
# 拼给模型的参考资料字符上限，防止超长上下文烧 token / 超限报错
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 6000, minimum=500)

_PROVIDER_KEY_ENV = {
    "deepseek": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
    "openai": ("OPENAI_API_KEY", "LLM_API_KEY"),
    "claude": ("ANTHROPIC_API_KEY", "LLM_API_KEY"),
    "ollama": ("OLLAMA_API_KEY", "LLM_API_KEY"),
}


def env_api_key(provider: str | None = None) -> str | None:
    """从环境变量取该 provider 的 Key（找不到返回 None）。

    `LLM_API_KEY` 是通用兜底名，任何 provider 都能用。
    """
    names = _PROVIDER_KEY_ENV.get(provider or LLM_PROVIDER, ("LLM_API_KEY",))
    for name in names:
        value = _raw(name)
        if value:
            return value
    return None


_RUNTIME_API_KEY: str | None = None


def api_key(provider: str | None = None) -> str | None:
    """当前生效的 Key：界面临时输入 > .env。"""
    if _RUNTIME_API_KEY:
        return _RUNTIME_API_KEY
    return env_api_key(provider)


def set_api_key(key: str | None) -> None:
    """设置本次运行使用的 Key（只存在内存里，不落盘）。"""
    global _RUNTIME_API_KEY
    _RUNTIME_API_KEY = (key or "").strip() or None


def mask_secret(secret: str | None, *, keep: int = 4) -> str:
    """把 Key 变成 `sk-1****abcd` 这样的安全展示形式。"""
    if not secret:
        return "（未设置）"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}{'*' * 6}{secret[-keep:]}"


# --------------------------------------------------------------------------
# 日志与其它
# --------------------------------------------------------------------------
LOG_LEVEL = _str("LOG_LEVEL", "INFO")
LOG_FILE = _str("LOG_FILE", "")
TRACE = _bool("TRACE", False)
UPLOAD_DIR = _path("UPLOAD_DIR", BASE_DIR / "uploads")

# REST API（api.py），留空则关闭鉴权，仅建议本机使用
API_HOST = _str("API_HOST", "127.0.0.1")
API_PORT = _int("API_PORT", 8000, minimum=1, maximum=65535)
API_TOKEN = _str("API_TOKEN", "")


# --------------------------------------------------------------------------
# 运行时覆盖
# --------------------------------------------------------------------------
# 只允许改这些字段，避免界面把名字拼错却悄悄生效
RUNTIME_OVERRIDABLE = {
    "DOC_PATH", "DOC_TYPE", "SOURCES", "CHROMA_DIR", "COLLECTION_NAME",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHUNK_CHARS", "CHUNK_CONTEXT_PREFIX",
    "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "EMBEDDING_LOCAL_ONLY",
    "EMBEDDING_BATCH_SIZE", "EMBEDDING_QUERY_PREFIX",
    "RETRIEVAL_MODE", "TOP_K", "CANDIDATE_K", "RRF_K",
    "HYBRID_ALPHA", "MMR_LAMBDA", "USE_MMR", "SCORE_THRESHOLD",
    "USE_RERANK", "RERANK_MODEL", "RERANK_TOP_K", "LLM_PROVIDER", "LLM_MODEL",
    "LLM_BASE_URL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "MAX_CONTEXT_CHARS",
    "LOG_LEVEL", "TRACE",
}


def apply_runtime(**overrides) -> list[str]:
    """把界面上的临时设置写进模块变量，返回被改动的字段名。"""
    changed: list[str] = []
    for key, value in overrides.items():
        if key not in RUNTIME_OVERRIDABLE:
            raise ConfigError(f"{key} 不是可运行时覆盖的配置项。")
        if value is None:
            continue
        if globals().get(key) != value:
            globals()[key] = value
            changed.append(key)
    return changed


def snapshot(*, mask: bool = True) -> dict[str, object]:
    """导出当前配置，便于界面展示或写日志。"""
    data: dict[str, object] = {key: globals().get(key) for key in sorted(RUNTIME_OVERRIDABLE)}
    data["LLM_API_KEY"] = mask_secret(api_key()) if mask else api_key()
    data["ENV_FILE"] = str(ENV_FILE)
    data["VERSION"] = __version__
    return data
