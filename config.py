'''集中配置 RAG 项目，支持用环境变量 / .env 覆盖。'''
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ---------- 文档（docx / pdf）与向量库 ----------
# DOC_PATH: 要入库的文档路径；DOC_TYPE: docx / pdf，留空则按扩展名自动判断
DOC_PATH = Path(os.getenv("DOC_PATH", str(BASE_DIR / "rag.docx")))
DOC_TYPE = os.getenv("DOC_TYPE", "")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db_qwen")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_kb")

# ---------- 分块 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------- 向量模型 ----------
# 中文推荐 Qwen/Qwen3-Embedding-0.6B；若想更轻量可换 BAAI/bge-small-zh-v1.5
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Qwen3-Embedding 在 HuggingFace 上需要 trust_remote_code
EMBEDDING_MODEL_KWARGS = {"trust_remote_code": True}
# 本地已有缓存时可置 1，强制不联网
EMBEDDING_LOCAL_ONLY = os.getenv("EMBEDDING_LOCAL_ONLY", "0") == "1"

# ---------- 生成端 ----------
# deepseek / openai / claude / mock（mock 用于无 API Key 时离线演示）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))