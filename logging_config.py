"""统一日志：控制台可读、可选写文件、顺便修好 Windows 控制台中文乱码。

Windows 上 stdout 被重定向时会退回 GBK，直接 print 中文/emoji 会抛
UnicodeEncodeError。这里把输出流强制成 UTF-8（errors="replace"），
所以无论从命令行还是从 IDE 跑，中文都不会变成问号。
"""
from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path

import config

_CONFIGURED = False


def _make_utf8(stream):
    with contextlib.suppress(Exception):
        stream.reconfigure(encoding="utf-8", errors="replace")
    return stream


def setup_logging(level: str | None = None, log_file: str | None = None,
                  *, force: bool = False) -> None:
    """初始化根 logger。重复调用不会叠加 handler。"""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = (level or config.LOG_LEVEL or "INFO").upper()
    handlers: list[logging.Handler] = [
        logging.StreamHandler(_make_utf8(sys.stdout)),
    ]

    target = log_file if log_file is not None else config.LOG_FILE
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # 第三方库的 INFO 日志太吵，压到 WARNING
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers",
                  "transformers", "huggingface_hub", "filelock", "jieba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
