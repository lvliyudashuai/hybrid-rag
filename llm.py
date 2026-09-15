"""大模型调用层：多 provider、流式输出、失败重试、友好报错。

只负责"把 messages 发出去、把文本拿回来"，不掺和检索和提示词构造。
支持：deepseek / openai / claude / ollama（OpenAI 兼容端口）。

没有 API Key 时不在这里抛异常——由上层决定是否退化成离线模拟，
否则用户第一次打开界面就得先配好 Key，体验太差。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # 秒，指数退避的基数


class LLMError(RuntimeError):
    """调用大模型失败，message 是给用户看的中文说明。"""


@dataclass(slots=True)
class LLMReply:
    text: str
    provider: str
    model: str
    usage: dict[str, Any] | None = None
    elapsed_ms: float = 0.0


# provider -> (默认 base_url, 默认模型)
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "claude": {"base_url": "", "model": "claude-3-5-sonnet-latest"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
}


def provider_options() -> dict[str, str]:
    """界面下拉框用的 provider 及展示名。"""
    return {
        "mock": "离线模拟（无需 Key）",
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "claude": "Claude",
        "ollama": "Ollama（本地模型）",
    }


def resolve_settings(provider: str | None = None, model: str | None = None,
                     base_url: str | None = None) -> tuple[str, str, str]:
    """决定这次调用真正用的 provider / 模型 / 地址。

    规则：显式传参 > config（.env 或界面）> provider 默认值。
    """
    provider = (provider or config.LLM_PROVIDER or "mock").lower()
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    resolved_model = model or config.LLM_MODEL or defaults.get("model", "")
    resolved_url = base_url or config.LLM_BASE_URL or defaults.get("base_url", "")
    if provider in PROVIDER_DEFAULTS and model is None and provider != config.LLM_PROVIDER:
        # 换了 provider 但没换模型名时，回退到该 provider 的默认模型
        resolved_model = defaults.get("model", resolved_model)
        resolved_url = defaults.get("base_url", resolved_url)
    return provider, resolved_model, resolved_url


def _friendly_error(exc: Exception, provider: str) -> LLMError:
    """把 SDK 的英文异常翻译成能照着做的中文提示。"""
    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()
    if "authentication" in lowered or "401" in lowered or "invalid api key" in lowered:
        return LLMError(f"{provider} 鉴权失败：API Key 不对或已过期。请检查 .env 或界面里的 Key。")
    if "insufficient" in lowered or "402" in lowered or "quota" in lowered or "balance" in lowered:
        return LLMError(f"{provider} 额度不足：请到平台充值或换一个 Key。")
    if "rate limit" in lowered or "429" in lowered:
        return LLMError(f"{provider} 触发限流（429）：稍等几秒再试，或降低请求频率。")
    if "timeout" in lowered or "timed out" in lowered:
        return LLMError(f"{provider} 请求超时：检查网络，或把 LLM_TIMEOUT 调大（当前 {config.LLM_TIMEOUT}s）。")
    if "connect" in lowered or "ssl" in lowered or "proxy" in lowered:
        return LLMError(f"{provider} 连不上：检查网络 / 代理，以及 LLM_BASE_URL 是否正确。")
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return LLMError(f"{provider} 找不到模型：请确认模型名（当前 LLM_MODEL={config.LLM_MODEL}）。")
    return LLMError(f"{provider} 调用失败（{name}）：{message[:300]}")


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(word in text for word in (
        "timeout", "timed out", "connection", "temporarily", "overloaded",
        "rate limit", "429", "500", "502", "503", "504",
    ))


def _openai_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMError("未安装 openai 包，请执行：pip install openai") from exc
    if not base_url:
        raise LLMError("缺少 base_url：请在配置里填 LLM_BASE_URL。")
    return OpenAI(api_key=api_key or "not-needed", base_url=base_url,
                  timeout=config.LLM_TIMEOUT, max_retries=0)


def complete(messages: Sequence[dict[str, str]], *, provider: str | None = None,
             model: str | None = None, api_key: str | None = None,
             base_url: str | None = None, temperature: float | None = None,
             max_tokens: int | None = None) -> LLMReply:
    """一次性拿到完整回答。"""
    provider, model, base_url = resolve_settings(provider, model, base_url)
    key = api_key if api_key is not None else config.api_key(provider)
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or config.LLM_MAX_TOKENS

    if provider == "mock":
        raise LLMError("mock provider 由上层处理，不应走到这里。")
    if not key and provider != "ollama":
        raise LLMError(
            f"没有配置 {provider} 的 API Key。请在 .env 里填 {_key_env_name(provider)}，"
            "或在界面「设置」里临时填入；也可以把生成模型切成「离线模拟」。"
        )

    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "claude":
                reply = _complete_claude(messages, key, model, temperature, max_tokens)
            else:
                reply = _complete_openai(messages, key, base_url, model,
                                         temperature, max_tokens)
            reply.provider, reply.model = provider, model
            reply.elapsed_ms = (time.perf_counter() - started) * 1000
            return reply
        except LLMError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES and _is_retryable(exc):
                wait = RETRY_BACKOFF ** attempt
                logger.warning("第 %d 次调用失败（%s），%.1fs 后重试", attempt, exc, wait)
                time.sleep(wait)
                continue
            raise _friendly_error(exc, provider) from exc
    raise _friendly_error(last_error or RuntimeError("unknown"), provider)


def stream(messages: Sequence[dict[str, str]], *, provider: str | None = None,
           model: str | None = None, api_key: str | None = None,
           base_url: str | None = None, temperature: float | None = None,
           max_tokens: int | None = None) -> Iterator[str]:
    """流式产出文本片段，用于界面打字机效果。"""
    provider, model, base_url = resolve_settings(provider, model, base_url)
    key = api_key if api_key is not None else config.api_key(provider)
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = max_tokens or config.LLM_MAX_TOKENS

    if provider == "mock":
        raise LLMError("mock provider 由上层处理，不应走到这里。")
    if not key and provider != "ollama":
        raise LLMError(f"没有配置 {provider} 的 API Key。")

    try:
        if provider == "claude":
            yield from _stream_claude(messages, key, model, temperature, max_tokens)
        else:
            yield from _stream_openai(messages, key, base_url, model, temperature, max_tokens)
    except LLMError:
        raise
    except Exception as exc:
        raise _friendly_error(exc, provider) from exc


def _key_env_name(provider: str) -> str:
    names = config._PROVIDER_KEY_ENV.get(provider, ("LLM_API_KEY",))
    return names[0]


def _split_system(messages: Sequence[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic 的 system 是独立参数，这里把 system 消息拆出来。"""
    system = ""
    rest: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") == "system":
            system = f"{system}\n{message.get('content', '')}".strip()
        else:
            rest.append({"role": message.get("role", "user"),
                         "content": message.get("content", "")})
    return system, rest


def _complete_openai(messages, api_key, base_url, model, temperature, max_tokens) -> LLMReply:
    client = _openai_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = getattr(response, "usage", None)
    return LLMReply(
        text=(response.choices[0].message.content or "").strip(),
        provider="", model="",
        usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
    )


def _stream_openai(messages, api_key, base_url, model, temperature, max_tokens):
    client = _openai_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


def _complete_claude(messages, api_key, model, temperature, max_tokens) -> LLMReply:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise LLMError("未安装 anthropic 包，请执行：pip install anthropic") from exc
    system, rest = _split_system(messages)
    client = Anthropic(api_key=api_key, timeout=config.LLM_TIMEOUT, max_retries=0)
    response = client.messages.create(
        model=model, system=system or None, messages=rest,
        max_tokens=max_tokens, temperature=temperature,
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    usage = getattr(response, "usage", None)
    return LLMReply(
        text=text.strip(), provider="", model="",
        usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
    )


def _stream_claude(messages, api_key, model, temperature, max_tokens):
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise LLMError("未安装 anthropic 包，请执行：pip install anthropic") from exc
    system, rest = _split_system(messages)
    client = Anthropic(api_key=api_key, timeout=config.LLM_TIMEOUT, max_retries=0)
    with client.messages.stream(
        model=model, system=system or None, messages=rest,
        max_tokens=max_tokens, temperature=temperature,
    ) as run:
        yield from run.text_stream
