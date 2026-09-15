"""配置解析：类型转换、边界校验、密钥处理。"""
from __future__ import annotations

import pytest

import config


def test_bool_accepts_common_forms(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("FLAG", raw)
        assert config._bool("FLAG", False) is True
    for raw in ("0", "false", "no", "off"):
        monkeypatch.setenv("FLAG", raw)
        assert config._bool("FLAG", True) is False


def test_bool_rejects_garbage(monkeypatch):
    """这是踩过的坑：'0' 曾被当成 True，导致本地模式一直开着。"""
    monkeypatch.setenv("FLAG", "maybe")
    with pytest.raises(config.ConfigError):
        config._bool("FLAG", False)


def test_bool_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("FLAG", raising=False)
    assert config._bool("FLAG", True) is True
    monkeypatch.setenv("FLAG", "   ")
    assert config._bool("FLAG", True) is True


def test_int_range_check(monkeypatch):
    monkeypatch.setenv("NUM", "0")
    with pytest.raises(config.ConfigError):
        config._int("NUM", 5, minimum=1)
    monkeypatch.setenv("NUM", "abc")
    with pytest.raises(config.ConfigError):
        config._int("NUM", 5)


def test_overlap_must_be_smaller_than_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_mask_secret_hides_middle():
    assert config.mask_secret(None) == "（未设置）"
    assert config.mask_secret("short") == "*****"
    masked = config.mask_secret("sk-1234567890abcdef")
    assert masked.startswith("sk-1")
    assert "******" in masked
    assert "7890" not in masked


def test_api_key_prefers_provider_specific_name(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert config.env_api_key("deepseek") == "deepseek-key"
    assert config.env_api_key("openai") == "openai-key"


def test_llm_api_key_is_generic_fallback(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    assert config.env_api_key("deepseek") == "generic-key"


def test_runtime_api_key_wins_but_is_not_persisted(monkeypatch):
    """界面里临时填的 Key 优先；但绝不能因为它被写进 .env。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    config.set_api_key("typed-key")
    try:
        assert config.api_key("deepseek") == "typed-key"
    finally:
        config.set_api_key(None)
    assert config.api_key("deepseek") == "env-key"


def test_apply_runtime_rejects_unknown_key():
    with pytest.raises(config.ConfigError):
        config.apply_runtime(NOT_A_REAL_FIELD=1)


def test_apply_runtime_updates_and_reports():
    original = config.TOP_K
    try:
        changed = config.apply_runtime(TOP_K=original + 3)
        assert changed == ["TOP_K"]
        assert original + 3 == config.TOP_K
        assert config.apply_runtime(TOP_K=original + 3) == []
    finally:
        config.apply_runtime(TOP_K=original)


def test_snapshot_masks_key_by_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnop")
    payload = config.snapshot()
    assert "sk-abcdefghijklmnop" not in str(payload)
