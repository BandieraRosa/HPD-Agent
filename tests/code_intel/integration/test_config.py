"""Integration tests for code_intel config loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.code_intel.config import code_intel_index_db_path, load_code_intel_config


def test_missing_config_uses_chinese_first_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".hpagent" / "config.json"

    result = load_code_intel_config(config_path)

    assert result.ok is True
    assert result.error is None
    assert result.config.enabled is True
    assert result.config.locale == "zh-CN"
    assert result.config.cache_dir == str(home / ".hpagent" / "index")
    assert result.config.index.auto_build_on_startup is True
    assert result.config.index.respect_gitignore is True
    assert result.config.providers.tree_sitter is True
    assert result.config.providers.text_search is True
    assert result.config.providers.lsp is True
    assert result.config.lsp.enabled is True
    assert result.config.lsp.languages == ["python", "typescript", "javascript"]
    assert result.config.verify.max_repair_rounds == 2
    assert result.config.tools.code_search_default_limit == 20
    assert not config_path.exists()


def test_existing_config_without_code_intel_section_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    config_path = home / ".hpagent" / "config.json"
    _ = config_path.parent.mkdir(parents=True)
    _ = config_path.write_text(json.dumps({"trace_mode": "off"}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    result = load_code_intel_config(config_path)

    assert result.ok is True
    assert result.error is None
    assert result.config.locale == "zh-CN"
    assert result.config.lsp.languages == ["python", "typescript", "javascript"]


def test_custom_config_overrides_defaults_without_writing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    _ = workspace.mkdir()
    config_path = home / ".hpagent" / "config.json"
    _ = config_path.parent.mkdir(parents=True)
    _ = config_path.write_text(
        json.dumps(
            {
                "code_intel": {
                    "enabled": True,
                    "locale": "zh-CN",
                    "cache_dir": str(tmp_path / "cache"),
                    "index": {"auto_build_on_startup": False, "max_file_size_bytes": 1234},
                    "lsp": {"languages": ["Python", "python", "javascript"], "idle_shutdown_minutes": 0},
                    "providers": {"lsp": False},
                    "verify": {"max_repair_rounds": 1},
                    "tools": {"code_search_default_limit": 7},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    result = load_code_intel_config(config_path)
    db_path = code_intel_index_db_path(workspace, result.config)

    assert result.ok is True
    assert result.error is None
    assert result.config.index.auto_build_on_startup is False
    assert result.config.index.max_file_size_bytes == 1234
    assert result.config.lsp.languages == ["python", "javascript"]
    assert result.config.providers.lsp is False
    assert result.config.verify.max_repair_rounds == 1
    assert result.config.tools.code_search_default_limit == 7
    assert db_path.parent.parent == tmp_path / "cache"
    assert not db_path.exists()


def test_malformed_config_returns_chinese_error_instead_of_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    config_path = home / ".hpagent" / "config.json"
    _ = config_path.parent.mkdir(parents=True)
    _ = config_path.write_text('{"code_intel": ', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    result = load_code_intel_config(config_path)

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "malformed_code_intel_config"
    assert result.error.message.startswith("配置错误")
    assert "code_intel" in result.error.message
    assert "JSON 格式无效" in (result.error.detail or "")
    assert result.config.locale == "zh-CN"


def test_invalid_code_intel_section_returns_structured_chinese_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    config_path = home / ".hpagent" / "config.json"
    _ = config_path.parent.mkdir(parents=True)
    _ = config_path.write_text(json.dumps({"code_intel": {"lsp": {"languages": []}}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    result = load_code_intel_config(config_path)

    assert result.ok is False
    assert result.error is not None
    assert result.error.format().startswith("配置错误")
    assert "lsp.languages" in (result.error.detail or "")
    assert "提示:" in result.error.format()
    assert result.config.lsp.languages == ["python", "typescript", "javascript"]
