"""Command integration tests for CodeIntelRuntime."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import pytest
from lsprotocol import types as lsp_types

from src.agents import QueryAgent
from src.commands import handle_command
from src.code_intel.config import CodeIntelConfig, CodeIntelConfigLoadResult
from src.code_intel.providers.lsp.manager import LSPManager
from src.code_intel.runtime import CodeIntelRuntime

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


@dataclass
class _BuildResult:
    indexed: int = 1
    rebuilt: int = 0
    reused: int = 0
    skipped: int = 0
    errors: int = 0
    fts_available: bool = True


def test_index_build_and_clear_prefer_agent_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    config = CodeIntelConfig(cache_dir=str(tmp_path / "cache"))
    loaded = CodeIntelConfigLoadResult(config=config)
    monkeypatch.setattr(
        "src.commands.handlers.index_cmd.load_code_intel_config", lambda: loaded
    )

    class Runtime:
        def __init__(self) -> None:
            self.builds = 0
            self.clears = 0

        async def build_symbol_index(self) -> _BuildResult:
            self.builds += 1
            return _BuildResult()

        async def clear_symbol_index(self) -> int:
            self.clears += 1
            return 2

    runtime = Runtime()
    agent = cast(
        QueryAgent, cast(object, type("Agent", (), {"code_intel_runtime": runtime})())
    )

    assert _run(handle_command("/index build", agent)) is False
    assert runtime.builds == 1
    assert "代码索引构建完成" in capsys.readouterr().out

    assert _run(handle_command("/index clear", agent)) is False
    assert runtime.clears == 1
    assert "删除 2 个文件" in capsys.readouterr().out


def test_lsp_start_uses_runtime_and_does_not_duplicate_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    config = CodeIntelConfig(cache_dir=str(tmp_path / "cache"))
    loaded = CodeIntelConfigLoadResult(config=config)
    monkeypatch.setattr(
        "src.commands.handlers.lsp_cmd.load_code_intel_config", lambda: loaded
    )

    async def fake_ensure(self: LSPManager, language: str) -> object:
        spec = self.spec_for(language)
        return type(
            "Handle",
            (),
            {
                "spec": spec,
                "capabilities": lsp_types.ServerCapabilities(definition_provider=True),
            },
        )()

    monkeypatch.setattr(LSPManager, "ensure_client", fake_ensure)
    runtime = CodeIntelRuntime(workspace, config=config)

    async def scenario() -> None:
        await runtime.initialize()
        agent = cast(
            QueryAgent,
            cast(object, type("Agent", (), {"code_intel_runtime": runtime})()),
        )
        assert await handle_command("/lsp start python", agent) is False
        assert await handle_command("/lsp start python", agent) is False
        assert [
            getattr(provider, "name", "") for provider in runtime.kernel.providers
        ].count("lsp") == 1
        await runtime.close()

    _run(scenario())
    output = capsys.readouterr().out
    assert output.count("语言服务器 LSP 已启动") == 2
