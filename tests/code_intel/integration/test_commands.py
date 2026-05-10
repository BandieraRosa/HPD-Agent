"""Integration tests for code_intel REPL commands."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import pytest
from lsprotocol import types as lsp_types
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from src.agents import QueryAgent
from src.commands import COMMAND_HANDLERS, CommandCompleter, handle_command
from src.commands.details import COMMAND_DETAILS
from src.commands.handlers import run_index, run_lsp
from src.code_intel.config import CodeIntelConfig, code_intel_index_db_path
from src.code_intel.providers.lsp.manager import LSPManager

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _agent() -> QueryAgent:
    return cast(QueryAgent, object())


def _prepare_home_and_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    _ = workspace.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workspace)
    return home, workspace


def test_index_and_lsp_commands_registered_in_registry_and_handlers() -> None:
    assert COMMAND_HANDLERS["/index"] is run_index
    assert COMMAND_HANDLERS["/lsp"] is run_lsp
    assert "/index" in COMMAND_DETAILS
    assert "/lsp" in COMMAND_DETAILS


def test_index_and_lsp_commands_parse_without_real_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home, workspace = _prepare_home_and_workspace(tmp_path, monkeypatch)

    async def fail_ensure_client(self: LSPManager, language: str) -> object:
        _ = self, language
        raise AssertionError("/lsp status must not start a real LSP server")

    monkeypatch.setattr(LSPManager, "ensure_client", fail_ensure_client)

    assert _run(handle_command("/index status", _agent())) is False
    index_output = capsys.readouterr().out
    assert index_output.startswith("代码索引状态")
    assert "未建立" in index_output
    assert not code_intel_index_db_path(workspace, CodeIntelConfig()).exists()
    assert not (home / ".hpagent" / "config.json").exists()

    assert _run(handle_command("/lsp status", _agent())) is False
    lsp_output = capsys.readouterr().out
    assert lsp_output.startswith("语言服务器 LSP 状态")
    assert "未绑定" in lsp_output
    assert "python / pyright: 未运行" in lsp_output

    assert _run(handle_command("/lsp stop", _agent())) is False
    stop_output = capsys.readouterr().out
    assert stop_output.startswith("语言服务器 LSP 停止")

    assert _run(handle_command("/lsp restart python", _agent())) is False
    restart_output = capsys.readouterr().out
    assert restart_output.startswith("语言服务器 LSP 重启")
    assert "未启动真实 server" in restart_output


def test_malformed_config_prints_chinese_error_for_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home, _workspace = _prepare_home_and_workspace(tmp_path, monkeypatch)
    config_path = home / ".hpagent" / "config.json"
    _ = config_path.parent.mkdir(parents=True)
    _ = config_path.write_text(json.dumps({"code_intel": {"lsp": {"languages": []}}}), encoding="utf-8")

    assert _run(handle_command("/index status", _agent())) is False
    index_output = capsys.readouterr().out
    assert index_output.startswith("配置错误")
    assert "提示:" in index_output
    assert "lsp.languages" in index_output

    assert _run(handle_command("/lsp status", _agent())) is False
    lsp_output = capsys.readouterr().out
    assert lsp_output.startswith("配置错误")
    assert "code_intel" in lsp_output


def test_help_and_completion_are_chinese_first_for_new_commands() -> None:
    assert COMMAND_DETAILS["/index"].startswith("代码索引命令")
    assert "不自动构建" in COMMAND_DETAILS["/index"]
    assert COMMAND_DETAILS["/lsp"].startswith("语言服务器 LSP 命令")
    assert "不启动 server" in COMMAND_DETAILS["/lsp"]

    completer = CommandCompleter()
    complete_event = CompleteEvent()
    top_level_completions = list(completer.get_completions(Document("/i"), complete_event))
    top_level = {completion.text for completion in top_level_completions}
    assert "/index" in top_level

    index_completions = list(completer.get_completions(Document("/index "), complete_event))
    assert {completion.text for completion in index_completions} == {"status", "build", "clear"}
    assert any("查看索引状态" in completion.display_meta_text for completion in index_completions)

    lsp_completions = list(completer.get_completions(Document("/lsp r"), complete_event))
    assert [completion.text for completion in lsp_completions] == ["restart"]
    assert "重启指定语言" in lsp_completions[0].display_meta_text

    language_completions = list(completer.get_completions(Document("/lsp restart p"), complete_event))
    assert [completion.text for completion in language_completions] == ["python"]
    assert "语言标识" in language_completions[0].display_meta_text


@dataclass(frozen=True)
class _RuntimeSpec:
    language: str
    name: str


@dataclass(frozen=True)
class _RuntimeHandle:
    spec: _RuntimeSpec
    capabilities: lsp_types.ServerCapabilities


class _Runtime:
    def __init__(self) -> None:
        self.handles: tuple[_RuntimeHandle, ...] = (
            _RuntimeHandle(
                spec=_RuntimeSpec(language="python", name="pyright"),
                capabilities=lsp_types.ServerCapabilities(definition_provider=True, hover_provider=True),
            ),
        )
        self.stopped: list[str | None] = []
        self.restarted: list[str] = []

    async def shutdown(self, language: str | None = None) -> None:
        self.stopped.append(language)

    async def restart(self, language: str) -> object:
        self.restarted.append(language)
        return object()


def test_lsp_status_stop_restart_use_bound_runtime_without_detecting_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _home, _workspace = _prepare_home_and_workspace(tmp_path, monkeypatch)
    runtime = _Runtime()
    agent = cast(QueryAgent, cast(object, type("Agent", (), {"lsp_provider": runtime})()))

    assert _run(handle_command("/lsp status", agent)) is False
    status_output = capsys.readouterr().out
    assert "runtime: 已绑定" in status_output
    assert "python / pyright: 运行中" in status_output
    assert "definition" in status_output and "hover" in status_output

    assert _run(handle_command("/lsp stop python", agent)) is False
    assert runtime.stopped == ["python"]
    assert "已停止: python" in capsys.readouterr().out

    assert _run(handle_command("/lsp restart python", agent)) is False
    assert runtime.restarted == ["python"]
    assert "重启完成: python" in capsys.readouterr().out
