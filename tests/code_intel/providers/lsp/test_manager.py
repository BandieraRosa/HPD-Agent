"""Tests for LSP manager registry, lazy lifecycle, and executable detection."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar

import pytest
from lsprotocol import types as lsp_types

from src.code_intel.core import ProviderUnavailable
from src.code_intel.providers.lsp import (
    LSPManager,
    LanguageServerSpec,
    default_language_server_specs,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


class _FakeLifecycleClient:
    def __init__(
        self,
        capabilities: lsp_types.ServerCapabilities | None = None,
        *,
        initialize_delay: float = 0.0,
    ) -> None:
        self.capabilities: lsp_types.ServerCapabilities = (
            capabilities or lsp_types.ServerCapabilities(definition_provider=True)
        )
        self.initialize_delay: float = initialize_delay
        self.initialize_calls: int = 0
        self.shutdown_calls: int = 0
        self.close_calls: int = 0
        self.running: bool = True

    async def initialize(
        self,
        *,
        root_uri: str | None = None,
        initialization_options: object | None = None,
    ) -> lsp_types.InitializeResult:
        self.initialize_calls += 1
        _ = initialization_options
        assert root_uri is not None
        if self.initialize_delay > 0:
            await asyncio.sleep(self.initialize_delay)
        return lsp_types.InitializeResult(capabilities=self.capabilities)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def is_running(self) -> bool:
        return self.running


class _RecordingFactory:
    def __init__(self, *, initialize_delay: float = 0.0) -> None:
        self.initialize_delay: float = initialize_delay
        self.clients: list[_FakeLifecycleClient] = []
        self.commands: list[tuple[str, ...]] = []
        self.languages: list[str] = []
        self.request_timeouts: list[float] = []

    def __call__(
        self,
        spec: LanguageServerSpec,
        workspace_root: Path,
        command: Sequence[str],
        *,
        request_timeout_seconds: float,
    ) -> _FakeLifecycleClient:
        assert workspace_root.is_absolute()
        self.commands.append(tuple(command))
        self.languages.append(spec.language)
        self.request_timeouts.append(request_timeout_seconds)
        client = _FakeLifecycleClient(initialize_delay=self.initialize_delay)
        self.clients.append(client)
        return client


def _spec(
    language: str = "python", *, init_options: dict[str, object] | None = None
) -> LanguageServerSpec:
    return LanguageServerSpec(
        language=language,
        name=f"mock-{language}",
        detect_command=[sys.executable, "--version"],
        launch_command=[sys.executable, "mock_lsp.py", language],
        install_hint="npm i -g pyright",
        root_markers=[".git"],
        init_options=init_options or {},
    )


def test_default_registry_specs_include_exact_install_hints() -> None:
    specs = default_language_server_specs()

    assert specs["python"].detect_command == ["pyright", "--version"]
    assert specs["python"].launch_command == ["pyright-langserver", "--stdio"]
    assert specs["python"].install_hint == "npm i -g pyright"
    assert (
        specs["typescript"].install_hint
        == "npm i -g typescript-language-server typescript"
    )
    assert (
        specs["javascript"].install_hint
        == "npm i -g typescript-language-server typescript"
    )


def test_manager_lazy_starts_once_and_keys_by_workspace_language_command_and_options(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        spec = _spec(init_options={"analysis": {"strict": True}})
        manager = LSPManager(tmp_path, specs=[spec], client_factory=factory)

        key = manager.key_for("python")
        first = await manager.ensure_client("python")
        second = await manager.ensure_client("python")

        assert first is second
        assert factory.languages == ["python"]
        assert factory.commands == [tuple(spec.launch_command)]
        assert factory.clients[0].initialize_calls == 1
        assert key == first.key
        assert key[0] == str(tmp_path.resolve(strict=False))
        assert key[1] == "python"
        assert key[2] == tuple(spec.launch_command)
        assert len(key[3]) == 16
        assert first.capabilities.definition_provider is True
        assert second.last_used_at >= first.started_at

    _run(scenario())


def test_manager_threads_request_timeout_to_custom_factory(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        manager = LSPManager(
            tmp_path,
            specs=[_spec()],
            client_factory=factory,
            request_timeout_seconds=1.25,
        )

        _ = await manager.ensure_client("python")

        assert manager.request_timeout_seconds == 1.25
        assert factory.request_timeouts == [1.25]

    _run(scenario())


def test_concurrent_ensure_client_starts_one_client(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory(initialize_delay=0.02)
        manager = LSPManager(tmp_path, specs=[_spec()], client_factory=factory)

        handles = await asyncio.gather(
            manager.ensure_client("python"),
            manager.ensure_client("python"),
            manager.ensure_client("python"),
        )

        assert handles[0] is handles[1] is handles[2]
        assert len(factory.clients) == 1
        assert factory.clients[0].initialize_calls == 1
        assert len(manager.handles) == 1

    _run(scenario())


def test_missing_executable_is_nonfatal_and_reports_chinese_install_hint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        spec = LanguageServerSpec(
            language="python",
            name="pyright",
            detect_command=["definitely-missing-hpd-lsp-binary", "--version"],
            launch_command=["definitely-missing-hpd-lsp-binary", "--stdio"],
            install_hint="npm i -g pyright",
            root_markers=[".git"],
        )
        manager = LSPManager(tmp_path, specs=[spec], client_factory=factory)

        with pytest.raises(ProviderUnavailable) as raised:
            _ = await manager.ensure_client("python")

        assert factory.clients == []
        assert "缺少语言服务器 pyright" in str(raised.value)
        assert "npm i -g pyright" in str(raised.value)

    _run(scenario())


def test_manager_check_health_reports_stopped_initialized_client(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        manager = LSPManager(tmp_path, specs=[_spec()], client_factory=factory)

        _ = await manager.ensure_client("python")
        assert await manager.check_health("python") is True
        factory.clients[0].running = False

        assert await manager.check_health("python") is False

    _run(scenario())


def test_detect_command_executes_full_command_before_starting_client(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        marker = tmp_path / "detect-ran.txt"
        detect_script = tmp_path / "detect_server.py"
        _ = detect_script.write_text(
            "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )
        spec = LanguageServerSpec(
            language="python",
            name="mock-detect",
            detect_command=[sys.executable, str(detect_script), str(marker)],
            launch_command=[sys.executable, "mock_lsp.py"],
            install_hint="npm i -g pyright",
            root_markers=[".git"],
        )
        manager = LSPManager(tmp_path, specs=[spec], client_factory=factory)

        _ = await manager.ensure_client("python")

        assert marker.read_text(encoding="utf-8") == "ran"
        assert len(factory.clients) == 1

    _run(scenario())


def test_detect_command_nonzero_is_nonfatal_and_reports_install_hint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        spec = LanguageServerSpec(
            language="python",
            name="pyright",
            detect_command=[sys.executable, "-c", "raise SystemExit(17)"],
            launch_command=[sys.executable, "mock_lsp.py"],
            install_hint="npm i -g pyright",
            root_markers=[".git"],
        )
        manager = LSPManager(tmp_path, specs=[spec], client_factory=factory)

        with pytest.raises(ProviderUnavailable) as raised:
            _ = await manager.ensure_client("python")

        assert factory.clients == []
        assert "检测命令退出码 17" in str(raised.value)
        assert "npm i -g pyright" in str(raised.value)

    _run(scenario())


def test_restart_and_idle_shutdown_are_explicit_hooks(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = _RecordingFactory()
        manager = LSPManager(
            tmp_path, specs=[_spec()], client_factory=factory, idle_timeout_seconds=0.5
        )

        first = await manager.ensure_client("python")
        restarted = await manager.restart("python")
        assert restarted is not first
        assert factory.clients[0].shutdown_calls == 1
        assert len(manager.handles) == 1

        restarted.last_used_at = 10.0
        closed_count = await manager.shutdown_idle(now=10.6)
        assert closed_count == 1
        assert factory.clients[1].shutdown_calls == 1
        assert manager.handles == ()

    _run(scenario())
