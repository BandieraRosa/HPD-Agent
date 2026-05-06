"""Tests for LSP provider health transitions around crashes and restarts."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar

import pytest
from lsprotocol import types as lsp_types

from src.code_intel import CodeIntelKernel
from src.code_intel.core import (
    Capability,
    CodeTarget,
    Location,
    ProviderStatus,
    ProviderUnavailable,
    Range,
)
from src.code_intel.providers.lsp import LSPManager, LSPProvider, LanguageServerSpec

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


def _target() -> CodeTarget:
    return CodeTarget(location=Location(path="src/mock.py", range=_range(0, 4, 0, 8)))


def _definition_location() -> Location:
    return Location(path="src/mock.py", range=_range(0, 4, 0, 8))


def _spec() -> LanguageServerSpec:
    return LanguageServerSpec(
        language="python",
        name="mock-lsp",
        detect_command=[sys.executable, "--version"],
        launch_command=[sys.executable, "mock-health-server.py"],
        install_hint="npm i -g pyright",
        root_markers=[".git"],
    )


class _FakeSemanticClient:
    def __init__(
        self,
        *,
        initialize_error: ProviderUnavailable | None = None,
        definition_error: ProviderUnavailable | None = None,
        definition_result: list[Location] | None = None,
        running: bool = True,
        capabilities: lsp_types.ServerCapabilities | None = None,
    ) -> None:
        self.initialize_error: ProviderUnavailable | None = initialize_error
        self.definition_error: ProviderUnavailable | None = definition_error
        self.definition_result: list[Location] = definition_result or [
            _definition_location()
        ]
        self.running: bool = running
        self.capabilities: lsp_types.ServerCapabilities = (
            capabilities
            if capabilities is not None
            else lsp_types.ServerCapabilities(definition_provider=True)
        )
        self.initialize_calls: int = 0
        self.shutdown_calls: int = 0
        self.close_calls: int = 0
        self.definition_calls: int = 0

    async def initialize(
        self,
        *,
        root_uri: str | None = None,
        initialization_options: object | None = None,
    ) -> lsp_types.InitializeResult:
        self.initialize_calls += 1
        _ = root_uri
        _ = initialization_options
        if self.initialize_error is not None:
            raise self.initialize_error
        return lsp_types.InitializeResult(capabilities=self.capabilities)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def is_running(self) -> bool:
        return self.running

    async def goto_definition(
        self, path: str, *, line: int, character: int
    ) -> list[Location]:
        self.definition_calls += 1
        _ = path
        _ = line
        _ = character
        if self.definition_error is not None:
            raise self.definition_error
        return list(self.definition_result)


class _SequenceFactory:
    def __init__(self, clients: list[_FakeSemanticClient]) -> None:
        self.clients: list[_FakeSemanticClient] = clients
        self.created: list[_FakeSemanticClient] = []

    def __call__(
        self,
        spec: LanguageServerSpec,
        workspace_root: Path,
        command: Sequence[str],
        *,
        request_timeout_seconds: float,
    ) -> _FakeSemanticClient:
        _ = spec
        _ = workspace_root
        _ = command
        _ = request_timeout_seconds
        client = self.clients.pop(0)
        self.created.append(client)
        return client


def test_crash_marks_unhealthy_kernel_skips_and_restart_success_restores_health(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        crashing = _FakeSemanticClient(
            definition_error=ProviderUnavailable("language server crashed")
        )
        recovered = _FakeSemanticClient(definition_result=[_definition_location()])
        factory = _SequenceFactory([crashing, recovered])
        provider = LSPProvider(
            tmp_path,
            manager=LSPManager(tmp_path, specs=[_spec()], client_factory=factory),
        )
        kernel = CodeIntelKernel([provider])

        assert await provider.supports(Capability.DEFINITION, "python") is True
        try:
            _ = await provider.goto_definition(_target())
        except ProviderUnavailable:
            pass
        else:
            raise AssertionError("crashing client should raise ProviderUnavailable")

        assert await provider.supports(Capability.DEFINITION, "python") is True
        unhealthy = await provider.health()
        assert unhealthy.status == ProviderStatus.UNAVAILABLE
        assert unhealthy.message is not None and "unhealthy" in unhealthy.message

        skipped = await kernel.call(Capability.DEFINITION, "python", target=_target())
        assert skipped.ok is False
        assert kernel.last_trace is not None
        assert kernel.last_trace.attempts[0].attempted is False
        assert kernel.last_trace.attempts[0].error_code == ProviderUnavailable.code

        restarted = await provider.restart("python")
        assert restarted.status == ProviderStatus.HEALTHY
        routed = await kernel.call(Capability.DEFINITION, "python", target=_target())
        assert routed.ok is True
        assert routed.data == [_definition_location()]

    _run(scenario())


def test_silent_stop_is_detected_by_health_before_kernel_invokes_semantics(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        silently_stopped = _FakeSemanticClient(
            definition_result=[_definition_location()]
        )
        fallback_ready = _FakeSemanticClient(definition_result=[_definition_location()])
        factory = _SequenceFactory([silently_stopped, fallback_ready])
        provider = LSPProvider(
            tmp_path,
            manager=LSPManager(tmp_path, specs=[_spec()], client_factory=factory),
        )
        kernel = CodeIntelKernel([provider])

        assert await provider.supports(Capability.DEFINITION, "python") is True
        silently_stopped.running = False

        health = await provider.health()
        assert health.status == ProviderStatus.UNAVAILABLE
        assert health.message is not None and "not running" in health.message

        skipped = await kernel.call(Capability.DEFINITION, "python", target=_target())
        assert skipped.ok is False
        assert silently_stopped.definition_calls == 0
        assert kernel.last_trace is not None
        assert kernel.last_trace.attempts[0].attempted is False
        assert kernel.last_trace.attempts[0].error_code == ProviderUnavailable.code

        restored = await provider.restart("python")
        assert restored.status == ProviderStatus.HEALTHY
        routed = await kernel.call(Capability.DEFINITION, "python", target=_target())
        assert routed.ok is True
        assert fallback_ready.definition_calls == 1

    _run(scenario())


def test_three_restart_failures_become_permanent_and_explicit_success_clears_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        crashing = _FakeSemanticClient(
            definition_error=ProviderUnavailable("language server crashed")
        )
        failures = [
            _FakeSemanticClient(
                initialize_error=ProviderUnavailable("restart failed one")
            ),
            _FakeSemanticClient(
                initialize_error=ProviderUnavailable("restart failed two")
            ),
            _FakeSemanticClient(
                initialize_error=ProviderUnavailable("restart failed three")
            ),
        ]
        recovered = _FakeSemanticClient(definition_result=[_definition_location()])
        factory = _SequenceFactory([crashing, *failures, recovered])
        provider = LSPProvider(
            tmp_path,
            manager=LSPManager(tmp_path, specs=[_spec()], client_factory=factory),
        )

        assert await provider.supports(Capability.DEFINITION, "python") is True
        try:
            _ = await provider.goto_definition(_target())
        except ProviderUnavailable:
            pass
        else:
            raise AssertionError("crashing client should raise ProviderUnavailable")

        first = await provider.restart("python")
        second = await provider.restart("python")
        third = await provider.restart("python")

        assert first.status == ProviderStatus.UNAVAILABLE
        assert second.status == ProviderStatus.UNAVAILABLE
        assert third.status == ProviderStatus.UNAVAILABLE
        assert third.message is not None and "permanently_unhealthy" in third.message

        assert await provider.supports(Capability.DEFINITION, "python") is True
        permanent = await provider.health()
        assert permanent.status == ProviderStatus.UNAVAILABLE
        assert (
            permanent.message is not None
            and "permanently_unhealthy" in permanent.message
        )

        restored = await provider.restart("python")
        assert restored.status == ProviderStatus.HEALTHY
        assert await provider.supports(Capability.DEFINITION, "python") is True
        assert await provider.goto_definition(_target()) == [_definition_location()]

    _run(scenario())


def test_initialized_server_with_no_routed_capabilities_is_unavailable(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        unusable = _FakeSemanticClient(capabilities=lsp_types.ServerCapabilities())
        factory = _SequenceFactory([unusable])
        provider = LSPProvider(
            tmp_path,
            manager=LSPManager(tmp_path, specs=[_spec()], client_factory=factory),
        )

        _ = await provider.manager.ensure_client("python")
        health = await provider.check_health("python")

        assert health.status == ProviderStatus.UNAVAILABLE
        assert health.message is not None and "未提供可用" in health.message
        with pytest.raises(ProviderUnavailable) as raised:
            _ = await provider.supports(Capability.DEFINITION, "python")
        assert "未提供可用" in str(raised.value)
        assert unusable.initialize_calls == 1

    _run(scenario())
