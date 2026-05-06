"""Focused tests for CodeIntelRuntime lifecycle behavior."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar

import pytest
from lsprotocol import types as lsp_types

from src.code_intel import CodeIntelKernel
from src.code_intel.config import CodeIntelConfig
from src.code_intel.kernel import CodeIntelKernel as KernelClass
from src.code_intel.providers.lsp.manager import LSPManager
from src.code_intel.runtime import CodeIntelRuntime
from src.code_intel.tools.runtime import get_code_intel_kernel, set_code_intel_kernel

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def reset_kernel() -> None:
    set_code_intel_kernel(None)


def _config(tmp_path: Path, **overrides: object) -> CodeIntelConfig:
    data: dict[str, object] = {"cache_dir": str(tmp_path / "cache")}
    data.update(overrides)
    return CodeIntelConfig.model_validate(data)


def test_initialize_registers_static_and_lazy_lsp_providers_without_starting_lsp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fail_ensure(self: LSPManager, language: str) -> object:
        calls.append(language)
        raise AssertionError("startup must not call ensure_client by default")

    monkeypatch.setattr(LSPManager, "ensure_client", fail_ensure)

    runtime = CodeIntelRuntime(tmp_path, config=_config(tmp_path))
    status = _run(runtime.initialize())

    assert status.initialized is True
    assert calls == []
    assert [getattr(provider, "name", "") for provider in runtime.kernel.providers] == [
        "text_search",
        "tree_sitter",
        "lsp",
    ]
    assert get_code_intel_kernel() is runtime.kernel
    _run(runtime.close())


def test_auto_build_on_startup_schedules_background_work_without_awaiting_full_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_build(self: CodeIntelRuntime) -> object:
        started.set()
        await release.wait()
        return object()

    monkeypatch.setattr(CodeIntelRuntime, "build_symbol_index", slow_build)

    async def scenario() -> None:
        runtime = CodeIntelRuntime(tmp_path, config=_config(tmp_path))
        status = await runtime.initialize()
        assert status.index_build_scheduled is True
        await asyncio.wait_for(started.wait(), timeout=1)
        assert status.index_build_running is True
        release.set()
        await runtime.close()

    _run(scenario())


def test_no_background_index_when_database_exists(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime = CodeIntelRuntime(tmp_path, config=config)
    db_path = runtime.status.db_path
    assert db_path is not None
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"placeholder")

    status = _run(runtime.initialize())

    assert status.index_build_scheduled is False
    _run(runtime.close())


def test_kernel_provider_registration_is_idempotent_by_provider_name() -> None:
    class Provider:
        name = "duplicate"

    kernel = CodeIntelKernel()

    kernel.register_provider(Provider())
    kernel.register_provider(Provider())

    assert len(kernel.providers) == 1


def test_attach_symbol_index_reuses_kernel_instance(tmp_path: Path) -> None:
    from src.code_intel.index import SymbolIndexStore

    kernel = CodeIntelKernel()
    store = SymbolIndexStore(tmp_path / "symbols.db")

    returned = kernel.attach_symbol_index(store, tmp_path)

    assert returned is kernel
    assert kernel.target_resolver is not None


def test_duplicate_lsp_start_does_not_duplicate_kernel_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    runtime = CodeIntelRuntime(tmp_path, config=_config(tmp_path))

    async def scenario() -> None:
        await runtime.initialize()
        await runtime.start_lsp("python")
        await runtime.start_lsp("python")
        assert [
            getattr(provider, "name", "") for provider in runtime.kernel.providers
        ].count("lsp") == 1
        await runtime.close()

    _run(scenario())
