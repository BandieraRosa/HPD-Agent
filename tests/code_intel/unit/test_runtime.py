"""Focused tests for CodeIntelRuntime lifecycle behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

import pytest

from src.code_intel import CodeIntelKernel
from src.code_intel.config import CodeIntelConfig
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


def test_initialize_registers_static_providers_and_sets_global_kernel(
    tmp_path: Path,
) -> None:
    runtime = CodeIntelRuntime(tmp_path, config=_config(tmp_path))

    status = _run(runtime.initialize())

    assert status.initialized is True
    assert [getattr(provider, "name", "") for provider in runtime.kernel.providers] == [
        "text_search",
        "tree_sitter",
    ]
    assert get_code_intel_kernel() is runtime.kernel
    _run(runtime.close())


def test_kernel_provider_registration_is_idempotent_by_provider_name() -> None:
    class Provider:
        name = "duplicate"

    kernel = CodeIntelKernel()

    kernel.register_provider(Provider())
    kernel.register_provider(Provider())

    assert len(kernel.providers) == 1
