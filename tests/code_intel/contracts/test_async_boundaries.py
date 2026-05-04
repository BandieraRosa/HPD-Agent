"""Contracts for async-native code_intel execution boundaries."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, cast

from langchain_core.tools import BaseTool

from src.code_intel.tools import code_context, code_outline, code_search, code_semantic, code_verify

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = PROJECT_ROOT / "src" / "code_intel" / "tools"
CODE_TOOL_FILES = sorted(TOOLS_ROOT.glob("code_*.py"))
_SYNC_ADAPTER_NAMES = {
    "asyncio.run",
    "run_until_complete",
    "anyio.run",
    "sync_to_async",
    "async_to_sync",
}


class _AsyncNativeTool(Protocol):
    name: str
    coroutine: Callable[..., Awaitable[object]] | None
    func: Callable[..., object] | None


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _calls(path: Path) -> list[str]:
    tree = ast.parse(_source(path), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            names.append(f"{func.value.id}.{func.attr}")
            names.append(func.attr)
    return names


def test_code_intel_tool_modules_do_not_use_sync_async_adapters() -> None:
    assert CODE_TOOL_FILES
    offenders: list[str] = []
    for path in CODE_TOOL_FILES:
        calls = set(_calls(path))
        blocked = sorted(calls.intersection(_SYNC_ADAPTER_NAMES))
        if blocked:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {blocked}")
        source = _source(path)
        asyncio_run = "asyncio" + ".run("
        if asyncio_run in source or ".run_until_complete(" in source:
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}: event loop sync adapter")

    assert offenders == []


def test_exported_code_intel_tools_have_only_coroutine_entrypoints() -> None:
    tools = [code_search, code_outline, code_context, code_semantic, code_verify]

    assert [tool.name for tool in tools] == [
        "code_search",
        "code_outline",
        "code_context",
        "code_semantic",
        "code_verify",
    ]
    for item in tools:
        assert isinstance(item, BaseTool)
        typed_item = cast(_AsyncNativeTool, cast(object, item))
        assert typed_item.func is None
        assert typed_item.coroutine is not None
        assert inspect.iscoroutinefunction(typed_item.coroutine)


def test_code_intel_tool_implementations_are_declared_async() -> None:
    offenders: list[str] = []
    for path in CODE_TOOL_FILES:
        tree = ast.parse(_source(path), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("code_"):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.name} is sync")

    assert offenders == []
