"""Tests for async-native tool invocation helper."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

import pytest

from src.tools.invocation import invoke_tool

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def test_invoke_tool_prefers_async_path() -> None:
    class Tool:
        name: str = "fake"

        def __init__(self) -> None:
            self.async_calls: int = 0
            self.sync_calls: int = 0

        async def ainvoke(self, args: dict[str, object]) -> str:
            self.async_calls += 1
            return f"async:{args['value']}"

        def invoke(self, args: dict[str, object]) -> str:
            _ = args
            self.sync_calls += 1
            raise AssertionError("sync invoke must not be called")

    tool = Tool()

    result = _run(invoke_tool(tool, {"value": "ok"}))

    assert result == "async:ok"
    assert tool.async_calls == 1
    assert tool.sync_calls == 0


def test_invoke_tool_rejects_sync_only_tool_without_calling_invoke() -> None:
    class SyncOnlyTool:
        name: str = "sync_only"

        def __init__(self) -> None:
            self.sync_calls: int = 0

        def invoke(self, args: dict[str, object]) -> str:
            _ = args
            self.sync_calls += 1
            raise AssertionError("sync fallback must not be used")

    tool = SyncOnlyTool()

    with pytest.raises(TypeError, match="工具不支持异步调用"):
        _ = _run(invoke_tool(tool, {"value": "blocked"}))

    assert tool.sync_calls == 0
