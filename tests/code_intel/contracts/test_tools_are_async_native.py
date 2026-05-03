"""Contract tests proving code_intel LangChain tools are async-native."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from langchain_core.tools import BaseTool

from src.code_intel.tools import code_context, code_outline, code_search, code_semantic, code_verify


class _AsyncNativeTool(Protocol):
    name: str
    description: str | None
    coroutine: Callable[..., Awaitable[object]] | None
    func: Callable[..., object] | None


def test_code_intel_tools_are_exactly_five_async_native_langchain_tools() -> None:
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
        assert typed_item.coroutine is not None
        assert inspect.iscoroutinefunction(typed_item.coroutine)
        assert typed_item.func is None


def test_code_intel_tool_descriptions_are_chinese_first() -> None:
    tools = [code_search, code_outline, code_context, code_semantic, code_verify]

    for item in tools:
        assert item.description is not None
        first_line = item.description.splitlines()[0]
        assert any("一" <= char <= "鿿" for char in first_line)
