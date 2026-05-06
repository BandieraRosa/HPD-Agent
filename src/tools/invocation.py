"""Async-native tool invocation helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class _AsyncInvokable(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


async def invoke_tool(tool: object, args: Mapping[str, object] | None = None) -> object:
    """Invoke a LangChain tool through its async path only.

    Code-intelligence tools are async-only, so centralizing invocation prevents
    accidental sync tool execution from crossing async boundaries.
    """
    if not isinstance(tool, _AsyncInvokable) or not callable(tool.ainvoke):
        raise TypeError(f"工具不支持异步调用：{type(tool).__name__} 必须提供 ainvoke。")

    payload = dict(args or {})
    return await tool.ainvoke(payload)


__all__ = ["invoke_tool"]
