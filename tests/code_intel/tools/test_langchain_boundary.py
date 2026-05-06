"""Tests for the LangChain-only code_intel tool result boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine
from typing import Protocol, TypeVar, cast

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.code_intel.core import ToolMeta
from src.code_intel.tools._helpers import error_json, success_json
from src.code_intel.tools._langchain import (
    _tool_result_metadata,
    code_intel_tool,
    strip_legacy_error_prefix,
)
from src.core.observability import TraceRecord, get_tracer

T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


class _BoundaryData(BaseModel):
    value: int


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _payload(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(strip_legacy_error_prefix(raw)))


def _start_trace() -> None:
    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="langchain boundary", session_id="test")


def _end_trace() -> TraceRecord:
    record = get_tracer().end_trace()
    assert record is not None
    return record


def _span_metadata(record: TraceRecord, name: str) -> dict[str, object]:
    for span in record.spans:
        if span.name == name:
            return span.metadata
    raise AssertionError(f"span not found: {name}")


def test_langchain_success_result_remains_exact_json_without_error_prefix() -> None:
    expected = success_json(_BoundaryData(value=1), ToolMeta(elapsed_ms=7))

    async def boundary_success() -> str:
        """Return a successful code_intel ToolResult."""
        return expected

    raw = _run(_ainvoke_text(code_intel_tool(boundary_success), {}))

    assert raw == expected
    assert not raw.startswith("[Error]")
    assert _payload(raw)["ok"] is True


def test_langchain_error_result_gets_legacy_prefix_and_traces_unprefixed_json() -> None:
    expected = error_json("boundary_error", "边界错误。", meta=ToolMeta(elapsed_ms=9))

    async def boundary_error(path: str) -> str:
        """Return a failed code_intel ToolResult."""
        _ = path
        return expected

    _start_trace()
    raw = _run(_ainvoke_text(code_intel_tool(boundary_error), {"path": "src/app.py"}))
    record = _end_trace()

    assert raw.startswith("[Error]")
    assert strip_legacy_error_prefix(raw) == expected
    payload = _payload(raw)
    assert payload["ok"] is False

    metadata = _span_metadata(record, "code_intel.boundary_error")
    error = cast(dict[str, object], metadata["error"])
    assert error["code"] == "boundary_error"


def test_tool_result_metadata_accepts_prefixed_error_json() -> None:
    prefixed = "[Error] " + error_json(
        "prefixed_error", "已加前缀。", meta=ToolMeta(truncated=True)
    )

    metadata = _tool_result_metadata(prefixed)

    assert metadata["truncated"] is True
    error = cast(dict[str, object], metadata["error"])
    assert error["code"] == "prefixed_error"
