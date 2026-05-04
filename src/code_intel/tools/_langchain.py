"""Typed LangChain tool decorator for code-intelligence tools."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import wraps
from typing import cast

import langchain_core.tools as langchain_tools
from langchain_core.tools import BaseTool

from src.code_intel.core import CodeTarget
from src.code_intel.tracing import trace_span

_AsyncToolFunction = Callable[..., Awaitable[str]]
_AsyncToolDecorator = Callable[[_AsyncToolFunction], BaseTool]
_KNOWN_LIST_RESULT_FIELDS = (
    "matches",
    "symbols",
    "locations",
    "document_symbols",
    "new_diagnostics",
    "resolved_diagnostics",
    "unchanged_diagnostics",
)


def code_intel_tool(func: _AsyncToolFunction) -> BaseTool:
    """Decorate an async code_intel tool and trace it with safe metadata."""

    @wraps(func)
    async def traced_tool(*args: object, **kwargs: object) -> str:
        with trace_span(f"code_intel.{func.__name__}", _tool_input_metadata(kwargs)) as span:
            result = await func(*args, **kwargs)
            span.add_metadata(_tool_result_metadata(result))
            return result

    return cast(_AsyncToolDecorator, langchain_tools.tool)(traced_tool)


def _tool_input_metadata(kwargs: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    path = kwargs.get("path")
    if isinstance(path, str):
        metadata["path"] = path
    paths = kwargs.get("paths")
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes, bytearray)):
        metadata["paths"] = list(paths)
    target = kwargs.get("target")
    target_path = _target_path(target)
    if target_path is not None:
        metadata["path"] = target_path
    return metadata


def _target_path(target: object) -> str | None:
    if isinstance(target, CodeTarget):
        if target.anchor is not None:
            return target.anchor.path
        if target.location is not None:
            return target.location.path
        return None
    if isinstance(target, Mapping):
        target_mapping = cast(Mapping[object, object], target)
        raw_anchor = target_mapping.get("anchor")
        if isinstance(raw_anchor, Mapping):
            anchor_mapping = cast(Mapping[object, object], raw_anchor)
            path = anchor_mapping.get("path")
            if isinstance(path, str):
                return path
        raw_location = target_mapping.get("location")
        if isinstance(raw_location, Mapping):
            location_mapping = cast(Mapping[object, object], raw_location)
            path = location_mapping.get("path")
            if isinstance(path, str):
                return path
    return None


def _tool_result_metadata(result: str) -> dict[str, object]:
    try:
        payload_obj = cast(object, json.loads(result))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload_obj, dict):
        return {}
    payload = cast(dict[str, object], payload_obj)

    metadata: dict[str, object] = {}
    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta_mapping = cast(dict[str, object], meta)
        elapsed_ms = meta_mapping.get("elapsed_ms")
        if elapsed_ms is not None:
            metadata["elapsed_ms"] = elapsed_ms
        truncated = meta_mapping.get("truncated")
        if truncated is not None:
            metadata["truncated"] = truncated
    error = payload.get("error")
    if isinstance(error, dict):
        error_mapping = cast(dict[str, object], error)
        metadata["error"] = {"code": error_mapping.get("code"), "message": error_mapping.get("message")}
    count = _known_result_count(payload.get("data"))
    if count is not None:
        metadata["result_count"] = count
    return metadata


def _known_result_count(data: object) -> int | None:
    if isinstance(data, list):
        return len(cast(list[object], data))
    if not isinstance(data, dict):
        return None
    data_mapping = cast(dict[str, object], data)
    for field in _KNOWN_LIST_RESULT_FIELDS:
        value = data_mapping.get(field)
        if isinstance(value, list):
            return len(cast(list[object], value))
    return None


__all__ = ["code_intel_tool"]
