"""Shared helpers for thin agent-facing code-intelligence tools."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TypeVar, cast

from pydantic import BaseModel

from src.code_intel.core.languages import (
    TOOL_LANGUAGE_BY_EXTENSION,
    language_for_path_or_default,
)
from src.code_intel.core import (
    CodeContext,
    CodeTarget,
    ContextPart,
    Diagnostic,
    HoverInfo,
    Location,
    Symbol,
    ToolError,
    ToolMeta,
    ToolResult,
)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_LANGUAGE = "python"
_CONTEXT_PARTS = {
    "signature": ContextPart.SIGNATURE,
    "body": ContextPart.BODY,
    "parents": ContextPart.PARENTS,
    "imports": ContextPart.IMPORTS,
    "nearby": ContextPart.NEARBY,
    "nearby_symbols": ContextPart.NEARBY,
}


def serialize_result(result: ToolResult[object]) -> str:
    """Serialize a ToolResult as Chinese-safe JSON for LangChain tool output."""
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def success_json(data: BaseModel, meta: ToolMeta | None = None) -> str:
    """Return a successful ToolResult JSON string."""
    return serialize_result(
        ToolResult[object](ok=True, data=data, meta=meta or ToolMeta())
    )


def error_json(
    code: str, message: str, hint: str | None = None, meta: ToolMeta | None = None
) -> str:
    """Return a Chinese-first ToolResult error JSON string."""
    return serialize_result(
        ToolResult[object](
            ok=False,
            error=ToolError(code=code, message=message, hint=hint),
            meta=meta or ToolMeta(),
        )
    )


def kernel_error_json(result: ToolResult[object]) -> str:
    """Serialize a kernel error result without changing its safe error text."""
    return serialize_result(result)


def merge_meta(
    meta: ToolMeta,
    *,
    truncated: bool = False,
    more_available: bool = False,
    sources_used: list[str] | None = None,
) -> ToolMeta:
    """Copy ToolMeta while preserving kernel timing and provider source metadata."""
    return meta.model_copy(
        update={
            "truncated": meta.truncated or truncated,
            "more_available": meta.more_available or more_available,
            "sources_used": (
                sources_used if sources_used is not None else meta.sources_used
            ),
        }
    )


def language_for_path(path: str | None) -> str:
    """Infer a kernel language from a workspace-relative path without touching the filesystem."""
    return language_for_path_or_default(
        path,
        default=_DEFAULT_LANGUAGE,
        languages_by_extension=TOOL_LANGUAGE_BY_EXTENSION,
    )


def target_path(target: CodeTarget) -> str | None:
    """Return the target path when a CodeTarget carries one."""
    if target.anchor is not None:
        return target.anchor.path
    if target.location is not None:
        return target.location.path
    return None


def language_for_target(target: CodeTarget) -> str:
    """Infer language from target path, falling back to Python for symbol_id-only targets."""
    return language_for_path(target_path(target))


def coerce_target(target: CodeTarget | dict[str, object]) -> CodeTarget:
    """Accept CodeTarget objects and JSON dictionaries from LangChain tool calls."""
    if isinstance(target, CodeTarget):
        return target
    return CodeTarget.model_validate(target)


def coerce_context_parts(include: Iterable[str] | None) -> set[ContextPart]:
    """Convert LLM include strings to T4 ContextPart enum values."""
    requested = list(include or ("signature", "body", "imports"))
    parts: set[ContextPart] = set()
    for item in requested:
        part = _CONTEXT_PARTS.get(item)
        if part is None:
            raise ValueError(f"不支持的 include 项：{item}")
        parts.add(part)
    return parts


def validation_error_json(error: Exception) -> str:
    """Return a stable Chinese validation error for malformed tool input."""
    _ = error
    return error_json(
        "invalid_input",
        "工具参数不符合代码智能格式要求。",
        "请检查 target、include、mode、operation 或路径参数。",
    )


def model_sequence(data: object, model_type: type[T]) -> list[T]:
    """Validate a kernel data object as a list of Pydantic models."""
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise TypeError("kernel data is not a sequence")
    return [
        item if isinstance(item, model_type) else model_type.model_validate(item)
        for item in data
    ]


def symbol_sequence(data: object) -> list[Symbol]:
    """Validate kernel data as symbols."""
    return model_sequence(data, Symbol)


def location_sequence(data: object) -> list[Location]:
    """Validate kernel data as locations."""
    return model_sequence(data, Location)


def diagnostic_sequence(data: object) -> list[Diagnostic]:
    """Validate kernel data as diagnostics."""
    return model_sequence(data, Diagnostic)


def context_model(data: object) -> CodeContext:
    """Validate kernel data as CodeContext."""
    if isinstance(data, CodeContext):
        return data
    return CodeContext.model_validate(data)


def hover_model(data: object) -> HoverInfo | None:
    """Validate kernel data as HoverInfo when present."""
    if data is None:
        return None
    if isinstance(data, HoverInfo):
        return data
    return HoverInfo.model_validate(data)


def stable_strings(values: object) -> list[str]:
    """Return deterministic strings for enum-like iterable values."""
    if isinstance(values, (str, bytes, bytearray)):
        return [str(values)]
    if not isinstance(values, Iterable):
        return []
    return sorted(str(getattr(item, "value", item)) for item in values)


def first_sources(*results: ToolResult[object]) -> list[str]:
    """Collect provider source names from kernel results without duplicates."""
    sources: list[str] = []
    for result in results:
        for source in result.meta.sources_used:
            if source not in sources:
                sources.append(source)
    return sources


def safe_cast_tool_result(result: object) -> ToolResult[object]:
    """Narrow dynamic kernel results for helpers that preserve ToolResult metadata."""
    return cast(ToolResult[object], result)
