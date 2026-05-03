"""LangChain tool for semantic code queries through CodeIntelKernel."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from src.code_intel.core import Capability, CodeTarget, Location, Symbol, ToolResult

from ._helpers import (
    coerce_target,
    error_json,
    hover_model,
    kernel_error_json,
    language_for_path,
    language_for_target,
    location_sequence,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    symbol_sequence,
    target_path,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import CodeSemanticData
from .runtime import get_code_intel_kernel

_OPERATION_CAPABILITY = {
    "definition": Capability.DEFINITION,
    "references": Capability.REFERENCES,
    "hover": Capability.HOVER,
    "document_symbols": Capability.DOCUMENT_SYMBOLS,
}


def _limited_locations(
    locations: list[Location],
    max_results: int,
    max_files: int,
) -> tuple[list[Location], dict[str, list[Location]], bool]:
    limited_by_result = locations[: max_results + 1]
    more_available = len(limited_by_result) > max_results
    visible_locations = limited_by_result[:max_results]

    grouped: dict[str, list[Location]] = {}
    for location in visible_locations:
        if location.path not in grouped and len(grouped) >= max_files:
            more_available = True
            continue
        grouped.setdefault(location.path, []).append(location)
    visible_paths = set(grouped)
    visible_locations = [location for location in visible_locations if location.path in visible_paths]
    return visible_locations, grouped, more_available


def _limited_symbols(symbols: list[Symbol], max_results: int) -> tuple[list[Symbol], bool]:
    limited = symbols[: max_results + 1]
    return limited[:max_results], len(limited) > max_results


@code_intel_tool
async def code_semantic(
    operation: Literal["definition", "references", "hover", "document_symbols"],
    target: CodeTarget,
    max_results: int = 50,
    max_files: int = 20,
) -> str:
    """语义查询：统一执行 definition、references、hover 或 document_symbols。

    Args:
        operation: 要执行的语义操作。
        target: CodeTarget JSON，可使用 symbol_id、anchor 或 location。
        max_results: 最多返回多少个位置或 symbol。
        max_files: references 分组时最多返回多少个文件。
    """
    if max_results < 1:
        return error_json("invalid_input", "max_results 必须大于 0。", "请提供正整数结果上限。")
    if max_files < 1:
        return error_json("invalid_input", "max_files 必须大于 0。", "请提供正整数文件上限。")
    capability = _OPERATION_CAPABILITY.get(operation)
    if capability is None:
        return error_json("invalid_input", "不支持的 operation。", "请使用 definition、references、hover 或 document_symbols。")

    try:
        target_model = coerce_target(target)
    except (ValidationError, ValueError) as error:
        return validation_error_json(error)

    kernel = get_code_intel_kernel()
    if operation == "document_symbols":
        path = target_path(target_model)
        if target_model.symbol_id is not None or path is None:
            resolved = safe_cast_tool_result(await kernel.resolve_target(target_model))
            if not resolved.ok:
                return kernel_error_json(resolved)
            resolved_locations = location_sequence([resolved.data])
            path = resolved_locations[0].path
        result = safe_cast_tool_result(await kernel.call(capability, language_for_path(path), path=path))
    else:
        result = safe_cast_tool_result(
            await kernel.call(capability, language_for_target(target_model), target=target_model)
        )
    if not result.ok:
        return kernel_error_json(result)

    try:
        if operation == "hover":
            data = CodeSemanticData(operation=operation, hover=hover_model(result.data), more_available=False)
            meta = merge_meta(result.meta)
        elif operation == "document_symbols":
            symbols, more_available = _limited_symbols(symbol_sequence(result.data), max_results)
            data = CodeSemanticData(
                operation=operation,
                document_symbols=symbols,
                more_available=more_available,
            )
            meta = merge_meta(result.meta, more_available=more_available)
        else:
            locations, grouped, more_available = _limited_locations(
                location_sequence(result.data),
                max_results,
                max_files,
            )
            data = CodeSemanticData(
                operation=operation,
                locations=locations,
                grouped_by_file=grouped if operation == "references" else {},
                more_available=more_available,
            )
            meta = merge_meta(result.meta, more_available=more_available)
    except (TypeError, ValueError) as error:
        return validation_error_json(error)

    return serialize_result(ToolResult[object](ok=True, data=data, meta=meta))
