"""LangChain tool for file outlines through CodeIntelKernel."""

from __future__ import annotations


from src.code_intel.core import Capability, Symbol, ToolResult

from ._helpers import (
    error_json,
    kernel_error_json,
    language_for_path,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    symbol_sequence,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import CodeOutlineData
from .runtime import get_code_intel_kernel


def _symbol_depth(
    symbol: Symbol, symbols_by_id: dict[str, Symbol], seen: set[str] | None = None
) -> int:
    if symbol.parent_id is None:
        return 1
    visited = set(seen or set())
    if symbol.id in visited:
        return 1
    visited.add(symbol.id)
    parent = symbols_by_id.get(symbol.parent_id)
    if parent is None:
        return 1
    return 1 + _symbol_depth(parent, symbols_by_id, visited)


def _limit_depth(symbols: list[Symbol], max_depth: int) -> list[Symbol]:
    symbols_by_id = {symbol.id: symbol for symbol in symbols}
    return [
        symbol
        for symbol in symbols
        if _symbol_depth(symbol, symbols_by_id) <= max_depth
    ]


def _estimated_line_count(symbols: list[Symbol]) -> int:
    line_count = 0
    for symbol in symbols:
        end_position = symbol.range.end_line + (1 if symbol.range.end_col > 0 else 0)
        line_count = max(line_count, end_position)
    return line_count


async def _outline_result(path: str, language: str) -> ToolResult[object]:
    kernel = get_code_intel_kernel()
    outline = safe_cast_tool_result(
        await kernel.call(Capability.OUTLINE, language, path=path)
    )
    if outline.ok:
        return outline
    document_symbols = safe_cast_tool_result(
        await kernel.call(Capability.DOCUMENT_SYMBOLS, language, path=path)
    )
    return document_symbols if document_symbols.ok else outline


@code_intel_tool
async def code_outline(path: str, max_depth: int = 3) -> str:
    """代码大纲：返回单个文件的结构化 symbol 树，避免盲目读取整文件。

    Args:
        path: 工作区相对路径，例如 src/service.py。
        max_depth: 返回的最大嵌套层级，默认 3。
    """
    if not path.strip():
        return error_json("invalid_input", "path 不能为空。", "请提供工作区相对路径。")
    if max_depth < 1:
        return error_json(
            "invalid_input", "max_depth 必须大于 0。", "请使用 1 或更大的层级。"
        )

    language = language_for_path(path)
    try:
        result = await _outline_result(path, language)
        if not result.ok:
            return kernel_error_json(result)
        symbols = symbol_sequence(result.data)
        data = CodeOutlineData(
            path=path,
            language=language,
            symbols=_limit_depth(symbols, max_depth),
            line_count=_estimated_line_count(symbols),
        )
        return serialize_result(
            ToolResult[object](ok=True, data=data, meta=merge_meta(result.meta))
        )
    except (TypeError, ValueError) as error:
        return validation_error_json(error)
