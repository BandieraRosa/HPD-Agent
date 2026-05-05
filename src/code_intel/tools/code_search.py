"""LangChain tool for code search through CodeIntelKernel."""

from __future__ import annotations

from typing import Literal


from src.code_intel.core import Capability, Location, Symbol, SymbolKind, ToolResult

from ._helpers import (
    error_json,
    first_sources,
    kernel_error_json,
    location_sequence,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    symbol_sequence,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import CodeSearchData, SearchMatch
from .runtime import get_code_intel_kernel

_SEARCH_LANGUAGE = "python"
_KIND_MAP = {
    "function": SymbolKind.FUNCTION,
    "class": SymbolKind.CLASS,
    "method": SymbolKind.METHOD,
    "interface": SymbolKind.INTERFACE,
    "any": None,
}


def _symbol_snippet(symbol: Symbol) -> str:
    parts = [symbol.signature or symbol.qualified_name or symbol.name]
    if symbol.doc:
        parts.append(symbol.doc)
    return "\n".join(parts)


def _symbol_match(symbol: Symbol) -> SearchMatch:
    return SearchMatch(
        symbol_id=symbol.id,
        name=symbol.name,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind.value,
        path=symbol.path,
        range=symbol.range,
        snippet=_symbol_snippet(symbol),
        source=symbol.source,
        confidence=symbol.confidence,
    )


def _text_match(query: str, source: str, location: Location) -> SearchMatch:
    return SearchMatch(
        symbol_id=None,
        name=query,
        qualified_name=None,
        kind="text",
        path=location.path,
        range=location.range,
        snippet=query,
        source=source,
        confidence=0.5,
    )


async def _search_symbols(
    query: str, kind: SymbolKind | None, limit: int
) -> tuple[ToolResult[object], list[SearchMatch]]:
    result = safe_cast_tool_result(
        await get_code_intel_kernel().call(
            Capability.SYMBOL_SEARCH,
            _SEARCH_LANGUAGE,
            query=query,
            kind=kind,
            limit=limit,
        )
    )
    if not result.ok:
        return result, []
    return result, [_symbol_match(symbol) for symbol in symbol_sequence(result.data)]


async def _search_text(
    query: str,
    limit: int,
    *,
    regex: bool = False,
    case_sensitive: bool = False,
) -> tuple[ToolResult[object], list[SearchMatch]]:
    kwargs: dict[str, object] = {"query": query, "limit": limit}
    if regex:
        kwargs["regex"] = True
    if case_sensitive:
        kwargs["case_sensitive"] = True

    result = safe_cast_tool_result(
        await get_code_intel_kernel().call(
            Capability.TEXT_SEARCH,
            _SEARCH_LANGUAGE,
            **kwargs,
        )
    )
    if not result.ok:
        return result, []
    source = result.meta.sources_used[0] if result.meta.sources_used else "text_search"
    return result, [
        _text_match(query, source, location)
        for location in location_sequence(result.data)
    ]


@code_intel_tool
async def code_search(
    query: str,
    mode: Literal["symbol", "text", "mixed"] = "mixed",
    kind: Literal["function", "class", "method", "interface", "any"] = "any",
    limit: int = 20,
    regex: bool = False,
    case_sensitive: bool = False,
) -> str:
    """代码搜索：按 symbol 名或文本跨文件查找，返回 ToolResult JSON 字符串。

    Args:
        query: 要查找的英文或代码片段。
        mode: symbol、text 或 mixed；默认 mixed。
        kind: symbol 搜索时的类型过滤；默认 any。
        limit: 最多返回多少条匹配。
        regex: text/mixed 文本搜索是否按正则表达式解释 query。
        case_sensitive: text/mixed 文本搜索是否区分大小写。
    """
    if not query.strip():
        return error_json(
            "invalid_input", "搜索关键词不能为空。", "请提供 symbol 名称或文本片段。"
        )
    if limit < 1:
        return error_json(
            "invalid_input", "limit 必须大于 0。", "请使用 1 到 100 之间的结果上限。"
        )
    if kind not in _KIND_MAP:
        return error_json(
            "invalid_input",
            "不支持的 kind。",
            "请使用 function、class、method、interface 或 any。",
        )

    capped_limit = min(limit, 100)
    requested_kind = _KIND_MAP[kind]
    matches: list[SearchMatch] = []
    results: list[ToolResult[object]] = []
    more_available = False

    try:
        if mode in {"symbol", "mixed"}:
            symbol_result, symbol_matches = await _search_symbols(
                query, requested_kind, capped_limit + 1
            )
            results.append(symbol_result)
            if not symbol_result.ok and mode == "symbol":
                return kernel_error_json(symbol_result)
            matches.extend(symbol_matches[:capped_limit])
            more_available = more_available or len(symbol_matches) > capped_limit

        if mode in {"text", "mixed"} and len(matches) < capped_limit:
            remaining = capped_limit - len(matches)
            text_result, text_matches = await _search_text(
                query,
                remaining + 1,
                regex=regex,
                case_sensitive=case_sensitive,
            )
            results.append(text_result)
            if not text_result.ok and mode == "text":
                return kernel_error_json(text_result)
            matches.extend(text_matches[:remaining])
            more_available = more_available or len(text_matches) > remaining
    except (TypeError, ValueError) as error:
        return validation_error_json(error)

    successful_results = [result for result in results if result.ok]
    if not successful_results:
        return (
            kernel_error_json(results[0])
            if results
            else error_json("search_failed", "代码搜索未能执行。")
        )

    meta = merge_meta(
        successful_results[0].meta,
        more_available=more_available,
        sources_used=first_sources(*successful_results),
    )
    return serialize_result(
        ToolResult[object](ok=True, data=CodeSearchData(matches=matches), meta=meta)
    )
