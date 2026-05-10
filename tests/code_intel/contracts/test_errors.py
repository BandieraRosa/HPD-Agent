"""Contract tests for code_intel error mapping."""

from __future__ import annotations

import re

import pytest

from src.code_intel.core import (
    CodeIntelError,
    IndexStale,
    LSPTimeout,
    LanguageNotSupported,
    ProviderUnavailable,
    SymbolNotFound,
    ToolError,
    ToolResult,
)


BASELINE_ERRORS: list[tuple[type[CodeIntelError], str]] = [
    (LanguageNotSupported, "unsupported_language"),
    (ProviderUnavailable, "provider_unavailable"),
    (IndexStale, "index_stale"),
    (SymbolNotFound, "symbol_not_found"),
    (LSPTimeout, "lsp_timeout"),
]


def _has_chinese_text(value: str) -> bool:
    return any("一" <= char <= "鿿" for char in value)


@pytest.mark.parametrize(("error_cls", "expected_code"), BASELINE_ERRORS)
def test_every_baseline_exception_inherits_code_intel_error(error_cls: type[CodeIntelError], expected_code: str) -> None:
    error = error_cls("raw provider exception: connection refused")

    assert isinstance(error, CodeIntelError)
    assert issubclass(error_cls, CodeIntelError)
    assert error.code == expected_code


@pytest.mark.parametrize(("error_cls", "expected_code"), BASELINE_ERRORS)
def test_errors_map_to_tool_error(error_cls: type[CodeIntelError], expected_code: str) -> None:
    error = error_cls("raw provider exception: secret traceback")
    tool_error = error.to_tool_error()

    assert isinstance(tool_error, ToolError)
    assert tool_error.code == expected_code
    assert re.fullmatch(r"[a-z][a-z0-9_]*", tool_error.code)
    assert _has_chinese_text(tool_error.message)
    assert tool_error.hint is not None
    assert _has_chinese_text(tool_error.hint)
    assert "raw provider exception" not in tool_error.message
    assert "raw provider exception" not in tool_error.hint
    assert "secret traceback" not in tool_error.message
    assert "secret traceback" not in tool_error.hint


def test_error_codes_messages_and_hints_are_stable() -> None:
    mapped = {error_cls.__name__: error_cls().to_tool_error().model_dump(mode="json") for error_cls, _ in BASELINE_ERRORS}

    assert mapped == {
        "LanguageNotSupported": {
            "code": "unsupported_language",
            "message": "当前语言不受支持。",
            "hint": "请跳过本文件，或使用 read_file 直接查看源码。",
        },
        "ProviderUnavailable": {
            "code": "provider_unavailable",
            "message": "代码智能提供方暂时不可用。",
            "hint": "请稍后重试，或改用基础文件读取和搜索。",
        },
        "IndexStale": {
            "code": "index_stale",
            "message": "代码索引已过期。",
            "hint": "请等待系统重建索引后重试。",
        },
        "SymbolNotFound": {
            "code": "symbol_not_found",
            "message": "未找到指定符号。",
            "hint": "symbol 已被修改或删除，请用 code_search 重定位。",
        },
        "LSPTimeout": {
            "code": "lsp_timeout",
            "message": "语言服务器响应超时。",
            "hint": "可降级到 tree-sitter，或稍后重试。",
        },
    }


def test_tool_result_can_wrap_mapped_code_intel_error() -> None:
    result = ToolResult[str](ok=False, error=SymbolNotFound().to_tool_error())

    dumped = result.model_dump(mode="json")

    assert dumped["ok"] is False
    assert dumped["data"] is None
    assert dumped["error"] == {
        "code": "symbol_not_found",
        "message": "未找到指定符号。",
        "hint": "symbol 已被修改或删除，请用 code_search 重定位。",
    }
    assert dumped["meta"]["sources_used"] == []
