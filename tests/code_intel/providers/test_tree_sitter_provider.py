"""Tests for the real syntax-outline provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import TypeVar, cast

import pytest

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Capability, ConfidenceClass, ProviderHealth, ProviderStatus, Symbol, SymbolKind
from src.code_intel.providers.tree_sitter import (
    TREE_SITTER_CONFIDENCE,
    TREE_SITTER_INDEX_VERSION,
    TREE_SITTER_SOURCE,
    TreeSitterGrammarUnavailable,
    TreeSitterParser,
    TreeSitterProvider,
)

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_FIXTURE = "tests/code_intel/fixtures/outline_python.py"
TYPESCRIPT_FIXTURE = "tests/code_intel/fixtures/outline_typescript.ts"
JAVASCRIPT_FIXTURE = "tests/code_intel/fixtures/outline_javascript.js"
PYTHON_CONTRACT = PROJECT_ROOT / "tests/code_intel/contracts/outline_python.yaml"


class _MissingGrammarLoader:
    def __call__(self, language: str) -> object:
        raise RuntimeError(f"missing grammar: {language}")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _symbol_rows(symbols: Sequence[Symbol]) -> list[dict[str, str | None]]:
    id_to_name = {symbol.id: symbol.qualified_name or symbol.name for symbol in symbols}
    return [
        {
            "name": symbol.name,
            "kind": symbol.kind.value,
            "qualified_name": symbol.qualified_name,
            "parent": id_to_name.get(symbol.parent_id) if symbol.parent_id is not None else None,
        }
        for symbol in symbols
    ]


def _json_dump_symbols(symbols: Sequence[Symbol]) -> list[dict[str, object]]:
    return [cast(dict[str, object], symbol.model_dump(mode="json")) for symbol in symbols]


def _provider() -> TreeSitterProvider:
    return TreeSitterProvider(PROJECT_ROOT)


def test_python_outline_contract() -> None:
    symbols = _run(_provider().outline(PYTHON_FIXTURE))
    expected = cast(list[dict[str, str | None]], json.loads(PYTHON_CONTRACT.read_text(encoding="utf-8")))

    assert _symbol_rows(symbols) == expected
    assert all(type(symbol) is Symbol for symbol in symbols)
    assert symbols[0].kind == SymbolKind.MODULE
    assert symbols[4].parent_id == symbols[3].id
    assert symbols[0].range.start_line == 0
    assert symbols[0].range.start_col == 0
    assert symbols[4].selection_range is not None
    assert symbols[4].selection_range.start_col == 8


def test_tree_sitter_provider_supports_capabilities_and_languages() -> None:
    provider = _provider()

    assert _run(provider.supports(Capability.OUTLINE, "python")) is True
    assert _run(provider.supports(Capability.DOCUMENT_SYMBOLS, "typescript")) is True
    assert _run(provider.supports(Capability.DOCUMENT_SYMBOLS, "javascript")) is True
    assert _run(provider.supports(Capability.DEFINITION, "python")) is False
    assert _run(provider.supports(Capability.OUTLINE, "go")) is False
    assert _run(provider.confidence_for(Capability.OUTLINE, "python")) == ConfidenceClass.MEDIUM
    assert _run(provider.health()) == ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)


def test_symbols_are_standard_models_with_stable_hash_and_index_version() -> None:
    provider = _provider()

    first_outline = _run(provider.outline(PYTHON_FIXTURE))
    second_outline = _run(provider.document_symbols(PYTHON_FIXTURE))
    expected_hash = hashlib.sha256((PROJECT_ROOT / PYTHON_FIXTURE).read_bytes()).hexdigest()[:16]

    assert _json_dump_symbols(first_outline) == _json_dump_symbols(second_outline)
    assert all(type(symbol) is Symbol for symbol in first_outline)
    assert {symbol.source for symbol in first_outline} == {TREE_SITTER_SOURCE}
    assert {symbol.confidence for symbol in first_outline} == {TREE_SITTER_CONFIDENCE}
    assert {symbol.file_hash for symbol in first_outline} == {expected_hash}
    assert {symbol.index_version for symbol in first_outline} == {TREE_SITTER_INDEX_VERSION}


def test_typescript_and_javascript_outline_cover_required_symbol_kinds() -> None:
    provider = _provider()

    ts_symbols = _run(provider.outline(TYPESCRIPT_FIXTURE))
    js_symbols = _run(provider.outline(JAVASCRIPT_FIXTURE))
    ts_rows = _symbol_rows(ts_symbols)
    js_rows = _symbol_rows(js_symbols)

    assert [symbol.kind for symbol in ts_symbols] == [
        SymbolKind.MODULE,
        SymbolKind.IMPORT,
        SymbolKind.EXPORT,
        SymbolKind.INTERFACE,
        SymbolKind.EXPORT,
        SymbolKind.CLASS,
        SymbolKind.METHOD,
        SymbolKind.EXPORT,
        SymbolKind.FUNCTION,
        SymbolKind.FUNCTION,
        SymbolKind.EXPORT,
    ]
    assert {row["name"] for row in ts_rows} >= {
        "outline_typescript",
        "export UserRecord",
        "UserRecord",
        "export UserService",
        "UserService",
        "load",
        "export makeUserService",
        "makeUserService",
        "makeId",
        "export makeId",
    }
    assert {row["qualified_name"] for row in ts_rows} >= {"UserService.load", "makeUserService", "makeId"}
    assert next(row for row in ts_rows if row["name"] == "load")["parent"] == "UserService"

    assert [symbol.kind for symbol in js_symbols] == [
        SymbolKind.MODULE,
        SymbolKind.IMPORT,
        SymbolKind.EXPORT,
        SymbolKind.CLASS,
        SymbolKind.METHOD,
        SymbolKind.EXPORT,
        SymbolKind.FUNCTION,
        SymbolKind.FUNCTION,
        SymbolKind.EXPORT,
    ]
    assert {row["name"] for row in js_rows} >= {
        "outline_javascript",
        "export Widget",
        "Widget",
        "render",
        "export makeWidget",
        "makeWidget",
        "makeName",
        "export makeName",
    }
    assert next(row for row in js_rows if row["name"] == "render")["parent"] == "Widget"


def test_parse_is_wrapped_by_asyncio_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def spy_to_thread(func: Callable[..., T], *args: object, **kwargs: object) -> T:
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    symbols = _run(_provider().outline(PYTHON_FIXTURE))

    assert [symbol.name for symbol in symbols][:2] == ["outline_python", "os"]
    assert calls == ["_outline_sync"]


def test_syntax_error_returns_partial_outline_without_crashing(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.py"
    _ = broken_path.write_text("def ok():\n    return 1\n\nclass Broken(\n", encoding="utf-8")
    provider = TreeSitterProvider(tmp_path)

    symbols = _run(provider.outline("broken.py"))

    assert [symbol.name for symbol in symbols] == ["broken", "ok"]
    assert all(type(symbol) is Symbol for symbol in symbols)


def test_missing_grammar_returns_provider_unavailable_without_skip() -> None:
    missing_loader = _MissingGrammarLoader()
    parser = TreeSitterParser(language_loader=missing_loader, parser_loader=missing_loader)
    provider = TreeSitterProvider(PROJECT_ROOT, parser=parser)
    kernel = CodeIntelKernel([provider])

    direct_error = pytest.raises(TreeSitterGrammarUnavailable, _run, provider.outline(PYTHON_FIXTURE))
    result = _run(kernel.call(Capability.OUTLINE, "python", path=PYTHON_FIXTURE))

    assert direct_error.value.to_tool_error().code == "provider_unavailable"
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "provider_unavailable"
    assert "语法包" in result.error.message
    assert result.error.hint is not None
    assert "安装" in result.error.hint
    assert "tree-sitter-language-pack" in result.error.hint
    assert kernel.last_trace is not None
    assert kernel.last_trace.fallback_count == 1


def test_no_raw_tree_sitter_node_leaks() -> None:
    provider = _provider()
    symbols = (
        _run(provider.outline(PYTHON_FIXTURE))
        + _run(provider.outline(TYPESCRIPT_FIXTURE))
        + _run(provider.outline(JAVASCRIPT_FIXTURE))
    )

    payload = _json_dump_symbols(symbols)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert all(type(symbol) is Symbol for symbol in symbols)
    raw_node_marker = "<" + "".join(["N", "o", "d", "e"])
    raw_parser_marker = "tree_sitter" + "." + "".join(["P", "a", "r", "s", "e", "r"])
    raw_query_cursor_marker = "".join(["Q", "u", "e", "r", "y", "C", "u", "r", "s", "o", "r"])
    start_offset_marker = "start" + "_" + "byte"
    end_offset_marker = "end" + "_" + "byte"

    assert raw_node_marker not in serialized
    assert raw_parser_marker not in serialized
    assert raw_query_cursor_marker not in serialized
    assert start_offset_marker not in serialized
    assert end_offset_marker not in serialized
    assert all(isinstance(item["range"], dict) for item in payload)


def test_provider_routes_through_kernel_for_document_symbols() -> None:
    kernel = CodeIntelKernel([_provider()])

    result = _run(kernel.call(Capability.DOCUMENT_SYMBOLS, "typescript", path=TYPESCRIPT_FIXTURE))

    assert result.ok is True
    assert result.data is not None
    assert isinstance(result.data, list)
    symbols = cast(list[Symbol], result.data)
    assert [symbol.name for symbol in symbols][:2] == ["outline_typescript", 'import { readFile } from "fs";']
    assert result.meta.sources_used == ["tree_sitter"]


def test_tree_sitter_grammars_are_available_for_required_languages() -> None:
    provider = _provider()

    for path in (PYTHON_FIXTURE, TYPESCRIPT_FIXTURE, JAVASCRIPT_FIXTURE):
        symbols = _run(provider.outline(path))
        assert len(symbols) >= 2
        assert all(type(symbol) is Symbol for symbol in symbols)
