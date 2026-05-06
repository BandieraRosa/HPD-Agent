"""Tests for index-backed CodeTarget resolution."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

from src.code_intel import CodeIntelKernel
from src.code_intel.core import (
    CodeTarget,
    Location,
    Range,
    Symbol,
    SymbolKind,
    TextAnchor,
)
from src.code_intel.index import (
    CurrentFileForStore,
    IndexBackedTargetResolver,
    SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME,
    SymbolIndexStore,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _metadata(path: str, content: str) -> CurrentFileForStore:
    return CurrentFileForStore(
        path=path,
        language="python",
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        mtime=1.0,
        size=len(content),
        grammar_version="grammar-v1",
        query_version="query-v1",
    )


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


def _symbol(
    *,
    name: str,
    qualified_name: str,
    path: str,
    file_hash: str,
    symbol_range: Range,
    selection_range: Range | None = None,
    kind: SymbolKind = SymbolKind.FUNCTION,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language="python",
        path=path,
        range=symbol_range,
        selection_range=selection_range,
        parent_id=None,
        signature=f"def {name}():",
        doc=None,
        source="test_index",
        confidence=0.9,
        file_hash=file_hash,
        index_version="test-v1",
    )


def test_resolver_prefers_symbol_id_then_anchor_then_location(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        path = "src/app.py"
        content = "def target():\n    return 'target'\n\ndef fallback():\n    return 'fallback'\n"
        _write(workspace / path, content)
        store = SymbolIndexStore(tmp_path / "symbols.db")
        file_hash = _hash(content)
        target_symbol = _symbol(
            name="target",
            qualified_name="target",
            path=path,
            file_hash=file_hash,
            symbol_range=_range(0, 0, 1, 19),
            selection_range=_range(0, 4, 0, 10),
        )
        fallback_symbol = _symbol(
            name="fallback",
            qualified_name="fallback",
            path=path,
            file_hash=file_hash,
            symbol_range=_range(3, 0, 4, 23),
            selection_range=_range(3, 4, 3, 12),
        )
        try:
            await store.initialize()
            await store.store_symbols(
                _metadata(path, content), [target_symbol, fallback_symbol]
            )
            resolver = IndexBackedTargetResolver(store, workspace)
            low_priority_location = Location(path=path, range=_range(4, 4, 4, 10))

            by_symbol = await resolver.resolve_target(
                CodeTarget(
                    symbol_id=target_symbol.id,
                    anchor=TextAnchor(path=path, symbol_name="fallback"),
                    location=low_priority_location,
                )
            )
            by_anchor = await resolver.resolve_target(
                CodeTarget(
                    anchor=TextAnchor(path=path, symbol_name="fallback"),
                    location=low_priority_location,
                )
            )
            by_location = await resolver.resolve_target(
                CodeTarget(location=low_priority_location)
            )

            assert by_symbol.source == "symbol_id"
            assert by_symbol.symbol is not None
            assert by_symbol.symbol.id == target_symbol.id
            assert by_symbol.location.range == target_symbol.selection_range
            assert by_anchor.source == "anchor"
            assert by_anchor.symbol is not None
            assert by_anchor.symbol.id == fallback_symbol.id
            assert by_location.source == "location"
            assert by_location.location == low_priority_location
        finally:
            await store.close()

    _run(scenario())


def test_stale_symbol_id_recovers_via_history_by_qualified_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = "src/app.py"
        old_content = "def stable():\n    return 1\n\ndef unrelated():\n    return 1\n"
        new_content = "def stable():\n    return 1\n\ndef unrelated():\n    return 2\n"
        store = SymbolIndexStore(tmp_path / "symbols.db")
        old_symbol = _symbol(
            name="stable",
            qualified_name="stable",
            path=path,
            file_hash=_hash(old_content),
            symbol_range=_range(0, 0, 1, 12),
        )
        new_symbol = _symbol(
            name="stable",
            qualified_name="stable",
            path=path,
            file_hash=_hash(new_content),
            symbol_range=_range(0, 0, 1, 12),
        )
        try:
            await store.initialize()
            await store.store_symbols(_metadata(path, old_content), [old_symbol])
            await store.store_symbols(_metadata(path, new_content), [new_symbol])
            resolver = IndexBackedTargetResolver(store, tmp_path)

            recovered = await resolver.resolve_target(
                CodeTarget(symbol_id=old_symbol.id)
            )

            assert recovered.symbol is not None
            assert recovered.symbol.id == new_symbol.id
            assert recovered.flags == (SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME,)
            assert recovered.source == "symbol_id_history"
        finally:
            await store.close()

    _run(scenario())


def test_stale_symbol_id_recovery_disambiguates_same_name_module_and_function(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = "src/main.py"
        old_content = "def main():\n    return 1\n"
        new_content = "def main():\n    return 2\n"
        store = SymbolIndexStore(tmp_path / "symbols.db")
        old_hash = _hash(old_content)
        new_hash = _hash(new_content)
        old_module = _symbol(
            name="main",
            qualified_name="main",
            path=path,
            file_hash=old_hash,
            symbol_range=_range(0, 0, 2, 0),
            selection_range=_range(0, 0, 2, 0),
            kind=SymbolKind.MODULE,
        )
        old_function = _symbol(
            name="main",
            qualified_name="main",
            path=path,
            file_hash=old_hash,
            symbol_range=_range(0, 0, 1, 12),
            selection_range=_range(0, 4, 0, 8),
            kind=SymbolKind.FUNCTION,
        )
        new_module = _symbol(
            name="main",
            qualified_name="main",
            path=path,
            file_hash=new_hash,
            symbol_range=_range(0, 0, 2, 0),
            selection_range=_range(0, 0, 2, 0),
            kind=SymbolKind.MODULE,
        )
        new_function = _symbol(
            name="main",
            qualified_name="main",
            path=path,
            file_hash=new_hash,
            symbol_range=_range(0, 0, 1, 12),
            selection_range=_range(0, 4, 0, 8),
            kind=SymbolKind.FUNCTION,
        )
        try:
            await store.initialize()
            await store.store_symbols(
                _metadata(path, old_content), [old_module, old_function]
            )
            await store.store_symbols(
                _metadata(path, new_content), [new_module, new_function]
            )
            resolver = IndexBackedTargetResolver(store, tmp_path / "workspace")

            recovered = await resolver.resolve_target(
                CodeTarget(symbol_id=old_function.id)
            )

            assert recovered.source == "symbol_id_history"
            assert recovered.symbol is not None
            assert recovered.symbol.kind == SymbolKind.FUNCTION
            assert recovered.symbol.id == new_function.id
            assert recovered.location.range == new_function.selection_range
        finally:
            await store.close()

    _run(scenario())


def test_unresolved_stale_symbol_id_returns_symbol_not_found_with_code_search_hint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = "src/app.py"
        old_content = "def gone():\n    return 1\n"
        new_content = "def renamed():\n    return 1\n"
        store = SymbolIndexStore(tmp_path / "symbols.db")
        old_symbol = _symbol(
            name="gone",
            qualified_name="gone",
            path=path,
            file_hash=_hash(old_content),
            symbol_range=_range(0, 0, 1, 12),
        )
        renamed_symbol = _symbol(
            name="renamed",
            qualified_name="renamed",
            path=path,
            file_hash=_hash(new_content),
            symbol_range=_range(0, 0, 1, 12),
        )
        try:
            await store.initialize()
            await store.store_symbols(_metadata(path, old_content), [old_symbol])
            await store.store_symbols(_metadata(path, new_content), [renamed_symbol])
            result = await CodeIntelKernel(
                symbol_index=store, workspace_root=tmp_path
            ).resolve_target(CodeTarget(symbol_id=old_symbol.id))

            assert result.ok is False
            assert result.error is not None
            assert result.error.code == "symbol_not_found"
            assert result.error.hint is not None
            assert "symbol 已被修改或删除" in result.error.hint
            assert "code_search" in result.error.hint
        finally:
            await store.close()

    _run(scenario())


def test_ambiguous_text_anchor_returns_symbol_not_found_tool_error(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        path = "src/app.py"
        content = "def first():\n    return value\n\ndef second():\n    return value\n"
        _write(workspace / path, content)
        store = SymbolIndexStore(tmp_path / "symbols.db")
        file_hash = _hash(content)
        first = _symbol(
            name="first",
            qualified_name="first",
            path=path,
            file_hash=file_hash,
            symbol_range=_range(0, 0, 1, 16),
        )
        second = _symbol(
            name="second",
            qualified_name="second",
            path=path,
            file_hash=file_hash,
            symbol_range=_range(3, 0, 4, 16),
        )
        try:
            await store.initialize()
            await store.store_symbols(_metadata(path, content), [first, second])
            result = await CodeIntelKernel(
                symbol_index=store, workspace_root=workspace
            ).resolve_target(
                CodeTarget(anchor=TextAnchor(path=path, needle="return value"))
            )

            assert result.ok is False
            assert result.error is not None
            assert result.error.code == "symbol_not_found"
            assert result.error.hint is not None
            assert "code_search" in result.error.hint
        finally:
            await store.close()

    _run(scenario())
