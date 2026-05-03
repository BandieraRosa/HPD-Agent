"""Tests for the async symbol index store."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

import aiosqlite
import pytest

from src.code_intel.core import Range, Symbol, SymbolKind
from src.code_intel.index import CurrentFileForStore, SymbolIndexStore

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _range(line: int = 0) -> Range:
    return Range(start_line=line, start_col=0, end_line=line, end_col=10)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _metadata(path: str, content: str = "content") -> CurrentFileForStore:
    return CurrentFileForStore(
        path=path,
        language="python",
        sha256=_hash(content),
        mtime=1.0,
        size=len(content),
        grammar_version="grammar-v1",
        query_version="query-v1",
    )


def _symbol(
    *,
    path: str = "src/app.py",
    name: str = "Alpha",
    qualified_name: str | None = "Alpha",
    file_hash: str = "abc123abc123abcd",
    kind: SymbolKind = SymbolKind.FUNCTION,
    parent_id: str | None = None,
    selection_range: Range | None = None,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language="python",
        path=path,
        range=_range(),
        selection_range=selection_range,
        parent_id=parent_id,
        signature=f"def {name}(): ...",
        doc=None,
        source="test_extractor",
        confidence=0.9,
        file_hash=file_hash,
        index_version="test-index-v1",
    )


def test_schema_wal_and_foreign_keys_initialized(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db")
        try:
            await store.initialize()

            pragmas = await store.pragma_values()
            tables = await store.schema_tables()
            file_columns = await store.table_columns("files")
            symbol_columns = await store.table_columns("symbols")
            history_columns = await store.table_columns("symbol_id_history")

            assert pragmas["journal_mode"] == "wal"
            assert pragmas["foreign_keys"] == 1
            assert pragmas["synchronous"] == 1
            assert {"files", "symbols", "symbols_fts", "symbol_id_history"}.issubset(tables)
            assert file_columns == [
                "path",
                "language",
                "sha256",
                "mtime",
                "size",
                "indexed_at",
                "grammar_version",
                "query_version",
                "schema_version",
            ]
            assert "sel_start_line" in symbol_columns
            assert "sel_start_col" in symbol_columns
            assert "sel_end_line" in symbol_columns
            assert "sel_end_col" in symbol_columns
            assert "file_sha256" in symbol_columns
            assert history_columns == [
                "symbol_id",
                "language",
                "path",
                "qualified_name",
                "file_hash",
                "created_at",
                "last_seen_at",
            ]
            assert store.fts_available is True
        finally:
            await store.close()

    _run(scenario())


def test_store_round_trips_symbols_and_history_mapping(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db")
        try:
            await store.initialize()
            parent = _symbol(name="Service", qualified_name="Service", kind=SymbolKind.CLASS, selection_range=_range())
            child = _symbol(
                name="run",
                qualified_name="Service.run",
                kind=SymbolKind.METHOD,
                parent_id=parent.id,
                selection_range=None,
            )

            await store.store_symbols(_metadata("src/app.py"), [parent, child])

            symbols = await store.get_symbols("src/app.py")
            history = await store.history_lookup(child.id)

            assert [symbol.id for symbol in symbols] == [parent.id, child.id]
            assert symbols[1].selection_range is None
            assert symbols[1].parent_id == parent.id
            assert history is not None
            assert history.symbol_id == child.id
            assert history.language == "python"
            assert history.path == "src/app.py"
            assert history.qualified_name == "Service.run"
            assert history.file_hash == child.file_hash
        finally:
            await store.close()

    _run(scenario())


def test_fts_enabled_search_matches_prefix_and_keeps_queries_safe(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db", max_search_results=2)
        try:
            await store.initialize()
            symbols = [
                _symbol(name="AlphaOne", qualified_name="AlphaOne", file_hash="hash000000000011"),
                _symbol(name="AlphaTwo", qualified_name="AlphaTwo", file_hash="hash000000000012"),
                _symbol(name="BetaOne", qualified_name="BetaOne", file_hash="hash000000000013"),
            ]
            await store.store_symbols(_metadata("src/app.py"), symbols)

            prefix_results = await store.search_symbols("Alpha", limit=50)
            quote_results = await store.search_symbols('Alpha "One"', limit=50)
            punctuation_results = await store.search_symbols("Alpha.One", limit=50)
            empty_results = await store.search_symbols("", limit=50)
            punctuation_only_results = await store.search_symbols("()", limit=50)

            assert store.fts_available is True
            assert [symbol.name for symbol in prefix_results] == ["AlphaOne", "AlphaTwo"]
            assert quote_results == []
            assert punctuation_results == []
            assert empty_results == []
            assert punctuation_only_results == []
        finally:
            await store.close()

    _run(scenario())


def test_fts_creation_failure_degrades_to_bounded_like_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        original_executescript = aiosqlite.Connection.executescript

        async def fail_fts_creation(connection: aiosqlite.Connection, sql_script: str) -> object:
            if "CREATE VIRTUAL TABLE" in sql_script:
                raise RuntimeError("simulated fts creation failure")
            return await original_executescript(connection, sql_script)

        monkeypatch.setattr(aiosqlite.Connection, "executescript", fail_fts_creation)
        store = SymbolIndexStore(tmp_path / "symbols.db", max_search_results=2)
        try:
            await store.initialize()
            symbols = [
                _symbol(name="AlphaOne", qualified_name="AlphaOne", file_hash="hash000000000031"),
                _symbol(name="AlphaTwo", qualified_name="AlphaTwo", file_hash="hash000000000032"),
                _symbol(name="AlphaThree", qualified_name="AlphaThree", file_hash="hash000000000033"),
            ]
            await store.store_symbols(_metadata("src/app.py"), symbols)

            results = await store.search_symbols("Alpha", limit=50)

            assert store.fts_available is False
            assert store.fts_error is not None
            assert "simulated fts creation failure" in store.fts_error
            assert len(results) == 2
            assert all(symbol.name.startswith("Alpha") for symbol in results)
        finally:
            await store.close()

    _run(scenario())


def test_fts_search_failure_degrades_to_bounded_like_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db", max_search_results=2)
        try:
            await store.initialize()
            symbols = [
                _symbol(name="AlphaOne", qualified_name="AlphaOne", file_hash="hash000000000021"),
                _symbol(name="AlphaTwo", qualified_name="AlphaTwo", file_hash="hash000000000022"),
                _symbol(name="AlphaThree", qualified_name="AlphaThree", file_hash="hash000000000023"),
            ]
            await store.store_symbols(_metadata("src/app.py"), symbols)
            assert store.fts_available is True

            async def fail_fts_search(*_args: object, **_kwargs: object) -> list[Symbol]:
                raise RuntimeError("simulated fts failure")

            monkeypatch.setattr(store, "_search_symbols_fts", fail_fts_search)

            results = await store.search_symbols("Alpha", limit=50)

            assert store.fts_available is False
            assert store.fts_error is not None
            assert "simulated fts failure" in store.fts_error
            assert len(results) == 2
            assert all(symbol.name.startswith("Alpha") for symbol in results)
        finally:
            await store.close()

    _run(scenario())


def test_fts_disabled_degrades_to_bounded_like_fallback(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db", enable_fts=False, max_search_results=2)
        try:
            await store.initialize()
            symbols = [
                _symbol(name="AlphaOne", qualified_name="AlphaOne", file_hash="hash000000000001"),
                _symbol(name="AlphaTwo", qualified_name="AlphaTwo", file_hash="hash000000000002"),
                _symbol(name="AlphaThree", qualified_name="AlphaThree", file_hash="hash000000000003"),
            ]
            await store.store_symbols(_metadata("src/app.py"), symbols)

            results = await store.search_symbols("Alpha", limit=50)

            assert store.fts_available is False
            assert store.fts_error == "FTS disabled for this store"
            assert len(results) == 2
            assert all(symbol.name.startswith("Alpha") for symbol in results)
        finally:
            await store.close()

    _run(scenario())


def test_delete_file_cascades_symbols_but_preserves_history(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SymbolIndexStore(tmp_path / "symbols.db")
        try:
            await store.initialize()
            symbol = _symbol(name="Gone", qualified_name="Gone")
            await store.store_symbols(_metadata("src/app.py"), [symbol])

            await store.delete_file("src/app.py")

            assert await store.get_file("src/app.py") is None
            assert await store.get_symbols("src/app.py") == []
            assert await store.history_lookup(symbol.id) is not None
        finally:
            await store.close()

    _run(scenario())
