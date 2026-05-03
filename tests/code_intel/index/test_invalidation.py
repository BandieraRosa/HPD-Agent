"""Tests for deterministic symbol index invalidation."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar

from src.code_intel.core import Range, Symbol, SymbolKind
from src.code_intel.index import (
    CurrentFileMetadata,
    IndexedFileMetadata,
    InvalidationAction,
    InvalidationReason,
    SymbolIndexer,
    SymbolIndexStore,
    decide_file_invalidation,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _full_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _range() -> Range:
    return Range(start_line=0, start_col=0, end_line=1, end_col=0)


def _previous(
    *,
    path: str = "src/app.py",
    language: str = "python",
    sha256: str = "a" * 64,
    mtime: float = 1.0,
    size: int = 10,
    indexed_at: float = 1.0,
    grammar_version: str = "grammar-v1",
    query_version: str = "query-v1",
    schema_version: str = "schema-v1",
) -> IndexedFileMetadata:
    return IndexedFileMetadata(
        path=path,
        language=language,
        sha256=sha256,
        mtime=mtime,
        size=size,
        indexed_at=indexed_at,
        grammar_version=grammar_version,
        query_version=query_version,
        schema_version=schema_version,
    )


def _current(
    *,
    path: str = "src/app.py",
    language: str = "python",
    sha256: str = "a" * 64,
    mtime: float = 2.0,
    size: int = 10,
    grammar_version: str = "grammar-v1",
    query_version: str = "query-v1",
    schema_version: str = "schema-v1",
    exists: bool = True,
) -> CurrentFileMetadata:
    return CurrentFileMetadata(
        path=path,
        language=language,
        sha256=sha256,
        mtime=mtime,
        size=size,
        grammar_version=grammar_version,
        query_version=query_version,
        schema_version=schema_version,
        exists=exists,
    )


class _RecordingExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, workspace_root: Path, path: str, language: str) -> Sequence[Symbol]:
        self.calls.append(path)
        absolute_path = workspace_root / path
        file_hash = _full_hash(absolute_path)[:16]
        stem = Path(path).stem
        return [
            Symbol(
                name="target",
                qualified_name=f"{stem}.target",
                kind=SymbolKind.FUNCTION,
                language=language,
                path=path,
                range=_range(),
                selection_range=_range(),
                parent_id=None,
                signature="def target(): ...",
                doc=None,
                source="test_extractor",
                confidence=0.9,
                file_hash=file_hash,
                index_version="test-index-v1",
            )
        ]


def test_hash_and_version_invalidation_decisions_do_not_trust_mtime_only() -> None:
    assert decide_file_invalidation(_previous(), _current()).action == InvalidationAction.REUSE
    assert decide_file_invalidation(_previous(), _current(sha256="b" * 64)).reason == (
        InvalidationReason.CONTENT_HASH_CHANGED
    )
    assert decide_file_invalidation(_previous(), _current(grammar_version="grammar-v2")).action == (
        InvalidationAction.REBUILD_ALL
    )
    assert decide_file_invalidation(_previous(), _current(query_version="query-v2")).reason == (
        InvalidationReason.QUERY_VERSION_CHANGED
    )
    assert decide_file_invalidation(_previous(), _current(schema_version="schema-v2")).reason == (
        InvalidationReason.SCHEMA_VERSION_CHANGED
    )
    assert decide_file_invalidation(_previous(), None).action == InvalidationAction.DELETE


def test_content_hash_change_rebuilds_single_file(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        cache = tmp_path / "cache" / "symbols.db"
        _write(workspace / "a.py", "def target():\n    return 1\n")
        _write(workspace / "b.py", "def target():\n    return 2\n")
        extractor = _RecordingExtractor()
        store = SymbolIndexStore(cache)
        indexer = SymbolIndexer(
            workspace,
            store=store,
            extractor=extractor,
            grammar_version="grammar-v1",
            query_version="query-v1",
        )
        try:
            first = await indexer.index_workspace()
            first_a = (await store.get_symbols("a.py"))[0]
            first_b = (await store.get_symbols("b.py"))[0]

            _write(workspace / "a.py", "def target():\n    return 10\n")
            second = await indexer.index_workspace()
            second_a = (await store.get_symbols("a.py"))[0]
            second_b = (await store.get_symbols("b.py"))[0]
            history = await store.history_entries()
            statuses = {status.path: status for status in second.statuses if status.path in {"a.py", "b.py"}}

            assert first.indexed == 2
            assert extractor.calls == ["a.py", "b.py", "a.py"]
            assert second.indexed == 1
            assert second.rebuilt == 1
            assert second.reused == 1
            assert statuses["a.py"].status == "indexed"
            assert statuses["a.py"].reason == InvalidationReason.CONTENT_HASH_CHANGED.value
            assert statuses["b.py"].status == "reused"
            assert second_a.id != first_a.id
            assert second_b.id == first_b.id
            a_history = [entry for entry in history if entry.path == "a.py" and entry.qualified_name == "a.target"]
            assert {entry.symbol_id for entry in a_history} == {first_a.id, second_a.id}
            assert {entry.file_hash for entry in a_history} == {first_a.file_hash, second_a.file_hash}
        finally:
            await store.close()

    _run(scenario())


def test_deleted_file_cascades_symbols_from_indexer(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        cache = tmp_path / "cache" / "symbols.db"
        _write(workspace / "gone.py", "def target():\n    return 1\n")
        extractor = _RecordingExtractor()
        store = SymbolIndexStore(cache)
        indexer = SymbolIndexer(
            workspace,
            store=store,
            extractor=extractor,
            grammar_version="grammar-v1",
            query_version="query-v1",
        )
        try:
            _ = await indexer.index_workspace()
            old_symbol = (await store.get_symbols("gone.py"))[0]

            (workspace / "gone.py").unlink()
            result = await indexer.index_workspace()

            assert result.deleted == 1
            assert await store.get_symbols("gone.py") == []
            assert await store.history_lookup(old_symbol.id) is not None
        finally:
            await store.close()

    _run(scenario())
