"""Tests for safe workspace scanning in the symbol indexer."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar

import pytest

from src.code_intel.core import IndexStale, Range, Symbol, SymbolKind
from src.code_intel.index import SymbolIndexer, SymbolIndexStore, default_symbol_index_path

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _range() -> Range:
    return Range(start_line=0, start_col=0, end_line=1, end_col=0)


class _SimpleExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, workspace_root: Path, path: str, language: str) -> Sequence[Symbol]:
        self.calls.append(path)
        file_hash = hashlib.sha256((workspace_root / path).read_bytes()).hexdigest()[:16]
        return [
            Symbol(
                name=Path(path).stem,
                qualified_name=Path(path).stem,
                kind=SymbolKind.MODULE,
                language=language,
                path=path,
                range=_range(),
                selection_range=None,
                parent_id=None,
                signature=None,
                doc=None,
                source="test_extractor",
                confidence=0.8,
                file_hash=file_hash,
                index_version="test-index-v1",
            )
        ]


class _StaleOnceExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._raised: bool = False

    def __call__(self, workspace_root: Path, path: str, language: str) -> Sequence[Symbol]:
        if not self._raised:
            self._raised = True
            self.calls.append(f"stale:{path}")
            raise IndexStale("simulated stale index")
        self.calls.append(path)
        file_hash = hashlib.sha256((workspace_root / path).read_bytes()).hexdigest()[:16]
        return [
            Symbol(
                name=Path(path).stem,
                qualified_name=Path(path).stem,
                kind=SymbolKind.MODULE,
                language=language,
                path=path,
                range=_range(),
                selection_range=None,
                parent_id=None,
                signature=None,
                doc=None,
                source="test_extractor",
                confidence=0.8,
                file_hash=file_hash,
                index_version="test-index-v1",
            )
        ]


def test_default_cache_path_hashes_workspace_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(home))

    expected_key = hashlib.sha256(str(workspace.resolve(strict=False)).encode("utf-8")).hexdigest()

    assert default_symbol_index_path(workspace) == home / ".hpagent" / "index" / expected_key / "symbols.db"


def test_workspace_scan_skips_ignored_secret_binary_oversized_and_symlink(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        cache = tmp_path / "cache" / "symbols.db"
        _write(workspace / ".gitignore", "ignored/\nsecret.txt\n")
        _write(workspace / "visible.py", "def ok():\n    return 1\n")
        _write(workspace / "ignored/hidden.py", "def hidden():\n    return 1\n")
        _write(workspace / "secret.txt", "token=abc\n")
        _write(workspace / ".env", "TOKEN=abc\n")
        _write(workspace / "large.py", "x" * 80)
        _write(workspace / "notes.txt", "plain text\n")
        _write(workspace / ".git/config.py", "def hidden():\n    return 1\n")
        _ = (workspace / "binary.py").write_bytes(b"def bad():\0hidden")
        outside = tmp_path / "outside.py"
        _write(outside, "def outside():\n    return 1\n")
        (workspace / "link.py").symlink_to(outside)
        extractor = _SimpleExtractor()
        store = SymbolIndexStore(cache)
        indexer = SymbolIndexer(
            workspace,
            store=store,
            extractor=extractor,
            grammar_version="grammar-v1",
            query_version="query-v1",
            max_file_size_bytes=32,
        )
        try:
            result = await indexer.index_workspace()
            statuses = {(status.path, status.reason) for status in result.statuses if status.status == "skipped"}
            symbols = await store.get_symbols()

            assert extractor.calls == ["visible.py"]
            assert [symbol.path for symbol in symbols] == ["visible.py"]
            assert result.indexed == 1
            assert result.skipped >= 7
            assert ("ignored", "ignored") in statuses
            assert ("secret.txt", "ignored") in statuses
            assert (".env", "secret") in statuses
            assert ("large.py", "oversized") in statuses
            assert ("binary.py", "binary") in statuses
            assert ("notes.txt", "unsupported_language") in statuses
            assert (".git", "ignored") in statuses
            assert ("link.py", "symlink") in statuses
        finally:
            await store.close()

    _run(scenario())


def test_index_file_skips_symlink_that_resolves_inside_workspace(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        cache = tmp_path / "cache" / "symbols.db"
        _write(workspace / "target.py", "def target():\n    return 1\n")
        (workspace / "link.py").symlink_to(workspace / "target.py")
        extractor = _SimpleExtractor()
        store = SymbolIndexStore(cache)
        indexer = SymbolIndexer(
            workspace,
            store=store,
            extractor=extractor,
            grammar_version="grammar-v1",
            query_version="query-v1",
        )
        try:
            decision = await indexer.index_file("link.py")

            assert decision.should_reuse is True
            assert extractor.calls == []
            assert await store.get_symbols("link.py") == []
        finally:
            await store.close()

    _run(scenario())


def test_index_file_self_heals_index_stale_by_rebuilding_single_file(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        cache = tmp_path / "cache" / "symbols.db"
        _write(workspace / "target.py", "def target():\n    return 1\n")
        initial_extractor = _SimpleExtractor()
        store = SymbolIndexStore(cache)
        first_indexer = SymbolIndexer(
            workspace,
            store=store,
            extractor=initial_extractor,
            grammar_version="grammar-v1",
            query_version="query-v1",
        )
        try:
            first_decision = await first_indexer.index_file("target.py")
            first_symbol = (await store.get_symbols("target.py"))[0]

            _write(workspace / "target.py", "def target():\n    return 2\n")
            stale_extractor = _StaleOnceExtractor()
            second_indexer = SymbolIndexer(
                workspace,
                store=store,
                extractor=stale_extractor,
                grammar_version="grammar-v1",
                query_version="query-v1",
            )
            second_decision = await second_indexer.index_file("target.py")
            second_symbol = (await store.get_symbols("target.py"))[0]
            history = await store.history_entries()

            assert first_decision.should_rebuild is True
            assert second_decision.should_rebuild is True
            assert stale_extractor.calls == ["stale:target.py", "target.py"]
            assert second_symbol.id != first_symbol.id
            assert {entry.symbol_id for entry in history if entry.path == "target.py"} == {
                first_symbol.id,
                second_symbol.id,
            }
        finally:
            await store.close()

    _run(scenario())
