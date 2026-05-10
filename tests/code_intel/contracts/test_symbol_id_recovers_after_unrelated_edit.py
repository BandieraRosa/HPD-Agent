"""Contract for stale symbol_id recovery after unrelated edits."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar, cast

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Capability, CodeContext, CodeTarget, ContextPart, Range, Symbol, SymbolKind
from src.code_intel.index import (
    CurrentFileForStore,
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
    return Range(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col)


def _symbol(
    *,
    name: str,
    qualified_name: str,
    path: str,
    file_hash: str,
    symbol_range: Range,
    selection_range: Range,
    signature: str,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=SymbolKind.FUNCTION,
        language="python",
        path=path,
        range=symbol_range,
        selection_range=selection_range,
        parent_id=None,
        signature=signature,
        doc=None,
        source="test_index",
        confidence=0.9,
        file_hash=file_hash,
        index_version="test-v1",
    )


def test_symbol_id_recovers_after_unrelated_edit(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        path = "src/recovery.py"
        old_content = "def target():\n    return 1\n\ndef unrelated():\n    return 'old'\n"
        new_content = "def target():\n    return 1\n\ndef unrelated():\n    return 'new'\n"
        _write(workspace / path, old_content)
        store = SymbolIndexStore(tmp_path / "symbols.db")
        old_target = _symbol(
            name="target",
            qualified_name="target",
            path=path,
            file_hash=_hash(old_content),
            symbol_range=_range(0, 0, 1, 12),
            selection_range=_range(0, 4, 0, 10),
            signature="def target():",
        )
        old_unrelated = _symbol(
            name="unrelated",
            qualified_name="unrelated",
            path=path,
            file_hash=_hash(old_content),
            symbol_range=_range(3, 0, 4, 16),
            selection_range=_range(3, 4, 3, 13),
            signature="def unrelated():",
        )
        new_target = _symbol(
            name="target",
            qualified_name="target",
            path=path,
            file_hash=_hash(new_content),
            symbol_range=_range(0, 0, 1, 12),
            selection_range=_range(0, 4, 0, 10),
            signature="def target():",
        )
        new_unrelated = _symbol(
            name="unrelated",
            qualified_name="unrelated",
            path=path,
            file_hash=_hash(new_content),
            symbol_range=_range(3, 0, 4, 16),
            selection_range=_range(3, 4, 3, 13),
            signature="def unrelated():",
        )
        try:
            await store.initialize()
            await store.store_symbols(_metadata(path, old_content), [old_target, old_unrelated])
            _write(workspace / path, new_content)
            await store.store_symbols(_metadata(path, new_content), [new_target, new_unrelated])
            result = await CodeIntelKernel(symbol_index=store, workspace_root=workspace).call(
                Capability.CONTEXT_EXTRACT,
                "python",
                target=CodeTarget(symbol_id=old_target.id),
                include={ContextPart.SIGNATURE, ContextPart.BODY},
                max_tokens=100,
            )

            assert result.ok is True
            context = cast(CodeContext, result.data)
            assert context.target_symbol.id == new_target.id
            assert context.target_symbol.qualified_name == "target"
            assert context.body is not None
            assert "return 1" in context.body
            assert result.meta.sources_used == ["symbol_index"]
            assert SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME in result.meta.flags
        finally:
            await store.close()

    _run(scenario())
