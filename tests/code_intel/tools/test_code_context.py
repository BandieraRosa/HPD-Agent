"""Tests for index-backed code_context tool behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Coroutine, Generator
from pathlib import Path
from typing import Protocol, TypeVar, cast

from langchain_core.tools import BaseTool

import pytest

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Range, Symbol, SymbolKind
from src.code_intel.index import CurrentFileForStore, SymbolIndexStore
from src.code_intel.tools import code_context, set_code_intel_kernel

T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _payload(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(raw))


def _data(raw: str) -> dict[str, object]:
    payload = _payload(raw)
    assert payload["ok"] is True
    return cast(dict[str, object], payload["data"])


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col)


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


def _symbol(
    *,
    name: str,
    qualified_name: str,
    kind: SymbolKind,
    path: str,
    file_hash: str,
    symbol_range: Range,
    selection_range: Range,
    signature: str | None,
    parent_id: str | None = None,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language="python",
        path=path,
        range=symbol_range,
        selection_range=selection_range,
        parent_id=parent_id,
        signature=signature,
        doc=None,
        source="test_index",
        confidence=0.9,
        file_hash=file_hash,
        index_version="test-v1",
    )


@pytest.fixture()
def indexed_context_kernel(tmp_path: Path) -> Generator[dict[str, object], None, None]:
    async def prepare() -> dict[str, object]:
        workspace = tmp_path / "workspace"
        source_path = "src/context_sample.py"
        content = (
            "from __future__ import annotations\n"
            "import os\n"
            "\n"
            "class Service:\n"
            "    def run(self, value: str) -> str:\n"
            "        result = value.strip()\n"
            "        return helper(result)\n"
            "\n"
            "def helper(value: str) -> str:\n"
            "    return value.upper()\n"
        )
        _write(workspace / source_path, content)
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        module = _symbol(
            name="context_sample",
            qualified_name="context_sample",
            kind=SymbolKind.MODULE,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(0, 0, 9, 24),
            selection_range=_range(0, 0, 9, 24),
            signature=None,
        )
        service = _symbol(
            name="Service",
            qualified_name="Service",
            kind=SymbolKind.CLASS,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(3, 0, 6, 29),
            selection_range=_range(3, 6, 3, 13),
            signature="class Service:",
            parent_id=module.id,
        )
        run = _symbol(
            name="run",
            qualified_name="Service.run",
            kind=SymbolKind.METHOD,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(4, 4, 6, 29),
            selection_range=_range(4, 8, 4, 11),
            signature="def run(self, value: str) -> str",
            parent_id=service.id,
        )
        helper = _symbol(
            name="helper",
            qualified_name="helper",
            kind=SymbolKind.FUNCTION,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(8, 0, 9, 24),
            selection_range=_range(8, 4, 8, 10),
            signature="def helper(value: str) -> str",
            parent_id=module.id,
        )
        store = SymbolIndexStore(tmp_path / "symbols.db")
        await store.initialize()
        await store.store_symbols(_metadata(source_path, content), [module, service, run, helper])
        set_code_intel_kernel(CodeIntelKernel(symbol_index=store, workspace_root=workspace))
        return {"store": store, "run": run, "source_path": source_path}

    state = _run(prepare())
    try:
        yield state
    finally:
        _run(cast(SymbolIndexStore, state["store"]).close())
        set_code_intel_kernel(None)


def test_code_context_returns_index_backed_signature_body_parents_imports_and_nearby_symbols(
    indexed_context_kernel: dict[str, object],
) -> None:
    run = cast(Symbol, indexed_context_kernel["run"])

    data = _data(
        _run(
            _ainvoke_text(
                code_context,
                {
                    "target": {"symbol_id": run.id},
                    "include": ["signature", "body", "parents", "imports", "nearby_symbols"],
                    "max_tokens": 300,
                },
            )
        )
    )

    target_symbol = cast(dict[str, object], data["target_symbol"])
    parents = cast(list[dict[str, object]], data["parents"])
    nearby = cast(list[dict[str, object]], data["nearby_symbols"])

    assert target_symbol["qualified_name"] == "Service.run"
    assert data["signature"] == "def run(self, value: str) -> str"
    assert "return helper(result)" in cast(str, data["body"])
    assert [parent["name"] for parent in parents] == ["context_sample", "Service"]
    assert data["imports"] == ["from __future__ import annotations", "import os"]
    assert {symbol["name"] for symbol in nearby} >= {"Service", "helper"}
    assert data["truncated"] is False


def test_context_respects_max_tokens_and_sets_truncated(indexed_context_kernel: dict[str, object]) -> None:
    run = cast(Symbol, indexed_context_kernel["run"])

    raw = _run(
        _ainvoke_text(
            code_context,
            {
                "target": {"symbol_id": run.id},
                "include": ["signature", "body", "parents", "imports", "nearby_symbols"],
                "max_tokens": 8,
            },
        )
    )
    payload = _payload(raw)
    assert payload["ok"] is True
    data = cast(dict[str, object], payload["data"])
    meta = cast(dict[str, object], payload["meta"])

    assert data["truncated"] is True
    assert meta["truncated"] is True
    assert len(cast(str, data["body"] or "")) <= 32
