"""Tests for index-backed code_semantic document_symbols behavior."""

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
from src.code_intel.core import (
    Capability,
    CodeTarget,
    ConfidenceClass,
    Location,
    ProviderHealth,
    ProviderStatus,
    Range,
    Symbol,
    SymbolKind,
)
from src.code_intel.index import CurrentFileForStore, SymbolIndexStore
from src.code_intel.tools import code_semantic, set_code_intel_kernel
from src.code_intel.tools._langchain import strip_legacy_error_prefix

T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _payload(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(strip_legacy_error_prefix(raw)))


class _ResolvingReferencesProvider:
    name: str = "resolving-references"
    confidence_class: ConfidenceClass = ConfidenceClass.HIGH

    def __init__(self) -> None:
        self.targets: list[CodeTarget] = []

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability == Capability.REFERENCES and language == "python"

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def find_references(self, target: CodeTarget) -> list[Location]:
        self.targets.append(target)
        if target.location is None:
            raise AssertionError(
                "semantic target must be resolved before provider dispatch"
            )
        return [target.location]


class _PassThroughReferencesProvider:
    name: str = "passthrough-references"
    confidence_class: ConfidenceClass = ConfidenceClass.HIGH

    def __init__(self) -> None:
        self.targets: list[CodeTarget] = []

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability == Capability.REFERENCES and language == "python"

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def find_references(self, target: CodeTarget) -> list[Location]:
        self.targets.append(target)
        return [Location(path="src/provider_native.py", range=_range(0, 0, 0, 6))]


def _data(raw: str) -> dict[str, object]:
    payload = _payload(raw)
    assert payload["ok"] is True
    return cast(dict[str, object], payload["data"])


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


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
def indexed_semantic_kernel(tmp_path: Path) -> Generator[dict[str, object], None, None]:
    async def prepare() -> dict[str, object]:
        workspace = tmp_path / "workspace"
        source_path = "src/semantic_sample.py"
        content = "class Service:\n    def run(self):\n        return 1\n\ndef helper():\n    return 2\n"
        _write(workspace / source_path, content)
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        service = _symbol(
            name="Service",
            qualified_name="Service",
            kind=SymbolKind.CLASS,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(0, 0, 2, 16),
            selection_range=_range(0, 6, 0, 13),
            signature="class Service:",
        )
        run = _symbol(
            name="run",
            qualified_name="Service.run",
            kind=SymbolKind.METHOD,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(1, 4, 2, 16),
            selection_range=_range(1, 8, 1, 11),
            signature="def run(self):",
            parent_id=service.id,
        )
        helper = _symbol(
            name="helper",
            qualified_name="helper",
            kind=SymbolKind.FUNCTION,
            path=source_path,
            file_hash=file_hash,
            symbol_range=_range(4, 0, 5, 12),
            selection_range=_range(4, 4, 4, 10),
            signature="def helper():",
        )
        store = SymbolIndexStore(tmp_path / "symbols.db")
        await store.initialize()
        await store.store_symbols(
            _metadata(source_path, content), [service, run, helper]
        )
        set_code_intel_kernel(
            CodeIntelKernel(symbol_index=store, workspace_root=workspace)
        )
        return {"store": store, "workspace": workspace, "path": source_path, "run": run}

    state = _run(prepare())
    try:
        yield state
    finally:
        _run(cast(SymbolIndexStore, state["store"]).close())
        set_code_intel_kernel(None)


def test_document_symbols_returns_symbols_from_index_path(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    source_path = cast(str, indexed_semantic_kernel["path"])

    data = _data(
        _run(
            _ainvoke_text(
                code_semantic,
                {
                    "operation": "document_symbols",
                    "target": {
                        "anchor": {"path": source_path, "symbol_name": "Service"}
                    },
                    "max_results": 10,
                },
            )
        )
    )
    symbols = cast(list[dict[str, object]], data["document_symbols"])

    assert data["operation"] == "document_symbols"
    assert [symbol["name"] for symbol in symbols] == ["Service", "run", "helper"]


def test_document_symbols_resolves_symbol_id_to_index_path(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    run = cast(Symbol, indexed_semantic_kernel["run"])

    data = _data(
        _run(
            _ainvoke_text(
                code_semantic,
                {
                    "operation": "document_symbols",
                    "target": {"symbol_id": run.id},
                    "max_results": 10,
                },
            )
        )
    )
    symbols = cast(list[dict[str, object]], data["document_symbols"])

    assert [symbol["qualified_name"] for symbol in symbols] == [
        "Service",
        "Service.run",
        "helper",
    ]


def test_document_symbols_respects_result_limit(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    source_path = cast(str, indexed_semantic_kernel["path"])
    raw = _run(
        _ainvoke_text(
            code_semantic,
            {
                "operation": "document_symbols",
                "target": {"anchor": {"path": source_path, "symbol_name": "Service"}},
                "max_results": 1,
            },
        )
    )
    payload = _payload(raw)
    assert payload["ok"] is True
    data = cast(dict[str, object], payload["data"])
    meta = cast(dict[str, object], payload["meta"])

    assert len(cast(list[object], data["document_symbols"])) == 1
    assert data["more_available"] is True
    assert meta["more_available"] is True


def test_references_preserves_provider_native_target_without_resolver() -> None:
    provider = _PassThroughReferencesProvider()
    set_code_intel_kernel(CodeIntelKernel([provider]))
    try:
        data = _data(
            _run(
                _ainvoke_text(
                    code_semantic,
                    {
                        "operation": "references",
                        "target": {"symbol_id": "provider-native-symbol"},
                    },
                )
            )
        )
    finally:
        set_code_intel_kernel(None)
    locations = cast(list[dict[str, object]], data["locations"])

    assert [location["path"] for location in locations] == ["src/provider_native.py"]
    assert len(provider.targets) == 1
    assert provider.targets[0].symbol_id == "provider-native-symbol"
    assert provider.targets[0].location is None


def test_references_resolves_symbol_id_before_provider_dispatch(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    store = cast(SymbolIndexStore, indexed_semantic_kernel["store"])
    workspace = cast(Path, indexed_semantic_kernel["workspace"])
    run = cast(Symbol, indexed_semantic_kernel["run"])
    provider = _ResolvingReferencesProvider()
    set_code_intel_kernel(
        CodeIntelKernel([provider], symbol_index=store, workspace_root=workspace)
    )

    data = _data(
        _run(
            _ainvoke_text(
                code_semantic,
                {
                    "operation": "references",
                    "target": {"symbol_id": run.id},
                    "max_results": 10,
                    "max_files": 5,
                },
            )
        )
    )
    locations = cast(list[dict[str, object]], data["locations"])

    assert data["operation"] == "references"
    assert [location["path"] for location in locations] == [run.path]
    assert len(provider.targets) == 1
    dispatched_target = provider.targets[0]
    assert dispatched_target.symbol_id is None
    assert dispatched_target.anchor is None
    assert dispatched_target.location is not None
    assert dispatched_target.location.path == run.path
    assert dispatched_target.location.range == run.selection_range


def test_references_resolution_failure_does_not_dispatch_provider(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    store = cast(SymbolIndexStore, indexed_semantic_kernel["store"])
    workspace = cast(Path, indexed_semantic_kernel["workspace"])
    provider = _ResolvingReferencesProvider()
    set_code_intel_kernel(
        CodeIntelKernel([provider], symbol_index=store, workspace_root=workspace)
    )

    raw = _run(
        _ainvoke_text(
            code_semantic,
            {
                "operation": "references",
                "target": {"symbol_id": "missing-symbol-id"},
            },
        )
    )
    payload = _payload(raw)
    error = cast(dict[str, object], payload["error"])

    assert payload["ok"] is False
    assert error["code"] == "symbol_not_found"
    assert provider.targets == []


def test_definition_without_lsp_provider_returns_provider_unavailable(
    indexed_semantic_kernel: dict[str, object],
) -> None:
    run = cast(Symbol, indexed_semantic_kernel["run"])

    raw = _run(
        _ainvoke_text(
            code_semantic, {"operation": "definition", "target": {"symbol_id": run.id}}
        )
    )
    payload = _payload(raw)
    error = cast(dict[str, object], payload["error"])

    assert payload["ok"] is False
    assert error["code"] == "provider_unavailable"
    assert "不可用" in cast(str, error["message"])
