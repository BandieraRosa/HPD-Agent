"""Integration coverage for code_intel observability spans."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine
from pathlib import Path
from typing import Protocol, TypeVar, cast

from langchain_core.tools import BaseTool

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Diagnostic, DiagnosticSeverity, Range, Symbol, SymbolKind
from src.code_intel.index import CurrentFileForStore, SymbolIndexStore
from src.code_intel.providers.fake import PYTHON_FAKE_PATH, create_fake_providers
from src.code_intel.providers.lsp.client import LSPClient
from src.code_intel.tools import code_outline, set_code_intel_kernel
from src.code_intel.verifier import compute_delta
from src.core.observability import TraceRecord, get_tracer

T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.notifications: list[str] = []

    async def add_notification_handler(self, method: str, handler: object) -> None:
        _ = (method, handler)

    async def start(self) -> None:
        return None

    async def request(self, method: str, params: object | None = None, *, timeout: float | None = None) -> object | None:
        _ = (params, timeout)
        self.requests.append(method)
        if method == "textDocument/documentSymbol":
            return []
        return None

    async def notify(self, method: str, params: object | None = None) -> None:
        _ = params
        self.notifications.append(method)

    async def is_running(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _start_trace() -> None:
    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="code_intel tracing test", session_id="test")


def _end_trace() -> TraceRecord:
    record = get_tracer().end_trace()
    assert record is not None
    return record


def _span_metadata(record: TraceRecord, name: str) -> dict[str, object]:
    for span in record.spans:
        if span.name == name:
            return span.metadata
    raise AssertionError(f"span not found: {name}")


def _all_metadata_text(record: TraceRecord) -> str:
    return json.dumps([span.metadata for span in record.spans], ensure_ascii=False, sort_keys=True)


def _symbol(path: str = "src/app.py") -> Symbol:
    return Symbol(
        name="Alpha",
        qualified_name="Alpha",
        kind=SymbolKind.FUNCTION,
        language="python",
        path=path,
        range=Range(start_line=0, start_col=0, end_line=1, end_col=0),
        selection_range=None,
        parent_id=None,
        signature="def Alpha() -> None",
        doc=None,
        source="test",
        confidence=1.0,
        file_hash="hash-alpha",
        index_version="test-v1",
    )


def _diagnostic(message: str) -> Diagnostic:
    return Diagnostic(
        path="src/app.py",
        range=Range(start_line=3, start_col=4, end_line=3, end_col=10),
        severity=DiagnosticSeverity.ERROR,
        message=message,
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint="fingerprint",
    )


def test_kernel_provider_lsp_index_and_verifier_spans_are_recorded(tmp_path: Path) -> None:
    async def scenario() -> TraceRecord:
        _start_trace()
        set_code_intel_kernel(CodeIntelKernel(create_fake_providers()))
        store = SymbolIndexStore(tmp_path / "symbols.db")
        try:
            raw = await _ainvoke_text(code_outline, {"path": PYTHON_FAKE_PATH, "max_depth": 3})
            assert json.loads(raw)["ok"] is True

            fake_transport = _FakeTransport()
            client = LSPClient(fake_transport, workspace_root=tmp_path, language="python", source="pyright")
            _ = await client.document_symbols("src/app.py")

            await store.initialize()
            symbol = _symbol()
            metadata = CurrentFileForStore(
                path="src/app.py",
                language="python",
                sha256="hash-alpha",
                mtime=1.0,
                size=10,
                grammar_version="grammar-v1",
                query_version="query-v1",
            )
            await store.store_symbols(metadata, [symbol])
            indexed_symbols = await store.get_symbols("src/app.py")
            assert [item.name for item in indexed_symbols] == ["Alpha"]

            _ = compute_delta([], [_diagnostic("Cannot find name 'SECRET_SYMBOL' at line 42")])
        finally:
            await store.close()
            set_code_intel_kernel(None)
        return _end_trace()

    record = _run(scenario())
    names = {span.name for span in record.spans}

    assert "code_intel.code_outline" in names
    assert "code_intel.kernel.dispatch" in names
    assert "code_intel.provider.fake_syntax.outline" in names
    assert "lsp.pyright.textDocument/documentSymbol" in names
    assert "code_intel.index.store_symbols" in names
    assert "code_intel.index.get_symbols" in names
    assert "code_intel.verifier.compute_delta" in names

    kernel_metadata = _span_metadata(record, "code_intel.kernel.dispatch")
    assert kernel_metadata["provider_name"] == "fake_syntax"
    assert kernel_metadata["fallback_chain"] == ["fake_syntax"]

    provider_metadata = _span_metadata(record, "code_intel.provider.fake_syntax.outline")
    assert provider_metadata["result_count"] == 3
    assert provider_metadata["path"] == PYTHON_FAKE_PATH

    lsp_metadata = _span_metadata(record, "lsp.pyright.textDocument/documentSymbol")
    assert lsp_metadata["provider_name"] == "pyright"
    assert lsp_metadata["language"] == "python"
    assert lsp_metadata["result_count"] == 0

    verifier_metadata = _span_metadata(record, "code_intel.verifier.compute_delta")
    assert verifier_metadata["diagnostic_templates"] == ["Cannot find name <quoted> at line <line>"]

    serialized = _all_metadata_text(record)
    assert "SECRET_SYMBOL" not in serialized
    assert str(tmp_path) not in serialized
    assert "textDocument" not in json.dumps(lsp_metadata, ensure_ascii=False)


def test_tracing_disabled_or_unstarted_does_not_change_kernel_result() -> None:
    async def scenario() -> None:
        _ = get_tracer().end_trace()
        kernel = CodeIntelKernel(create_fake_providers())
        result = await kernel.call("outline", "python", path=PYTHON_FAKE_PATH)

        assert result.ok is True
        assert result.meta.sources_used == ["fake_syntax"]
        assert get_tracer().active_record is None

    _run(scenario())
