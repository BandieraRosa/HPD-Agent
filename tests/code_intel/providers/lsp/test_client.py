"""Tests for the typed LSP client boundary."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

import pytest
from lsprotocol import types as lsp_types

from src.code_intel.core import (
    Diagnostic,
    DiagnosticSeverity,
    LSPTimeout,
    ProviderUnavailable,
    Symbol,
    SymbolKind,
)
from src.code_intel.providers.lsp import LSPClient, LSPTransport

T = TypeVar("T")

_MOCK_CLIENT_SERVER = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEPARATOR = b"\r\n\r\n"


def read_message():
    header = b""
    while not header.endswith(SEPARATOR):
        chunk = sys.stdin.buffer.read(1)
        if not chunk:
            return None
        header += chunk
    length = None
    for line in header[:-len(SEPARATOR)].split(b"\r\n"):
        name, separator, value = line.partition(b":")
        if separator and name.lower() == b"content-length":
            length = int(value.strip())
    if length is None:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def write_message(payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


def file_uri(root, relative):
    return (Path(root) / relative).resolve().as_uri()


def initialize_result():
    return {
        "capabilities": {"documentSymbolProvider": True, "hoverProvider": True},
        "serverInfo": {"name": "mock-lsp", "version": "1"},
    }


def diagnostic_notification(root, empty=False, version=None, message="bad value"):
    diagnostics = [] if empty else [
        {
            "range": {"start": {"line": 2, "character": 4}, "end": {"line": 2, "character": 9}},
            "severity": 1,
            "code": "E001",
            "source": "mock-lsp",
            "message": message,
        }
    ]
    params = {
        "uri": file_uri(root, "src/mock.py"),
        "diagnostics": diagnostics,
    }
    if version is not None:
        params["version"] = version
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": params,
    }


def document_symbols():
    return [
        {
            "name": "MockService",
            "kind": 5,
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 4, "character": 0}},
            "selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 17}},
            "children": [
                {
                    "name": "run",
                    "kind": 6,
                    "range": {"start": {"line": 1, "character": 4}, "end": {"line": 3, "character": 0}},
                    "selectionRange": {"start": {"line": 1, "character": 8}, "end": {"line": 1, "character": 11}},
                }
            ],
        },
        {
            "name": "helper",
            "kind": 12,
            "range": {"start": {"line": 5, "character": 0}, "end": {"line": 6, "character": 0}},
            "selectionRange": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 10}},
        },
    ]


def main():
    mode = sys.argv[1]
    root = sys.argv[2]
    if mode == "crash":
        print("Traceback redacted by client boundary", file=sys.stderr, flush=True)
        raise SystemExit(9)
    while True:
        message = read_message()
        if message is None:
            return
        method = message.get("method")
        if mode == "timeout" and method == "textDocument/documentSymbol":
            continue
        if method == "initialize":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": initialize_result()})
            write_message(diagnostic_notification(root, empty=mode == "empty_diagnostics"))
            continue
        if method == "textDocument/documentSymbol":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": document_symbols()})
            continue
        if method == "textDocument/didChange":
            document_version = message.get("params", {}).get("textDocument", {}).get("version")
            publish_version = document_version
            if mode == "stale_diagnostics" and isinstance(document_version, int):
                publish_version = document_version - 1
            write_message(
                diagnostic_notification(
                    root,
                    empty=mode == "empty_diagnostics",
                    version=publish_version,
                    message=f"version {publish_version}",
                )
            )
            continue
        if method == "initialized":
            continue


if __name__ == "__main__":
    main()
"""


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _write_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    source = workspace / "src/mock.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    _ = source.write_text(
        "class MockService:\n    def run(self):\n        bad\n\ndef helper():\n    pass\n",
        encoding="utf-8",
    )
    return workspace


def _server_command(tmp_path: Path, mode: str, workspace: Path) -> list[str]:
    server_path = tmp_path / f"mock_client_{mode}.py"
    _ = server_path.write_text(_MOCK_CLIENT_SERVER, encoding="utf-8")
    return [sys.executable, str(server_path), mode, str(workspace)]


async def _wait_for_diagnostics(client: LSPClient, path: str) -> list[Diagnostic]:
    for _ in range(50):
        diagnostics = await client.diagnostics(path)
        if diagnostics:
            return diagnostics
        await asyncio.sleep(0.01)
    return []


def test_mock_lsp_initialize_and_publish_diagnostics(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "normal", workspace), default_timeout=1.0
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            initialize_result = await client.initialize()
            diagnostics = await _wait_for_diagnostics(client, "src/mock.py")
        finally:
            await client.close()

        assert isinstance(initialize_result, lsp_types.InitializeResult)
        assert initialize_result.capabilities.document_symbol_provider is True
        assert [type(diagnostic) for diagnostic in diagnostics] == [Diagnostic]
        assert diagnostics[0].path == "src/mock.py"
        assert diagnostics[0].severity == DiagnosticSeverity.ERROR
        assert diagnostics[0].code == "E001"
        assert diagnostics[0].source == "mock-lsp"
        assert diagnostics[0].fingerprint

    _run(scenario())


def test_wait_for_diagnostics_treats_empty_publish_as_snapshot(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "empty_diagnostics", workspace),
            default_timeout=1.0,
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            _ = await client.initialize()
            ready = await client.wait_for_diagnostics("src/mock.py", timeout=1.0)
            diagnostics = await client.diagnostics("src/mock.py")
        finally:
            await client.close()

        assert ready is True
        assert diagnostics == []

    _run(scenario())


def test_wait_for_diagnostics_can_wait_for_next_snapshot(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "normal", workspace), default_timeout=1.0
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            _ = await client.initialize()
            assert await client.wait_for_diagnostics("src/mock.py", timeout=1.0) is True
            previous_version = client.diagnostics_version("src/mock.py")
            await client.did_change("src/mock.py", text="changed", version=2)
            ready = await client.wait_for_diagnostics(
                "src/mock.py",
                timeout=1.0,
                after_version=previous_version,
                document_version=2,
            )
            next_version = client.diagnostics_version("src/mock.py")
            diagnostics = await client.diagnostics("src/mock.py")
        finally:
            await client.close()

        assert ready is True
        assert next_version > previous_version
        assert diagnostics[0].message == "version 2"

    _run(scenario())


def test_wait_for_diagnostics_rejects_stale_document_version(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "stale_diagnostics", workspace),
            default_timeout=1.0,
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            _ = await client.initialize()
            assert await client.wait_for_diagnostics("src/mock.py", timeout=1.0) is True
            previous_version = client.diagnostics_version("src/mock.py")
            await client.did_change("src/mock.py", text="changed", version=2)
            ready = await client.wait_for_diagnostics(
                "src/mock.py",
                timeout=0.05,
                after_version=previous_version,
                document_version=2,
            )
            next_version = client.diagnostics_version("src/mock.py")
            diagnostics = await client.diagnostics("src/mock.py")
        finally:
            await client.close()

        assert ready is False
        assert next_version > previous_version
        assert diagnostics[0].message == "version 1"

    _run(scenario())


def test_document_symbols_are_core_symbols_not_raw_payloads(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "normal", workspace), default_timeout=1.0
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            _ = await client.initialize()
            symbols = await client.document_symbols("src/mock.py")
        finally:
            await client.close()

        assert [type(symbol) for symbol in symbols] == [Symbol, Symbol, Symbol]
        assert [symbol.name for symbol in symbols] == ["MockService", "run", "helper"]
        assert [symbol.kind for symbol in symbols] == [
            SymbolKind.CLASS,
            SymbolKind.METHOD,
            SymbolKind.FUNCTION,
        ]
        assert [symbol.path for symbol in symbols] == [
            "src/mock.py",
            "src/mock.py",
            "src/mock.py",
        ]
        assert symbols[1].parent_id == symbols[0].id
        assert symbols[1].qualified_name == "MockService.run"
        assert symbols[0].source == "mock_lsp"

    _run(scenario())


def test_client_timeout_uses_lsp_timeout(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "timeout", workspace), default_timeout=0.05
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            _ = await client.initialize()
            with pytest.raises(LSPTimeout):
                _ = await client.document_symbols("src/mock.py")
        finally:
            await client.close()

    _run(scenario())


def test_client_server_crash_maps_to_provider_unavailable_without_raw_trace(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "crash", workspace), default_timeout=0.5
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            with pytest.raises(ProviderUnavailable) as raised:
                _ = await client.initialize()
        finally:
            await client.close()

        assert "Traceback" not in str(raised.value)

    _run(scenario())


def test_invalid_workspace_path_is_provider_unavailable(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        transport = LSPTransport(
            _server_command(tmp_path, "normal", workspace), default_timeout=1.0
        )
        client = LSPClient(
            transport, workspace_root=workspace, language="python", source="mock_lsp"
        )
        try:
            with pytest.raises(ProviderUnavailable):
                _ = await client.document_symbols("../outside.py")
        finally:
            await client.close()

    _run(scenario())
