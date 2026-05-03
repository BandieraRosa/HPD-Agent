"""Behavior tests for the routed LSP provider."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar, cast

import pytest

from src.code_intel.core import (
    Capability,
    CodeTarget,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    ProviderUnavailable,
    Range,
    Symbol,
)
from src.code_intel.providers.lsp import LSPManager, LSPProvider, LanguageServerSpec

T = TypeVar("T")

_MOCK_PROVIDER_SERVER = r"""
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


def append_log(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_uri(root, relative):
    return (Path(root) / relative).resolve().as_uri()


def full_capabilities():
    return {
        "definitionProvider": True,
        "referencesProvider": True,
        "hoverProvider": True,
        "documentSymbolProvider": True,
        "textDocumentSync": {"openClose": True, "change": 1, "save": {"includeText": True}},
    }


def hover_only_capabilities():
    return {"hoverProvider": True}


def diagnostic_notification(root):
    return {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": file_uri(root, "src/mock.py"),
            "diagnostics": [
                {
                    "range": {"start": {"line": 1, "character": 4}, "end": {"line": 1, "character": 9}},
                    "severity": 2,
                    "code": "W001",
                    "source": "mock-lsp",
                    "message": "mock warning",
                }
            ],
        },
    }


def location(root, line, character):
    return {
        "uri": file_uri(root, "src/mock.py"),
        "range": {"start": {"line": line, "character": character}, "end": {"line": line, "character": character + 4}},
    }


def document_symbols():
    return [
        {
            "name": "mock_function",
            "kind": 12,
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 13}},
            "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 17}},
        }
    ]


def main():
    root = sys.argv[1]
    log_path = sys.argv[2]
    mode = sys.argv[3]
    capabilities = hover_only_capabilities() if mode == "hover_only" else full_capabilities()
    while True:
        message = read_message()
        if message is None:
            return
        method = message.get("method")
        append_log(log_path, {"method": method, "params": message.get("params", {})})
        if method == "initialize":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": capabilities}})
            continue
        if method == "initialized":
            continue
        if method == "textDocument/didOpen":
            write_message(diagnostic_notification(root))
            continue
        if method == "textDocument/definition":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": [location(root, 0, 4)]})
            continue
        if method == "textDocument/references":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": [location(root, 0, 4), location(root, 2, 8)]})
            continue
        if method == "textDocument/hover":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"contents": {"kind": "markdown", "value": "**mock** hover"}}})
            continue
        if method == "textDocument/documentSymbol":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": document_symbols()})
            continue
        if method == "shutdown":
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": None})
            continue
        if method == "exit":
            return


if __name__ == "__main__":
    main()
"""


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _write_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    source = workspace / "src/mock.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    _ = source.write_text("def mock_function():\n    value = 1\n    return value\n", encoding="utf-8")
    return workspace


def _server_spec(tmp_path: Path, workspace: Path, log_path: Path, mode: str = "full") -> LanguageServerSpec:
    server_path = tmp_path / f"mock_provider_{mode}.py"
    _ = server_path.write_text(_MOCK_PROVIDER_SERVER, encoding="utf-8")
    return LanguageServerSpec(
        language="python",
        name="mock-lsp",
        detect_command=[sys.executable, "--version"],
        launch_command=[sys.executable, str(server_path), str(workspace), str(log_path), mode],
        install_hint="python mock provider server",
        root_markers=[".git"],
    )


def _target(path: str = "src/mock.py") -> CodeTarget:
    return CodeTarget(location=Location(path=path, range=Range(start_line=0, start_col=4, end_line=0, end_col=8)))


async def _wait_for_diagnostics(provider: LSPProvider, path: str) -> list[Diagnostic]:
    for _ in range(50):
        diagnostics = await provider.diagnostics(path)
        if diagnostics:
            return diagnostics
        await asyncio.sleep(0.01)
    return []


def _logged_methods(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []
    return [cast(dict[str, object], json.loads(line)) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_provider_negotiates_capabilities_and_returns_core_models(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "provider.log"
        spec = _server_spec(tmp_path, workspace, log_path)
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            assert await provider.supports(Capability.DEFINITION, "python") is True
            assert await provider.supports(Capability.REFERENCES, "python") is True
            assert await provider.supports(Capability.HOVER, "python") is True
            assert await provider.supports(Capability.DOCUMENT_SYMBOLS, "python") is True
            assert await provider.supports(Capability.DIAGNOSTICS, "python") is True
            assert await provider.supports(Capability.RENAME, "python") is False
            assert Capability.RENAME not in provider.capabilities

            definition = await provider.goto_definition(_target())
            references = await provider.find_references(_target())
            hover = await provider.hover(_target())
            symbols = await provider.document_symbols("src/mock.py")
            await provider.notify_did_change("src/mock.py", "def mock_function():\n    return 2\n")
            await provider.notify_did_save("src/mock.py", "def mock_function():\n    return 2\n")
            diagnostics = await _wait_for_diagnostics(provider, "src/mock.py")
            cached_diagnostics = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert [type(item) for item in definition] == [Location]
        assert [location.range.start_col for location in references] == [4, 8]
        assert isinstance(hover, HoverInfo)
        assert hover.contents == "**mock** hover"
        assert [type(symbol) for symbol in symbols] == [Symbol]
        assert symbols[0].name == "mock_function"
        assert [type(diagnostic) for diagnostic in diagnostics] == [Diagnostic]
        assert diagnostics[0].severity == DiagnosticSeverity.WARNING
        assert cached_diagnostics == diagnostics

        did_notifications = [entry for entry in _logged_methods(log_path) if str(entry["method"]).startswith("textDocument/did")]
        assert [entry["method"] for entry in did_notifications] == [
            "textDocument/didOpen",
            "textDocument/didChange",
            "textDocument/didSave",
        ]
        versions = [cast(dict[str, object], cast(dict[str, object], entry["params"])["textDocument"]).get("version") for entry in did_notifications]
        assert versions == [1, 2, None]

    _run(scenario())


def test_provider_refuses_unadvertised_semantic_operation_without_requesting_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "hover-only.log"
        spec = _server_spec(tmp_path, workspace, log_path, mode="hover_only")
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            assert await provider.supports(Capability.HOVER, "python") is True
            assert await provider.supports(Capability.DEFINITION, "python") is False
            with pytest.raises(ProviderUnavailable):
                _ = await provider.goto_definition(_target())
        finally:
            await provider.shutdown()

        methods = [entry["method"] for entry in _logged_methods(log_path)]
        assert "textDocument/definition" not in methods

    _run(scenario())
