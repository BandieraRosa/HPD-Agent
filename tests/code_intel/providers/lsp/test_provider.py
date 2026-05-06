"""Behavior tests for the routed LSP provider."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import TypeVar, cast

import pytest
from lsprotocol import types as lsp_types

from src.code_intel.core import (
    Capability,
    CodeTarget,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    LSPTimeout,
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
        if method == "textDocument/didChange":
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
    _ = source.write_text(
        "def mock_function():\n    value = 1\n    return value\n", encoding="utf-8"
    )
    return workspace


def _server_spec(
    tmp_path: Path, workspace: Path, log_path: Path, mode: str = "full"
) -> LanguageServerSpec:
    server_path = tmp_path / f"mock_provider_{mode}.py"
    _ = server_path.write_text(_MOCK_PROVIDER_SERVER, encoding="utf-8")
    return LanguageServerSpec(
        language="python",
        name="mock-lsp",
        detect_command=[sys.executable, "--version"],
        launch_command=[
            sys.executable,
            str(server_path),
            str(workspace),
            str(log_path),
            mode,
        ],
        install_hint="python mock provider server",
        root_markers=[".git"],
    )


def _target(path: str = "src/mock.py") -> CodeTarget:
    return CodeTarget(
        location=Location(
            path=path, range=Range(start_line=0, start_col=4, end_line=0, end_col=8)
        )
    )


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
    return [
        cast(dict[str, object], json.loads(line))
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]


def _fake_spec() -> LanguageServerSpec:
    return LanguageServerSpec(
        language="python",
        name="fake-lsp",
        detect_command=[sys.executable, "--version"],
        launch_command=[sys.executable, "fake-lsp.py"],
        install_hint="python fake lsp",
        root_markers=[".git"],
    )


class _FreshDiagnosticsClient:
    def __init__(
        self,
        *,
        publish_delay: float = 0.0,
        publish_changes: bool = True,
        stale_change_publish: bool = False,
        publish_asynchronously: bool = False,
        publish_document_versions: bool = True,
    ) -> None:
        self.publish_delay: float = publish_delay
        self.publish_changes: bool = publish_changes
        self.stale_change_publish: bool = stale_change_publish
        self.publish_asynchronously: bool = publish_asynchronously
        self.publish_document_versions: bool = publish_document_versions
        self.initialize_calls: int = 0
        self.shutdown_calls: int = 0
        self.close_calls: int = 0
        self.running: bool = True
        self.did_open_calls: list[tuple[str, str, int]] = []
        self.did_change_calls: list[tuple[str, str, int]] = []
        self.did_save_calls: list[tuple[str, str | None]] = []
        self._diagnostics_by_path: dict[str, list[Diagnostic]] = {}
        self._versions: dict[str, int] = {}
        self._document_versions: dict[str, int | None] = {}
        self._conditions: dict[str, asyncio.Condition] = {}

    async def initialize(
        self,
        *,
        root_uri: str | None = None,
        initialization_options: object | None = None,
    ) -> lsp_types.InitializeResult:
        _ = root_uri, initialization_options
        self.initialize_calls += 1
        return lsp_types.InitializeResult(
            capabilities=lsp_types.ServerCapabilities(
                text_document_sync=lsp_types.TextDocumentSyncKind.Full,
                hover_provider=True,
            )
        )

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def is_running(self) -> bool:
        return self.running

    async def diagnostics(self, path: str) -> list[Diagnostic]:
        return list(self._diagnostics_by_path.get(path, ()))

    def diagnostics_version(self, path: str) -> int:
        return self._versions.get(path, 0)

    async def wait_for_diagnostics(
        self,
        path: str,
        *,
        timeout: float,
        after_version: int | None = None,
        document_version: int | None = None,
    ) -> bool:
        target_version = 0 if after_version is None else after_version
        if self._snapshot_is_fresh(path, target_version, document_version):
            return True
        condition = self._conditions.setdefault(path, asyncio.Condition())

        async def wait_until_newer() -> bool:
            async with condition:
                while not self._snapshot_is_fresh(
                    path, target_version, document_version
                ):
                    _ = await condition.wait()
                return True

        try:
            return await asyncio.wait_for(wait_until_newer(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    def _snapshot_is_fresh(
        self, path: str, after_publish_version: int, document_version: int | None
    ) -> bool:
        if self._versions.get(path, 0) <= after_publish_version:
            return False
        if document_version is None:
            return True
        published_document_version = self._document_versions.get(path)
        if published_document_version is None:
            return True
        return published_document_version >= document_version

    async def did_open(
        self, path: str, *, language_id: str, text: str, version: int
    ) -> None:
        _ = language_id
        self.did_open_calls.append((path, text, version))
        publish_version = version if self.publish_document_versions else None
        await self._schedule_publish(path, text, document_version=publish_version)

    async def did_change(self, path: str, *, text: str, version: int) -> None:
        self.did_change_calls.append((path, text, version))
        if not self.publish_changes:
            return
        publish_version = version - 1 if self.stale_change_publish else version
        if not self.publish_document_versions:
            publish_version = None
        await self._schedule_publish(path, text, document_version=publish_version)

    async def did_save(self, path: str, *, text: str | None = None) -> None:
        self.did_save_calls.append((path, text))

    async def hover(self, path: str, *, line: int, character: int) -> HoverInfo | None:
        return HoverInfo(
            contents=f"hover:{path}:{line}:{character}",
            range=Range(
                start_line=line,
                start_col=character,
                end_line=line,
                end_col=character + 1,
            ),
            source="fresh-lsp",
        )

    async def _schedule_publish(
        self, path: str, text: str, *, document_version: int | None
    ) -> None:
        if self.publish_asynchronously:
            _ = asyncio.create_task(
                self._publish(path, text, document_version=document_version)
            )
            return
        await self._publish(path, text, document_version=document_version)

    async def _publish(
        self, path: str, text: str, *, document_version: int | None
    ) -> None:
        if self.publish_delay > 0:
            await asyncio.sleep(self.publish_delay)
        condition = self._conditions.setdefault(path, asyncio.Condition())
        async with condition:
            self._diagnostics_by_path[path] = [self._diagnostic(path, text)]
            self._versions[path] = self._versions.get(path, 0) + 1
            self._document_versions[path] = document_version
            condition.notify_all()

    @staticmethod
    def _diagnostic(path: str, text: str) -> Diagnostic:
        first_line = text.strip().splitlines()[0] if text.strip() else "empty"
        return Diagnostic(
            path=path,
            range=Range(start_line=0, start_col=0, end_line=0, end_col=1),
            severity=DiagnosticSeverity.WARNING,
            message=f"snapshot:{first_line}",
            code="fresh",
            source="fresh-lsp",
            fingerprint=f"fresh:{path}:{first_line}",
        )


class _FreshDiagnosticsFactory:
    def __init__(self, client: _FreshDiagnosticsClient) -> None:
        self.client: _FreshDiagnosticsClient = client
        self.created: list[_FreshDiagnosticsClient] = []

    def __call__(
        self,
        spec: LanguageServerSpec,
        workspace_root: Path,
        command: Sequence[str],
        *,
        request_timeout_seconds: float,
    ) -> _FreshDiagnosticsClient:
        _ = spec, workspace_root, command, request_timeout_seconds
        self.created.append(self.client)
        return self.client


def test_provider_negotiates_capabilities_and_returns_core_models(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "provider.log"
        spec = _server_spec(tmp_path, workspace, log_path)
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            assert await provider.supports(Capability.DEFINITION, "python") is True
            assert await provider.supports(Capability.REFERENCES, "python") is True
            assert await provider.supports(Capability.HOVER, "python") is True
            assert (
                await provider.supports(Capability.DOCUMENT_SYMBOLS, "python") is True
            )
            assert await provider.supports(Capability.DIAGNOSTICS, "python") is True
            assert await provider.supports(Capability.RENAME, "python") is False
            assert Capability.RENAME not in provider.capabilities

            definition = await provider.goto_definition(_target())
            references = await provider.find_references(_target())
            hover = await provider.hover(_target())
            symbols = await provider.document_symbols("src/mock.py")
            await provider.notify_did_change(
                "src/mock.py", "def mock_function():\n    return 2\n"
            )
            await provider.notify_did_save(
                "src/mock.py", "def mock_function():\n    return 2\n"
            )
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

        did_notifications = [
            entry
            for entry in _logged_methods(log_path)
            if str(entry["method"]).startswith("textDocument/did")
        ]
        assert [entry["method"] for entry in did_notifications] == [
            "textDocument/didOpen",
            "textDocument/didChange",
            "textDocument/didSave",
        ]
        versions = [
            cast(
                dict[str, object],
                cast(dict[str, object], entry["params"])["textDocument"],
            ).get("version")
            for entry in did_notifications
        ]
        assert versions == [1, 2, None]

    _run(scenario())


def test_semantic_operations_prefer_identifier_position_from_source_range(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "semantic-position.log"
        spec = _server_spec(tmp_path, workspace, log_path)
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            wide_target = CodeTarget(
                location=Location(
                    path="src/mock.py",
                    range=Range(start_line=0, start_col=0, end_line=0, end_col=20),
                )
            )
            _ = await provider.hover(wide_target)
            _ = await provider.goto_definition(wide_target)
            _ = await provider.find_references(wide_target)
        finally:
            await provider.shutdown()

        requests = [
            entry
            for entry in _logged_methods(log_path)
            if entry["method"]
            in {
                "textDocument/hover",
                "textDocument/definition",
                "textDocument/references",
            }
        ]
        positions = [
            cast(dict[str, object], entry["params"])["position"] for entry in requests
        ]
        assert positions == [
            {"line": 0, "character": 4},
            {"line": 0, "character": 4},
            {"line": 0, "character": 4},
        ]

    _run(scenario())


def test_diagnostics_opens_unsynced_document_on_first_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "diagnostics-open.log"
        spec = _server_spec(tmp_path, workspace, log_path)
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            first = await provider.diagnostics("src/mock.py")
            cached = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert [type(diagnostic) for diagnostic in first] == [Diagnostic]
        assert first[0].message == "mock warning"
        assert cached == first
        did_open = [
            entry
            for entry in _logged_methods(log_path)
            if entry["method"] == "textDocument/didOpen"
        ]
        assert len(did_open) == 1

    _run(scenario())


def test_diagnostics_missing_unsynced_file_reports_safe_read_error(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        log_path = tmp_path / "diagnostics-missing.log"
        spec = _server_spec(tmp_path, workspace, log_path)
        provider = LSPProvider(workspace, manager=LSPManager(workspace, specs=[spec]))
        try:
            with pytest.raises(ProviderUnavailable) as raised:
                _ = await provider.diagnostics("src/missing.py")
        finally:
            await provider.shutdown()

        assert "无法读取文件" in str(raised.value)
        assert _logged_methods(log_path) == []

    _run(scenario())


def test_provider_refuses_unadvertised_semantic_operation_without_requesting_it(
    tmp_path: Path,
) -> None:
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


def test_concurrent_diagnostics_opens_document_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        client = _FreshDiagnosticsClient(publish_delay=0.01)
        factory = _FreshDiagnosticsFactory(client)
        provider = LSPProvider(
            workspace,
            manager=LSPManager(workspace, specs=[_fake_spec()], client_factory=factory),
        )
        try:
            first, second = await asyncio.gather(
                provider.diagnostics("src/mock.py"),
                provider.diagnostics("src/mock.py"),
            )
        finally:
            await provider.shutdown()

        assert len(factory.created) == 1
        assert len(client.did_open_calls) == 1
        assert client.did_change_calls == []
        assert first == second
        assert first[0].message.startswith("snapshot:def mock_function")

    _run(scenario())


def test_concurrent_notify_did_change_serializes_versions(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        client = _FreshDiagnosticsClient()
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
            ),
        )
        try:
            await provider.notify_did_open("src/mock.py", "first")
            await asyncio.gather(
                provider.notify_did_change("src/mock.py", "second"),
                provider.notify_did_change("src/mock.py", "third"),
            )
        finally:
            await provider.shutdown()

        assert [call[2] for call in client.did_open_calls] == [1]
        assert [call[2] for call in client.did_change_calls] == [2, 3]

    _run(scenario())


def test_diagnostics_waits_after_semantic_operation_opens_document(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        client = _FreshDiagnosticsClient(
            publish_delay=0.02, publish_asynchronously=True
        )
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
                request_timeout_seconds=0.5,
            ),
        )
        try:
            hover = await provider.hover(_target())
            assert hover is not None
            diagnostics = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert len(client.did_open_calls) == 1
        assert client.did_change_calls == []
        assert diagnostics[0].message == "snapshot:def mock_function():"
        assert client._document_versions["src/mock.py"] == 1

    _run(scenario())


def test_diagnostics_waits_after_semantic_change_with_versionless_publish(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        source = workspace / "src/mock.py"
        client = _FreshDiagnosticsClient(
            publish_delay=0.02,
            publish_asynchronously=True,
            publish_document_versions=False,
        )
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
                request_timeout_seconds=0.5,
            ),
        )
        try:
            first = await provider.diagnostics("src/mock.py")
            _ = source.write_text(
                "def changed_function():\n    return 2\n", encoding="utf-8"
            )
            hover = await provider.hover(_target())
            assert hover is not None
            second = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert first[0].message == "snapshot:def mock_function():"
        assert second[0].message == "snapshot:def changed_function():"
        assert len(client.did_open_calls) == 1
        assert len(client.did_change_calls) == 1
        assert client._versions["src/mock.py"] == 2
        assert client._document_versions["src/mock.py"] is None

    _run(scenario())


def test_diagnostics_resyncs_open_document_when_disk_changes(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        source = workspace / "src/mock.py"
        client = _FreshDiagnosticsClient()
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
            ),
        )
        try:
            first = await provider.diagnostics("src/mock.py")
            _ = source.write_text(
                "def changed_function():\n    return 2\n", encoding="utf-8"
            )
            second = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert first[0].message == "snapshot:def mock_function():"
        assert second[0].message == "snapshot:def changed_function():"
        assert len(client.did_open_calls) == 1
        assert len(client.did_change_calls) == 1
        assert client.did_change_calls[0][2] == 2

    _run(scenario())


def test_diagnostics_times_out_instead_of_returning_stale_cache(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        source = workspace / "src/mock.py"
        client = _FreshDiagnosticsClient(publish_changes=False)
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
                request_timeout_seconds=0.01,
            ),
        )
        try:
            first = await provider.diagnostics("src/mock.py")
            _ = source.write_text(
                "def changed_function():\n    return 2\n", encoding="utf-8"
            )
            with pytest.raises(LSPTimeout):
                _ = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert first[0].message == "snapshot:def mock_function():"
        assert len(client.did_open_calls) == 1
        assert len(client.did_change_calls) == 1
        assert client.did_change_calls[0][2] == 2

    _run(scenario())


def test_diagnostics_rejects_stale_versioned_publish_after_change(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        source = workspace / "src/mock.py"
        client = _FreshDiagnosticsClient(stale_change_publish=True)
        provider = LSPProvider(
            workspace,
            manager=LSPManager(
                workspace,
                specs=[_fake_spec()],
                client_factory=_FreshDiagnosticsFactory(client),
                request_timeout_seconds=0.01,
            ),
        )
        try:
            first = await provider.diagnostics("src/mock.py")
            _ = source.write_text(
                "def changed_function():\n    return 2\n", encoding="utf-8"
            )
            with pytest.raises(LSPTimeout):
                _ = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()

        assert first[0].message == "snapshot:def mock_function():"
        assert len(client.did_change_calls) == 1
        assert client.did_change_calls[0][2] == 2
        assert client._versions["src/mock.py"] == 2
        assert client._document_versions["src/mock.py"] == 1

    _run(scenario())


def test_diagnostics_rejects_symlink_escape_before_lsp_start(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    external = tmp_path / "outside.py"
    _ = external.write_text("secret = True\n", encoding="utf-8")
    link = workspace / "src/escape.py"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is not available")

    async def scenario() -> None:
        client = _FreshDiagnosticsClient()
        factory = _FreshDiagnosticsFactory(client)
        provider = LSPProvider(
            workspace,
            manager=LSPManager(workspace, specs=[_fake_spec()], client_factory=factory),
        )
        try:
            with pytest.raises(ProviderUnavailable):
                _ = await provider.diagnostics("src/escape.py")
        finally:
            await provider.shutdown()
        assert factory.created == []

    _run(scenario())


def test_diagnostics_rejects_oversized_file_before_lsp_start(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = _write_workspace(tmp_path)
        _ = (workspace / "src/mock.py").write_text("too large\n", encoding="utf-8")
        client = _FreshDiagnosticsClient()
        factory = _FreshDiagnosticsFactory(client)
        provider = LSPProvider(
            workspace,
            manager=LSPManager(workspace, specs=[_fake_spec()], client_factory=factory),
            max_sync_file_size_bytes=4,
        )
        try:
            with pytest.raises(ProviderUnavailable):
                _ = await provider.diagnostics("src/mock.py")
        finally:
            await provider.shutdown()
        assert factory.created == []

    _run(scenario())
