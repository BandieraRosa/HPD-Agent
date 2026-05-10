"""Full code_intel baseline integration hardening."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine, Generator, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, cast

import pytest
from langchain_core.tools import BaseTool
from lsprotocol import types as lsp_types
from typing_extensions import override

from src.agents import QueryAgent
from src.code_intel import CodeIntelKernel
from src.code_intel.config import (
    CodeIntelConfig,
    CodeIntelConfigLoadResult,
    code_intel_index_db_path,
)
from src.code_intel.core import (
    Capability,
    CodeTarget,
    ContextPart,
    Diagnostic,
    DiagnosticSeverity,
    Location,
    ProviderStatus,
    Range,
    Symbol,
)
from src.code_intel.index import SymbolIndexer, SymbolIndexStore
from src.code_intel.providers.fake import (
    PYTHON_FAKE_PATH,
    create_fake_providers,
    fake_symbols,
)
from src.code_intel.providers.lsp import LSPManager, LSPProvider, LanguageServerSpec
from src.code_intel.providers.text_search import TextSearchProvider
from src.code_intel.providers.tree_sitter import (
    TREE_SITTER_INDEX_VERSION,
    TREE_SITTER_QUERY_VERSION,
    TreeSitterProvider,
)
from src.code_intel.tools import (
    code_context,
    code_outline,
    code_search,
    code_semantic,
    code_verify,
    set_code_intel_kernel,
)
from src.code_intel.verifier import clear_baseline_cache
from src.code_intel.workflow.patch_gate import PatchGate, PatchGateConfig
from src.commands import handle_command

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


def _ok_data(raw: str) -> dict[str, object]:
    payload = _payload(raw)
    assert payload["ok"] is True
    return cast(dict[str, object], payload["data"])


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


def _diagnostic(
    path: str, message: str = "Mock LSP warning for deterministic baseline."
) -> Diagnostic:
    return Diagnostic(
        path=path,
        range=_range(2, 8, 2, 13),
        severity=DiagnosticSeverity.WARNING,
        message=message,
        code="mock-warning",
        source="mock-lsp",
        fingerprint=f"mock-lsp:{path}:{message}",
    )


def _symbols(data: object) -> list[Symbol]:
    assert isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray))
    return [
        item if isinstance(item, Symbol) else Symbol.model_validate(item)
        for item in data
    ]


def _diagnostics(data: object) -> list[Diagnostic]:
    assert isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray))
    return [
        item if isinstance(item, Diagnostic) else Diagnostic.model_validate(item)
        for item in data
    ]


def _locations(data: object) -> list[Location]:
    assert isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray))
    return [
        item if isinstance(item, Location) else Location.model_validate(item)
        for item in data
    ]


def _write_workspace(workspace: Path) -> str:
    source = """class AlphaService:
    def run(self, value: str) -> str:
        return helper(value)


def helper(value: str) -> str:
    return f"alpha:{value}"
"""
    source_path = workspace / "src" / "app.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    _ = source_path.write_text(source, encoding="utf-8")
    return source


def _spec() -> LanguageServerSpec:
    return LanguageServerSpec(
        language="python",
        name="mock-lsp",
        detect_command=["mock-lsp", "--version"],
        launch_command=["mock-lsp", "--stdio"],
        install_hint="npm i -g pyright",
        root_markers=["pyproject.toml", ".git"],
    )


class _FakeDiagnosticsClient:
    def __init__(self, diagnostics: list[Diagnostic], *, running: bool = True) -> None:
        self.diagnostics_result: list[Diagnostic] = list(diagnostics)
        self.running: bool = running
        self.initialize_calls: int = 0
        self.shutdown_calls: int = 0
        self.close_calls: int = 0
        self.diagnostics_calls: int = 0
        self.did_open_calls: list[tuple[str, str, int]] = []
        self.did_change_calls: list[tuple[str, str, int]] = []
        self.did_save_calls: list[tuple[str, str | None]] = []

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
                text_document_sync=lsp_types.TextDocumentSyncKind.Full
            )
        )

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def is_running(self) -> bool:
        return self.running

    async def diagnostics(self, path: str) -> list[Diagnostic]:
        self.diagnostics_calls += 1
        return [
            diagnostic.model_copy(update={"path": path})
            for diagnostic in self.diagnostics_result
        ]

    async def did_open(
        self, path: str, *, language_id: str, text: str, version: int
    ) -> None:
        _ = language_id
        self.did_open_calls.append((path, text, version))

    async def did_change(self, path: str, *, text: str, version: int) -> None:
        self.did_change_calls.append((path, text, version))

    async def did_save(self, path: str, *, text: str | None = None) -> None:
        self.did_save_calls.append((path, text))


class _FakeLSPFactory:
    def __init__(self, clients: list[_FakeDiagnosticsClient]) -> None:
        self.clients: list[_FakeDiagnosticsClient] = list(clients)
        self.created: list[_FakeDiagnosticsClient] = []

    def __call__(
        self,
        spec: LanguageServerSpec,
        workspace_root: Path,
        command: Sequence[str],
        *,
        request_timeout_seconds: float,
    ) -> _FakeDiagnosticsClient:
        _ = spec, workspace_root, command, request_timeout_seconds
        client = self.clients.pop(0)
        self.created.append(client)
        return client


class _NoDetectLSPManager(LSPManager):
    def __init__(self, workspace_root: Path, factory: _FakeLSPFactory) -> None:
        super().__init__(workspace_root, specs=[_spec()], client_factory=factory)
        self.detected: list[str] = []

    @override
    async def detect_server(self, language: str) -> None:
        self.detected.append(language)


class _RecordingInvalidator:
    def __init__(self) -> None:
        self.paths: list[str] = []

    async def invalidate_paths(self, paths: list[str]) -> None:
        self.paths.extend(paths)


def _agent(runtime: object | None = None) -> QueryAgent:
    if runtime is None:
        return cast(QueryAgent, object())
    agent = type("Agent", (), {"lsp_provider": runtime})()
    return cast(QueryAgent, cast(object, agent))


@pytest.fixture(autouse=True)
def clean_code_intel_runtime() -> Generator[None, None, None]:
    clear_baseline_cache()
    set_code_intel_kernel(None)
    yield
    set_code_intel_kernel(None)
    clear_baseline_cache()


def test_full_kernel_baseline_components_coexist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    _ = _write_workspace(workspace)
    config = CodeIntelConfig(cache_dir=str(tmp_path / "index-cache"))
    db_path = code_intel_index_db_path(workspace, config)
    loaded_config = CodeIntelConfigLoadResult(config=config)
    monkeypatch.setattr(
        "src.commands.handlers.index_cmd.load_code_intel_config", lambda: loaded_config
    )
    monkeypatch.setattr(
        "src.commands.handlers.lsp_cmd.load_code_intel_config", lambda: loaded_config
    )

    async def scenario() -> tuple[LSPProvider, _FakeLSPFactory]:
        fake_run_symbol = next(
            symbol
            for symbol in fake_symbols()
            if symbol.qualified_name == "FakeService.run"
        )
        set_code_intel_kernel(CodeIntelKernel(create_fake_providers()))

        fake_outline = _ok_data(
            await _ainvoke_text(
                code_outline, {"path": PYTHON_FAKE_PATH, "max_depth": 3}
            )
        )
        fake_search = _ok_data(
            await _ainvoke_text(
                code_search, {"query": "fake", "mode": "mixed", "limit": 5}
            )
        )
        fake_context = _ok_data(
            await _ainvoke_text(
                code_context,
                {
                    "target": {"symbol_id": fake_run_symbol.id},
                    "include": [
                        "signature",
                        "body",
                        "parents",
                        "imports",
                        "nearby_symbols",
                    ],
                    "max_tokens": 256,
                },
            )
        )
        fake_semantic = _ok_data(
            await _ainvoke_text(
                code_semantic,
                {"operation": "hover", "target": {"symbol_id": fake_run_symbol.id}},
            )
        )
        fake_verify = _ok_data(
            await _ainvoke_text(
                code_verify,
                {"scope": "file", "paths": [PYTHON_FAKE_PATH], "baseline": True},
            )
        )

        assert [
            symbol["name"]
            for symbol in cast(list[dict[str, object]], fake_outline["symbols"])
        ] == [
            "FakeService",
            "run",
            "helper",
        ]
        assert len(cast(list[dict[str, object]], fake_search["matches"])) >= 3
        assert fake_context["signature"] == "def run(self, value: str) -> str"
        assert (
            cast(dict[str, object], fake_semantic["hover"])["source"] == "fake_semantic"
        )
        assert fake_verify["call_source"] == "agent"
        assert fake_verify["verification_status"] == "success"

        tree_provider = TreeSitterProvider(workspace)

        async def extract_symbols(
            workspace_root: Path, path: str, language: str
        ) -> Sequence[Symbol]:
            _ = workspace_root, language
            return await tree_provider.outline(path)

        store = SymbolIndexStore(db_path)
        indexer = SymbolIndexer(
            workspace,
            extractor=extract_symbols,
            store=store,
            grammar_version=TREE_SITTER_INDEX_VERSION,
            query_version=TREE_SITTER_QUERY_VERSION,
        )
        first_lsp_client = _FakeDiagnosticsClient([_diagnostic("src/app.py")])
        recovered_lsp_client = _FakeDiagnosticsClient(
            [_diagnostic("src/app.py", "Recovered mock warning.")]
        )
        factory = _FakeLSPFactory([first_lsp_client, recovered_lsp_client])
        lsp_provider = LSPProvider(
            workspace,
            manager=_NoDetectLSPManager(workspace, factory),
            name="mock_lsp",
            languages={"python"},
        )
        text_provider = TextSearchProvider(workspace)
        kernel = CodeIntelKernel(
            [tree_provider, text_provider, lsp_provider, *create_fake_providers()],
            symbol_index=store,
            workspace_root=workspace,
        )
        set_code_intel_kernel(kernel)
        try:
            index_result = await indexer.index_workspace()
            assert index_result.indexed == 1
            assert index_result.errors == 0
            assert db_path.exists()

            indexed_search = await store.search_symbols("AlphaService", limit=5)
            assert "AlphaService" in {symbol.name for symbol in indexed_search}

            outline = await kernel.call(Capability.OUTLINE, "python", path="src/app.py")
            assert outline.ok is True
            assert {symbol.name for symbol in _symbols(outline.data)} >= {
                "AlphaService",
                "run",
                "helper",
            }
            assert outline.meta.sources_used == ["tree_sitter"]

            indexed_symbols = await kernel.call(
                Capability.DOCUMENT_SYMBOLS, "python", path="src/app.py"
            )
            assert indexed_symbols.ok is True
            assert [symbol.name for symbol in _symbols(indexed_symbols.data)][:1] == [
                "AlphaService"
            ]
            assert indexed_symbols.meta.sources_used == ["symbol_index"]

            text_search = await kernel.call(
                Capability.TEXT_SEARCH, "python", query="AlphaService", limit=5
            )
            assert text_search.ok is True
            assert _locations(text_search.data)[0].path == "src/app.py"
            assert text_search.meta.sources_used == ["text_search"]

            diagnostics = await kernel.call(
                Capability.DIAGNOSTICS, "python", path="src/app.py"
            )
            assert diagnostics.ok is True
            assert [item.source for item in _diagnostics(diagnostics.data)] == [
                "mock-lsp"
            ]
            assert diagnostics.meta.sources_used == ["mock_lsp"]
            assert first_lsp_client.initialize_calls == 1
            assert first_lsp_client.diagnostics_calls == 1

            first_lsp_client.running = False
            unhealthy = await lsp_provider.check_health("python")
            assert unhealthy.status == ProviderStatus.UNAVAILABLE
            fallback = await kernel.call(
                Capability.DIAGNOSTICS, "python", path=PYTHON_FAKE_PATH
            )
            assert fallback.ok is True
            assert fallback.meta.sources_used == ["fake_semantic"]
            assert first_lsp_client.diagnostics_calls == 1

            restored = await lsp_provider.restart("python")
            assert restored.status == ProviderStatus.HEALTHY
            recovered = await kernel.call(
                Capability.DIAGNOSTICS, "python", path="src/app.py"
            )
            assert recovered.ok is True
            assert _diagnostics(recovered.data)[0].message == "Recovered mock warning."
            assert recovered.meta.sources_used == ["mock_lsp"]
            assert recovered_lsp_client.initialize_calls == 1
            assert recovered_lsp_client.diagnostics_calls == 1

            target = CodeTarget(
                location=Location(path="src/app.py", range=_range(1, 8, 1, 11))
            )
            context_result = await kernel.call(
                Capability.CONTEXT_EXTRACT,
                "python",
                target=target,
                include={ContextPart.SIGNATURE, ContextPart.BODY},
                max_tokens=256,
            )
            assert context_result.ok is True
            assert context_result.meta.sources_used == ["symbol_index"]

            invalidator = _RecordingInvalidator()
            gate = PatchGate(
                PatchGateConfig(
                    workspace_root=workspace,
                    verify_changed=True,
                    invalidate_index=True,
                    notify_lsp_did_change=True,
                ),
                kernel=kernel,
                index_invalidator=invalidator,
                lsp_notifier=lsp_provider,
            )
            gate_result = await gate.after_apply_patch(["src/app.py"])
            assert gate_result.changed_files == ["src/app.py"]
            assert gate_result.verify_scope == "changed"
            assert gate_result.workspace_verify_started is False
            assert gate_result.verification is not None
            assert gate_result.verification.call_source == "workflow"
            assert gate_result.verification.verification_status == "partial"
            assert gate_result.verification.recommended_next_action == "proceed"
            assert gate_result.verification.baseline_refreshed is False
            assert [
                diagnostic.message
                for diagnostic in gate_result.verification.new_diagnostics
            ] == ["Recovered mock warning."]
            assert invalidator.paths == ["src/app.py"]
            assert gate_result.notified_paths == ["src/app.py"]
            assert recovered_lsp_client.did_open_calls[0][0] == "src/app.py"
            assert recovered_lsp_client.did_change_calls[0][0] == "src/app.py"

            full_tool_search = _payload(
                await _ainvoke_text(
                    code_search, {"query": "AlphaService", "mode": "text", "limit": 5}
                )
            )
            serialized_tool_payload = json.dumps(
                full_tool_search, ensure_ascii=False, sort_keys=True
            )
            assert full_tool_search["ok"] is True
            assert str(workspace) not in serialized_tool_payload
            assert "lsp_request" not in serialized_tool_payload
            assert "lsp_response" not in serialized_tool_payload
            return lsp_provider, factory
        finally:
            await store.close()

    lsp_provider, factory = _run(scenario())
    created_before_commands = len(factory.created)

    assert _run(handle_command("/index status", _agent())) is False
    index_output = capsys.readouterr().out
    assert index_output.startswith("代码索引状态")
    assert "状态: 已建立" in index_output
    assert "符号数:" in index_output

    assert _run(handle_command("/lsp status", _agent(lsp_provider))) is False
    lsp_output = capsys.readouterr().out
    assert lsp_output.startswith("语言服务器 LSP 状态")
    assert "status 不会启动 server" in lsp_output
    assert "runtime: 已绑定" in lsp_output
    assert "python / mock-lsp: 运行中" in lsp_output
    assert len(factory.created) == created_before_commands
