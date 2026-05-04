"""Integration tests for workflow patch gate changed-file handling."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from src.code_intel.core import Capability, Diagnostic, DiagnosticSeverity, ProviderUnavailable, Range, ToolError, ToolResult
from src.code_intel.verifier import clear_baseline_cache
from src.code_intel.workflow.patch_gate import (
    PatchGate,
    PatchGateConfig,
    changed_files_from_patch_text,
    changed_files_from_tool_log,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_workflow_baselines() -> Generator[None, None, None]:
    clear_baseline_cache()
    yield
    clear_baseline_cache()


def _tool_log(*paths: str, dry_run: bool = False) -> str:
    status = "[DRY-RUN OK] 补丁验证通过,计划" if dry_run else "[OK] Applied patch:"
    dry_run_text = "True" if dry_run else "False"
    lines = [f"[Tool: apply_patch(patch_text='*** Begin Patch\\n*** End Patch', dry_run={dry_run_text})]", status]
    for path in paths:
        lines.append(f"- Update: {path} (+1/-1, final 24 bytes)")
    return "\n".join(lines)


def _diagnostic() -> Diagnostic:
    return Diagnostic(
        path="src/app.py",
        range=Range(start_line=0, start_col=0, end_line=0, end_col=1),
        severity=DiagnosticSeverity.ERROR,
        message="Cannot find name 'value'",
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint="raw-src-app",
    )


class _Provider:
    name: str = "diag"
    capabilities: set[Capability] = {Capability.DIAGNOSTICS}
    languages: set[str] = {"python"}


class _Kernel:
    providers: tuple[_Provider, ...] = (_Provider(),)

    def __init__(self, diagnostics: list[Diagnostic] | None = None, error: ToolError | None = None) -> None:
        self.diagnostics: list[Diagnostic] = diagnostics or []
        self.error: ToolError | None = error
        self.calls: list[tuple[Capability, str, dict[str, object]]] = []

    async def call(self, capability: Capability | str, language: str, **kwargs: object) -> ToolResult[object]:
        selected = Capability(capability)
        self.calls.append((selected, language, dict(kwargs)))
        if self.error is not None:
            return ToolResult[object](ok=False, error=self.error)
        return ToolResult[object](ok=True, data=list(self.diagnostics))


@dataclass(frozen=True)
class _RawDiagnosticPayload:
    detail: str = "provider_internal_detail"


class _MalformedDiagnosticsKernel:
    providers: tuple[_Provider, ...] = (_Provider(),)

    async def call(self, capability: Capability | str, language: str, **kwargs: object) -> ToolResult[object]:
        _ = Capability(capability), language, kwargs
        return ToolResult[object](ok=True, data=[_RawDiagnosticPayload()])


class _PathsInvalidator:
    def __init__(self) -> None:
        self.paths: list[str] = []

    async def invalidate_paths(self, paths: list[str]) -> None:
        self.paths.extend(paths)


class _DeleteInvalidator:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_file(self, path: str) -> None:
        self.deleted.append(path)


class _LSPNotifier:
    def __init__(self) -> None:
        self.changes: list[tuple[str, str]] = []

    async def notify_did_change(self, path: str, content: str) -> None:
        self.changes.append((path, content))


def test_changed_files_from_patch_text_and_successful_apply_log_only() -> None:
    patch_text = """*** Begin Patch
*** Update File: src/app.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
*** Add File: src/new.py
<<<<<<< CONTENT
value = 1
>>>>>>> END
*** End Patch"""
    tool_log = "\n\n".join(
        [
            _tool_log("src/dry_run.py", dry_run=True),
            _tool_log("src/app.py", "src/new.py", dry_run=False),
        ]
    )

    assert changed_files_from_patch_text(patch_text) == ["src/app.py", "src/new.py"]
    assert changed_files_from_tool_log(tool_log) == ["src/app.py", "src/new.py"]


def test_patch_gate_invalidates_and_notifies_configured_changed_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = 1\n", encoding="utf-8")
    syntax = _PathsInvalidator()
    index = _DeleteInvalidator()
    notifier = _LSPNotifier()
    gate = PatchGate(
        PatchGateConfig(
            workspace_root=tmp_path,
            invalidate_syntax=True,
            invalidate_index=True,
            notify_lsp_did_change=True,
            verify_changed=False,
        ),
        syntax_invalidator=syntax,
        index_invalidator=index,
        lsp_notifier=notifier,
    )

    result = _run(gate.handle_tool_log(_tool_log("src/app.py")))

    assert result is not None
    assert result.changed_files == ["src/app.py"]
    assert syntax.paths == ["src/app.py"]
    assert index.deleted == ["src/app.py"]
    assert result.invalidated_paths == ["src/app.py"]
    assert notifier.changes == [("src/app.py", "value = 1\n")]
    assert result.notified_paths == ["src/app.py"]
    assert result.verification is None
    assert result.verify_scope == "changed"
    assert result.workspace_verify_started is False


def test_patch_gate_changed_verify_never_uses_workspace_scope(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = 1\n", encoding="utf-8")
    kernel = _Kernel([])
    gate = PatchGate(PatchGateConfig(workspace_root=tmp_path, verify_changed=True), kernel=kernel)

    result = _run(gate.after_apply_patch(["src/app.py"]))

    assert result.verify_scope == "changed"
    assert result.workspace_verify_started is False
    assert kernel.calls == [(Capability.DIAGNOSTICS, "python", {"path": "src/app.py"})]
    assert result.verification is not None
    assert result.verification.call_source == "workflow"
    assert result.verification.checks_run == ["lsp_diagnostics"]
    skipped = {item.check for item in result.verification.checks_skipped}
    assert "workspace_verify" in skipped


def test_patch_gate_missing_workflow_baseline_blocks_new_error(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = missing\n", encoding="utf-8")
    kernel = _Kernel([_diagnostic()])
    gate = PatchGate(PatchGateConfig(workspace_root=tmp_path, verify_changed=True), kernel=kernel)

    result = _run(gate.after_apply_patch(["src/app.py"]))

    assert result.verification is not None
    assert result.verification.ok is False
    assert result.verification.verification_status == "blocked"
    assert result.verification.recommended_next_action == "repair"
    assert result.verification.baseline_key is not None
    assert result.verification.baseline_refreshed is False
    assert result.verification.unchanged_diagnostics == []
    assert [diagnostic.code for diagnostic in result.verification.new_diagnostics] == ["reportUndefinedVariable"]
    assert result.repair_decision is not None
    assert result.repair_decision.action.value == "repair"
    assert result.repair_decision.status.value == "blocked"


def test_patch_gate_invalid_diagnostic_payload_uses_safe_static_hint(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = 1\n", encoding="utf-8")
    gate = PatchGate(
        PatchGateConfig(workspace_root=tmp_path, verify_changed=True),
        kernel=_MalformedDiagnosticsKernel(),
    )

    result = _run(gate.after_apply_patch(["src/app.py"]))

    assert result.verification is not None
    assert result.verification.ok is False
    assert result.verification.provider_error is not None
    assert result.verification.provider_error.code == "invalid_diagnostics"
    assert result.verification.provider_error.hint == "请检查诊断 provider 输出格式。"
    serialized = result.model_dump_json()
    assert "RawDiagnosticPayload" not in serialized
    assert "provider_internal_detail" not in serialized


def test_patch_gate_provider_unavailable_returns_structured_partial(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = 1\n", encoding="utf-8")
    kernel = _Kernel(error=ProviderUnavailable("diagnostics unavailable").to_tool_error())
    gate = PatchGate(PatchGateConfig(workspace_root=tmp_path, verify_changed=True), kernel=kernel)

    result = _run(gate.after_apply_patch(["src/app.py"]))

    assert result.verification is not None
    assert result.verification.ok is False
    assert result.verification.call_source == "workflow"
    assert result.verification.verification_status == "partial"
    assert result.verification.recommended_next_action == "abort"
    assert result.verification.provider_error is not None
    assert result.verification.provider_error.code == "provider_unavailable"
    assert result.repair_decision is not None
    assert result.repair_decision.status.value == "partial"
    assert result.workspace_verify_started is False
