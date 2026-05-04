"""Tests for semantic diagnostics delta computation."""

from __future__ import annotations

from src.code_intel.core import Diagnostic, DiagnosticSeverity, Range, Symbol, SymbolKind
from src.code_intel.verifier import (
    BaselineSnapshot,
    PatchLineEdit,
    PatchSummary,
    RepairAction,
    RepairPolicy,
    VerificationStatus,
    build_baseline_key,
    compute_delta,
)

_PATH = "src/app.py"


def _range(start_line: int, start_col: int = 0, end_line: int | None = None, end_col: int = 1) -> Range:
    return Range(start_line=start_line, start_col=start_col, end_line=start_line if end_line is None else end_line, end_col=end_col)


def _diagnostic(
    line: int,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    message: str = "Cannot find name 'foo'",
    code: str = "reportUndefinedVariable",
    path: str = _PATH,
) -> Diagnostic:
    return Diagnostic(
        path=path,
        range=_range(line),
        severity=severity,
        message=message,
        code=code,
        source="pyright",
        fingerprint=f"raw-{path}-{line}-{severity.value}-{message}",
    )


def _symbol(name: str, qualified_name: str, start_line: int, end_line: int, *, path: str = _PATH) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=SymbolKind.FUNCTION,
        language="python",
        path=path,
        range=Range(start_line=start_line, start_col=0, end_line=end_line, end_col=0),
        selection_range=Range(start_line=start_line, start_col=4, end_line=start_line, end_col=4 + len(name)),
        signature=f"def {name}() -> None",
        doc=None,
        source="test",
        confidence=1.0,
        file_hash=f"hash-{qualified_name}-{start_line}",
        index_version="test-v1",
    )


def _baseline_snapshot(diagnostics: list[Diagnostic], symbols: list[Symbol]) -> BaselineSnapshot:
    key = build_baseline_key(
        workspace_root=".",
        relevant_paths=[_PATH],
        provider_id="test-provider",
        provider_config_hash="test-config",
        call_source="workflow",
        content_hash="content",
    )
    return BaselineSnapshot(key=key, diagnostics=diagnostics, symbols_by_path={_PATH: symbols})


def test_new_error_blocks_patch_gate() -> None:
    diagnostic = _diagnostic(4)

    delta = compute_delta([], [diagnostic], PatchSummary())
    decision = RepairPolicy(max_rounds=2).decide(delta, round=0)

    assert delta.new_diagnostics == [diagnostic]
    assert delta.severity_delta["error"] == 1
    assert decision.action == RepairAction.REPAIR
    assert decision.status == VerificationStatus.BLOCKED


def test_resolved_error_and_warning_default_partial_cases() -> None:
    old_error = _diagnostic(4)
    resolved_delta = compute_delta([old_error], [], PatchSummary())

    assert resolved_delta.resolved_diagnostics == [old_error]
    assert resolved_delta.severity_delta["error"] == -1
    assert RepairPolicy().decide(resolved_delta).status == VerificationStatus.SUCCESS

    warning = _diagnostic(
        8,
        severity=DiagnosticSeverity.WARNING,
        message="Variable 'value' is unused",
        code="reportUnusedVariable",
    )
    warning_delta = compute_delta([], [warning], PatchSummary())
    decision = RepairPolicy().decide(warning_delta)

    assert warning_delta.new_diagnostics == [warning]
    assert warning_delta.severity_delta["warning"] == 1
    assert decision.action == RepairAction.PROCEED
    assert decision.status == VerificationStatus.PARTIAL


def test_semantic_anchor_survives_unrelated_line_edits() -> None:
    baseline_diagnostic = _diagnostic(6, message="Cannot find name 'before_name'")
    after_diagnostic = _diagnostic(9, message="Cannot find name 'after_name'")
    baseline_symbol = _symbol("run", "Service.run", 3, 8)
    after_symbol = _symbol("run", "Service.run", 6, 11)

    delta = compute_delta(
        _baseline_snapshot([baseline_diagnostic], [baseline_symbol]),
        [after_diagnostic],
        PatchSummary(edits=[PatchLineEdit(path=_PATH, start_line=0, old_line_count=0, new_line_count=3)]),
        after_symbols_by_path={_PATH: [after_symbol]},
    )

    assert delta.new_diagnostics == []
    assert delta.resolved_diagnostics == []
    assert delta.unchanged_diagnostics == [after_diagnostic]
    assert delta.unchanged[0].semantic_anchor == "Service.run"


def test_deleted_enclosing_symbol_resolves_baseline_diagnostic() -> None:
    baseline_diagnostic = _diagnostic(4, message="Cannot find name 'removed_value'")
    removed_symbol = _symbol("removed", "removed", 2, 6)

    delta = compute_delta(
        _baseline_snapshot([baseline_diagnostic], [removed_symbol]),
        [],
        PatchSummary(),
        after_symbols_by_path={_PATH: []},
    )

    assert delta.new_diagnostics == []
    assert delta.resolved_diagnostics == [baseline_diagnostic]
    assert delta.resolved[0].semantic_anchor == "<deleted>"


def test_line_drift_from_patch_summary_without_symbols() -> None:
    baseline_diagnostic = _diagnostic(5, message="Expected 1 argument, got 2")
    after_diagnostic = _diagnostic(7, message="Expected 3 argument, got 4")

    delta = compute_delta(
        [baseline_diagnostic],
        [after_diagnostic],
        PatchSummary(edits=[PatchLineEdit(path=_PATH, start_line=2, old_line_count=0, new_line_count=2)]),
    )

    assert delta.new_diagnostics == []
    assert delta.resolved_diagnostics == []
    assert delta.unchanged_diagnostics == [after_diagnostic]
    assert delta.unchanged[0].semantic_anchor == "<line:7>"
