"""Standalone contract: enclosing semantic anchor survives unrelated edits and line drift.

Plan acceptance criterion T13-AC6: prove that editing elsewhere in the file
does not break baseline-to-after diagnostic matching when the enclosing
symbol is still present (just shifted).
"""

from __future__ import annotations

from src.code_intel.core import Diagnostic, DiagnosticSeverity, Range, Symbol, SymbolKind
from src.code_intel.verifier import (
    BaselineSnapshot,
    PatchLineEdit,
    PatchSummary,
    build_baseline_key,
    compute_delta,
)

_PATH = "src/app.py"


def _range(start_line: int, start_col: int = 0, end_line: int | None = None, end_col: int = 1) -> Range:
    return Range(
        start_line=start_line,
        start_col=start_col,
        end_line=start_line if end_line is None else end_line,
        end_col=end_col,
    )


def _diagnostic(
    line: int,
    message: str = "Cannot find name 'foo'",
    code: str = "reportUndefinedVariable",
    path: str = _PATH,
) -> Diagnostic:
    return Diagnostic(
        path=path,
        range=_range(line),
        severity=DiagnosticSeverity.ERROR,
        message=message,
        code=code,
        source="pyright",
        fingerprint=f"raw-{path}-{line}-{message}",
    )


def _symbol(
    name: str,
    qualified_name: str,
    start_line: int,
    end_line: int,
    *,
    path: str = _PATH,
) -> Symbol:
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


def test_anchor_survives_unrelated_edits() -> None:
    """Unrelated edits elsewhere in the file shift diagnostic line numbers.

    The enclosing symbol anchor 'Service.run' is still present after the edit
    (just shifted down by 3 lines), so the baseline and after diagnostics
    should match as unchanged.
    """
    baseline_diagnostic = _diagnostic(6, message="Cannot find name 'before_name'")
    after_diagnostic = _diagnostic(9, message="Cannot find name 'after_name'")
    baseline_symbol = _symbol("run", "Service.run", 3, 8)
    after_symbol = _symbol("run", "Service.run", 6, 11)

    # Insert 3 lines at line 0 (unrelated edit before the enclosing symbol)
    delta = compute_delta(
        _baseline_snapshot([baseline_diagnostic], [baseline_symbol]),
        [after_diagnostic],
        PatchSummary(edits=[PatchLineEdit(path=_PATH, start_line=0, old_line_count=0, new_line_count=3)]),
        after_symbols_by_path={_PATH: [after_symbol]},
    )

    assert delta.new_diagnostics == [], "no new diagnostics expected"
    assert delta.resolved_diagnostics == [], "baseline diagnostic should not be resolved as deleted"
    assert delta.unchanged_diagnostics == [after_diagnostic], "baseline diagnostic should match after"
    assert delta.unchanged[0].semantic_anchor == "Service.run", (
        "semantic anchor should be the enclosing symbol, not a line-number fallback"
    )


def test_anchor_falls_back_to_line_when_no_enclosing_symbol() -> None:
    """Without enclosing symbols the anchor falls back to <line:N> after drift.

    Even without symbols, the drift calculation should still match baseline
    diagnostics with their shifted-line identities.
    """
    baseline_diagnostic = _diagnostic(5, message="Expected 1 argument, got 2")
    after_diagnostic = _diagnostic(7, message="Expected 3 argument, got 4")

    delta = compute_delta(
        [baseline_diagnostic],
        [after_diagnostic],
        PatchSummary(edits=[PatchLineEdit(path=_PATH, start_line=2, old_line_count=0, new_line_count=2)]),
    )

    assert delta.new_diagnostics == [], "no new diagnostics with line-only anchor"
    assert delta.resolved_diagnostics == [], "baseline should not be treated as deleted"
    assert delta.unchanged_diagnostics == [after_diagnostic]
    assert delta.unchanged[0].semantic_anchor == "<line:7>", "fallback anchor should reflect drifted line"
