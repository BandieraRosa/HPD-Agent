"""Tests for verifier repair policy decisions."""

from __future__ import annotations

from src.code_intel.core import Diagnostic, DiagnosticSeverity, ProviderUnavailable, Range
from src.code_intel.verifier import DiagnosticsDelta, RepairAction, RepairPolicy, VerificationStatus, compute_delta


def _diagnostic(severity: DiagnosticSeverity) -> Diagnostic:
    return Diagnostic(
        path="src/app.py",
        range=Range(start_line=1, start_col=0, end_line=1, end_col=1),
        severity=severity,
        message="Cannot find name 'value'",
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint=f"raw-{severity.value}",
    )


def test_max_two_repair_rounds_then_retreat() -> None:
    delta = compute_delta([], [_diagnostic(DiagnosticSeverity.ERROR)])
    policy = RepairPolicy(max_rounds=2)

    assert policy.decide(delta, round=0).action == RepairAction.REPAIR
    assert policy.decide(delta, round=1).action == RepairAction.REPAIR
    final_decision = policy.decide(delta, round=2)

    assert final_decision.action == RepairAction.RETREAT
    assert final_decision.status == VerificationStatus.BLOCKED


def test_provider_unavailable_returns_structured_partial_not_success() -> None:
    error = ProviderUnavailable().to_tool_error()
    delta = DiagnosticsDelta.provider_partial(error)

    decision = RepairPolicy(max_rounds=2).decide(delta, round=0)

    assert decision.action == RepairAction.PROCEED
    assert decision.status == VerificationStatus.PARTIAL
    assert decision.status != VerificationStatus.SUCCESS
    assert decision.provider_error == error
    assert decision.provider_error is not None
    assert decision.provider_error.code == "provider_unavailable"


def test_warnings_are_partial_not_blockers_by_default() -> None:
    delta = compute_delta([], [_diagnostic(DiagnosticSeverity.WARNING)])

    decision = RepairPolicy().decide(delta)

    assert decision.action == RepairAction.PROCEED
    assert decision.status == VerificationStatus.PARTIAL
    assert decision.new_warnings == 1
    assert decision.new_errors == 0
