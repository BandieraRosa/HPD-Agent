"""Repair policy decisions for diagnostics deltas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.code_intel.core import DiagnosticSeverity, ToolError

from .diagnostics_delta import DiagnosticsDelta


class RepairAction(str, Enum):
    """Allowed verifier actions after diagnostics delta evaluation."""

    PROCEED = "proceed"
    REPAIR = "repair"
    RETREAT = "retreat"


class VerificationStatus(str, Enum):
    """Human- and machine-readable verification status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class RepairDecision(BaseModel):
    """Structured result returned by RepairPolicy."""

    action: RepairAction = Field(description="Next verifier action.")
    status: VerificationStatus = Field(
        description="Whether verification succeeded, was partial, or blocked."
    )
    reason: str = Field(description="Concise English reason for this decision.")
    new_errors: int = Field(ge=0, description="Number of new error diagnostics.")
    new_warnings: int = Field(ge=0, description="Number of new warning diagnostics.")
    repair_round: int = Field(ge=0, description="Current repair round.")
    max_rounds: int = Field(ge=0, description="Maximum repair rounds before retreat.")
    provider_error: ToolError | None = Field(
        default=None, description="Safe provider error for partial verification."
    )


class RepairPolicy(BaseModel):
    """Decide proceed/repair/retreat from a diagnostics delta."""

    max_rounds: int = Field(
        default=2, ge=0, description="Maximum repair rounds before retreat."
    )
    block_on_warnings: bool = Field(
        default=False, description="Whether warnings should block like errors."
    )

    def decide(
        self,
        delta: DiagnosticsDelta,
        round: int = 0,
        patch_history: list[object] | None = None,
    ) -> RepairDecision:
        """Return PROCEED, REPAIR, or RETREAT for the current delta."""
        _ = patch_history or []
        repair_round = max(0, round)
        new_errors = sum(
            1
            for item in delta.new
            if item.diagnostic.severity == DiagnosticSeverity.ERROR
        )
        new_warnings = sum(
            1
            for item in delta.new
            if item.diagnostic.severity == DiagnosticSeverity.WARNING
        )

        if delta.provider_error is not None:
            return RepairDecision(
                action=RepairAction.PROCEED,
                status=VerificationStatus.PARTIAL,
                reason="diagnostic provider unavailable; verification is partial, not successful",
                new_errors=new_errors,
                new_warnings=new_warnings,
                repair_round=repair_round,
                max_rounds=self.max_rounds,
                provider_error=delta.provider_error,
            )

        blocking_count = new_errors + (new_warnings if self.block_on_warnings else 0)
        if blocking_count > 0:
            if repair_round >= self.max_rounds:
                return RepairDecision(
                    action=RepairAction.RETREAT,
                    status=VerificationStatus.BLOCKED,
                    reason="new blocking diagnostics remain after max repair rounds",
                    new_errors=new_errors,
                    new_warnings=new_warnings,
                    repair_round=repair_round,
                    max_rounds=self.max_rounds,
                )
            return RepairDecision(
                action=RepairAction.REPAIR,
                status=VerificationStatus.BLOCKED,
                reason="new blocking diagnostics require repair",
                new_errors=new_errors,
                new_warnings=new_warnings,
                repair_round=repair_round,
                max_rounds=self.max_rounds,
            )

        if new_warnings > 0 or delta.partial:
            return RepairDecision(
                action=RepairAction.PROCEED,
                status=VerificationStatus.PARTIAL,
                reason="new warnings are non-blocking by default",
                new_errors=new_errors,
                new_warnings=new_warnings,
                repair_round=repair_round,
                max_rounds=self.max_rounds,
            )

        return RepairDecision(
            action=RepairAction.PROCEED,
            status=VerificationStatus.SUCCESS,
            reason="no new blocking diagnostics",
            new_errors=new_errors,
            new_warnings=new_warnings,
            repair_round=repair_round,
            max_rounds=self.max_rounds,
        )

    async def adecide(
        self,
        delta: DiagnosticsDelta,
        round: int = 0,
        patch_history: list[object] | None = None,
    ) -> RepairDecision:
        """Async-compatible wrapper for future workflow integration."""
        return self.decide(delta, round=round, patch_history=patch_history)


__all__ = [
    "RepairAction",
    "RepairDecision",
    "RepairPolicy",
    "VerificationStatus",
]
