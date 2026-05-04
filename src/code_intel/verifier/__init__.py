"""Diagnostics verifier package."""

from .diagnostics_delta import (
    BaselineCache,
    BaselineKey,
    BaselineSnapshot,
    CallSource,
    DeltaDiagnostic,
    DiagnosticsDelta,
    PatchLineEdit,
    PatchSummary,
    normalize_message,
    build_baseline_key,
    clear_baseline_cache,
    compute_delta,
    content_hash_for_paths,
    create_baseline_snapshot,
    get_cached_baseline,
    refresh_agent_baseline,
    refresh_workflow_baseline,
    stable_json_hash,
    workspace_hash_for_root,
)
from .repair_policy import RepairAction, RepairDecision, RepairPolicy, VerificationStatus

# Re-export as private name so the module still exposes _normalize_message
# for callers who imported it before the public alias existed.
_normalize_message = normalize_message

__all__ = [
    "BaselineCache",
    "BaselineKey",
    "BaselineSnapshot",
    "CallSource",
    "DeltaDiagnostic",
    "DiagnosticsDelta",
    "PatchLineEdit",
    "PatchSummary",
    "RepairAction",
    "RepairDecision",
    "RepairPolicy",
    "VerificationStatus",
    "_normalize_message",
    "normalize_message",
    "build_baseline_key",
    "clear_baseline_cache",
    "compute_delta",
    "content_hash_for_paths",
    "create_baseline_snapshot",
    "get_cached_baseline",
    "refresh_agent_baseline",
    "refresh_workflow_baseline",
    "stable_json_hash",
    "workspace_hash_for_root",
]
