"""Pydantic data envelopes returned by agent-facing code-intelligence tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.code_intel.core import Diagnostic, HoverInfo, Location, Range, Symbol, ToolError


class SearchMatch(BaseModel):
    """One code_search match normalized for LLM consumption."""

    symbol_id: str | None = Field(default=None, description="Stable symbol ID when the match is a symbol.")
    name: str = Field(description="Matched symbol name, or the query label for text matches.")
    qualified_name: str | None = Field(default=None, description="Qualified symbol name when available.")
    kind: str = Field(description="English match kind, such as function, class, method, or text.")
    path: str = Field(description="Workspace-relative path using forward slashes.")
    range: Range = Field(description="0-based half-open source range for this match.")
    snippet: str = Field(description="Short snippet or signature around the match.")
    source: str = Field(description="Provider source that produced the match.")
    confidence: float = Field(ge=0.0, le=1.0, description="Provider confidence from 0.0 to 1.0.")


class CodeSearchData(BaseModel):
    """Data payload for code_search."""

    matches: list[SearchMatch] = Field(default_factory=list, description="Normalized search matches.")


class CodeOutlineData(BaseModel):
    """Data payload for code_outline."""

    path: str = Field(description="Workspace-relative path requested by the caller.")
    language: str = Field(description="Detected language passed to the kernel.")
    symbols: list[Symbol] = Field(default_factory=list, description="File symbols, nested by parent_id.")
    line_count: int = Field(ge=0, description="Stable estimated line count derived from symbol ranges.")


class CodeContextData(BaseModel):
    """Data payload for code_context."""

    target_symbol: Symbol = Field(description="Symbol that the context describes.")
    signature: str | None = Field(default=None, description="Target signature when requested.")
    body: str | None = Field(default=None, description="Target body when requested.")
    parents: list[Symbol] = Field(default_factory=list, description="Outer symbols such as classes or modules.")
    imports: list[str] = Field(default_factory=list, description="Relevant import lines.")
    nearby_symbols: list[Symbol] = Field(default_factory=list, description="Nearby same-file symbols.")
    truncated: bool = Field(default=False, description="Whether token budgeting truncated the context.")


class CodeSemanticData(BaseModel):
    """Data payload for code_semantic."""

    operation: str = Field(description="Semantic operation that was executed.")
    locations: list[Location] = Field(default_factory=list, description="Definition or reference locations.")
    hover: HoverInfo | None = Field(default=None, description="Hover result when requested.")
    document_symbols: list[Symbol] = Field(default_factory=list, description="Document symbols when requested.")
    grouped_by_file: dict[str, list[Location]] = Field(
        default_factory=dict,
        description="Reference locations grouped by workspace-relative file path.",
    )
    more_available: bool = Field(default=False, description="Whether result limits hid additional data.")


class ChecksSkipped(BaseModel):
    """A verification check that was intentionally not run."""

    check: str = Field(description="English check name.")
    reason: str = Field(description="Chinese-first reason shown to the agent.")


class CodeVerifyData(BaseModel):
    """Data payload for code_verify."""

    ok: bool = Field(description="True when verification found no blocking errors.")
    new_diagnostics: list[Diagnostic] = Field(default_factory=list, description="Diagnostics considered new.")
    resolved_diagnostics: list[Diagnostic] = Field(default_factory=list, description="Diagnostics considered resolved.")
    unchanged_diagnostics: list[Diagnostic] = Field(default_factory=list, description="Diagnostics unchanged from baseline.")
    severity_delta: dict[str, int] = Field(default_factory=dict, description="Severity deltas keyed by English severity.")
    checks_run: list[str] = Field(default_factory=list, description="Checks that were actually executed.")
    checks_skipped: list[ChecksSkipped] = Field(default_factory=list, description="Checks intentionally skipped.")
    recommended_next_action: Literal["proceed", "repair", "abort"] = Field(
        description="Recommended next step for the agent.",
    )
    call_source: Literal["agent", "workflow"] = Field(default="agent", description="Verification call-source bucket.")
    verification_status: Literal["success", "partial", "blocked"] = Field(
        default="success",
        description="Structured status; warnings/provider issues are partial, new errors are blocked.",
    )
    baseline_key: str | None = Field(default=None, description="Verifier baseline cache key used for this call.")
    baseline_refreshed: bool = Field(default=False, description="Whether this call captured a new agent baseline.")
    provider_error: ToolError | None = Field(default=None, description="Safe provider error when verification is partial.")


__all__ = [
    "ChecksSkipped",
    "CodeContextData",
    "CodeOutlineData",
    "CodeSearchData",
    "CodeSemanticData",
    "CodeVerifyData",
    "SearchMatch",
]
