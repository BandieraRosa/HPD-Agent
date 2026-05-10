"""Provider-free diagnostics baseline and delta computation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator

from src.code_intel.core import (
    DIAGNOSTIC_SEVERITIES,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Symbol,
    ToolError,
)
from src.code_intel.core.models import (
    range_contains,
    range_size,
    validate_workspace_relative_path,
)
from src.code_intel.tracing import normalize_diagnostic_message, trace_span

CallSource = Literal["workflow", "agent"]
_DELETED_ANCHOR = "<deleted>"

# Public alias so package __init__.py can import without reportPrivateUsage.
normalize_message = normalize_diagnostic_message
_normalize_message = normalize_diagnostic_message


def _normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({validate_workspace_relative_path(path) for path in paths}))


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        sorted_keys = sorted(mapping, key=str)
        return {str(k): _jsonable(mapping[k]) for k in sorted_keys}
    if isinstance(value, list):
        lst = cast(list[object], value)
        return [_jsonable(item) for item in lst]
    if isinstance(value, tuple):
        tup = cast(tuple[object, ...], value)
        return [_jsonable(item) for item in tup]
    if isinstance(value, set):
        st = cast(set[object], value)
        return sorted(
            (_jsonable(item) for item in st),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, frozenset):
        return sorted(
            (_jsonable(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)


def stable_json_hash(value: object) -> str:
    """Return a deterministic SHA-256 hash for JSON-like configuration values."""
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workspace_hash_for_root(workspace_root: str | Path) -> str:
    """Hash a workspace root path without scanning the workspace."""
    root = Path(workspace_root).expanduser().resolve(strict=False)
    return hashlib.sha256(root.as_posix().encode("utf-8")).hexdigest()


def content_hash_for_paths(
    workspace_root: str | Path, relevant_paths: Iterable[str]
) -> str:
    """Hash relevant path contents for baseline identity.

    Missing or escaped files are represented explicitly so tests and fake-provider
    paths stay deterministic without requiring real source files.
    """
    root = Path(workspace_root).expanduser().resolve(strict=False)
    digest = hashlib.sha256()
    for relative_path in _normalize_paths(relevant_paths):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        absolute_path = (root / relative_path).resolve(strict=False)
        try:
            _ = absolute_path.relative_to(root)
        except ValueError:
            digest.update(b"<escaped>")
            continue
        try:
            if absolute_path.is_file():
                digest.update(absolute_path.read_bytes())
            else:
                digest.update(b"<missing>")
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()


class PatchLineEdit(BaseModel):
    """Line-count summary for one patch hunk using 0-based line numbers."""

    path: str = Field(description="Workspace-relative path touched by the edit.")
    start_line: int = Field(
        ge=0, description="0-based original start line for the edit."
    )
    old_line_count: int = Field(
        ge=0, description="Number of original lines replaced or deleted."
    )
    new_line_count: int = Field(
        ge=0, description="Number of replacement or inserted lines."
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)

    @property
    def line_delta(self) -> int:
        return self.new_line_count - self.old_line_count


class PatchSummary(BaseModel):
    """Compact patch summary used to account for diagnostic line drift."""

    edits: list[PatchLineEdit] = Field(
        default_factory=list,
        description="Patch line edits in original file coordinates.",
    )

    def shifted_line(self, path: str, original_line: int) -> int:
        """Return the expected post-patch line for an original diagnostic line."""
        relative_path = validate_workspace_relative_path(path)
        shifted = original_line
        for edit in sorted(
            (item for item in self.edits if item.path == relative_path),
            key=lambda item: item.start_line,
        ):
            original_edit_end = edit.start_line + edit.old_line_count
            if original_line < edit.start_line:
                continue
            if original_line >= original_edit_end:
                shifted += edit.line_delta
                continue
            offset = original_line - edit.start_line
            if edit.new_line_count == 0:
                shifted = edit.start_line
            else:
                shifted = edit.start_line + min(offset, edit.new_line_count - 1)
        return max(0, shifted)


class BaselineKey(BaseModel):
    """Cache key for an isolated diagnostics baseline bucket."""

    workspace_hash: str = Field(description="Hash of the workspace identity.")
    relevant_paths: tuple[str, ...] = Field(
        description="Sorted workspace-relative paths in this baseline scope."
    )
    content_hash: str = Field(
        description="Hash of the relevant path contents at baseline capture time."
    )
    provider_id: str = Field(
        description="Stable provider identity label used for diagnostics."
    )
    provider_config_hash: str = Field(
        description="Hash of provider configuration that can affect diagnostics."
    )
    call_source: CallSource = Field(
        description="Caller bucket; workflow and agent baselines are isolated."
    )

    @field_validator("relevant_paths", mode="before")
    @classmethod
    def normalize_relevant_paths(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return _normalize_paths([value])
        if isinstance(value, Iterable):
            return _normalize_paths(str(item) for item in value)
        raise TypeError("relevant_paths must be iterable")

    def cache_key(self) -> str:
        """Return the full cache key, including content hash and call source."""
        return stable_json_hash(
            {
                "workspace_hash": self.workspace_hash,
                "relevant_paths": list(self.relevant_paths),
                "content_hash": self.content_hash,
                "provider_id": self.provider_id,
                "provider_config_hash": self.provider_config_hash,
                "call_source": self.call_source,
            }
        )

    def bucket_key(self) -> str:
        """Return the latest-baseline bucket key, excluding only content hash."""
        return stable_json_hash(
            {
                "workspace_hash": self.workspace_hash,
                "relevant_paths": list(self.relevant_paths),
                "provider_id": self.provider_id,
                "provider_config_hash": self.provider_config_hash,
                "call_source": self.call_source,
            }
        )


def build_baseline_key(
    *,
    workspace_root: str | Path,
    relevant_paths: Iterable[str],
    provider_id: str,
    provider_config_hash: str,
    call_source: CallSource,
    content_hash: str | None = None,
) -> BaselineKey:
    """Build a baseline key with sorted paths and deterministic workspace/content hashes."""
    paths = _normalize_paths(relevant_paths)
    return BaselineKey(
        workspace_hash=workspace_hash_for_root(workspace_root),
        relevant_paths=paths,
        content_hash=content_hash or content_hash_for_paths(workspace_root, paths),
        provider_id=provider_id,
        provider_config_hash=provider_config_hash,
        call_source=call_source,
    )


class BaselineSnapshot(BaseModel):
    """Diagnostics plus outline snapshot captured for semantic anchoring."""

    key: BaselineKey = Field(description="Isolated baseline cache key.")
    diagnostics: list[Diagnostic] = Field(
        default_factory=list, description="Diagnostics captured for this baseline."
    )
    symbols_by_path: dict[str, list[Symbol]] = Field(
        default_factory=dict,
        description="Baseline document symbols used to resolve semantic anchors.",
    )


class BaselineCache:
    """In-memory baseline cache with full keys and isolated latest buckets."""

    def __init__(self) -> None:
        self._snapshots: dict[str, BaselineSnapshot] = {}
        self._latest_by_bucket: dict[str, str] = {}

    def refresh(self, snapshot: BaselineSnapshot) -> BaselineSnapshot:
        key = snapshot.key.cache_key()
        self._snapshots[key] = snapshot
        self._latest_by_bucket[snapshot.key.bucket_key()] = key
        return snapshot

    def get_exact(self, key: BaselineKey) -> BaselineSnapshot | None:
        return self._snapshots.get(key.cache_key())

    def get_latest(self, key: BaselineKey) -> BaselineSnapshot | None:
        latest_key = self._latest_by_bucket.get(key.bucket_key())
        if latest_key is None:
            return None
        return self._snapshots.get(latest_key)

    def clear(self) -> None:
        self._snapshots.clear()
        self._latest_by_bucket.clear()

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)


_BASELINE_CACHE = BaselineCache()


def create_baseline_snapshot(
    *,
    key: BaselineKey,
    diagnostics: Iterable[Diagnostic],
    symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
) -> BaselineSnapshot:
    """Create a baseline snapshot without writing it to the cache."""
    return BaselineSnapshot(
        key=key,
        diagnostics=list(diagnostics),
        symbols_by_path={
            path: list(symbols) for path, symbols in (symbols_by_path or {}).items()
        },
    )


def refresh_workflow_baseline(
    *,
    key: BaselineKey,
    diagnostics: Iterable[Diagnostic],
    symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
    cache: BaselineCache | None = None,
) -> BaselineSnapshot:
    """Explicitly refresh a workflow baseline for future patch-gate integration."""
    workflow_key = key.model_copy(update={"call_source": "workflow"})
    snapshot = create_baseline_snapshot(
        key=workflow_key, diagnostics=diagnostics, symbols_by_path=symbols_by_path
    )
    return (cache or _BASELINE_CACHE).refresh(snapshot)


def refresh_agent_baseline(
    *,
    key: BaselineKey,
    diagnostics: Iterable[Diagnostic],
    symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
    cache: BaselineCache | None = None,
) -> BaselineSnapshot:
    """Refresh an agent baseline through the agent-call-source API."""
    agent_key = key.model_copy(update={"call_source": "agent"})
    snapshot = create_baseline_snapshot(
        key=agent_key, diagnostics=diagnostics, symbols_by_path=symbols_by_path
    )
    return (cache or _BASELINE_CACHE).refresh(snapshot)


def get_cached_baseline(
    key: BaselineKey, *, exact: bool = False, cache: BaselineCache | None = None
) -> BaselineSnapshot | None:
    """Read a baseline without crossing call-source buckets."""
    selected_cache = cache or _BASELINE_CACHE
    if exact:
        return selected_cache.get_exact(key)
    return selected_cache.get_latest(key)


def clear_baseline_cache() -> None:
    """Clear the process-local verifier baseline cache."""
    _BASELINE_CACHE.clear()


class DeltaDiagnostic(BaseModel):
    """A diagnostic decorated with the stable fields used for delta matching."""

    diagnostic: Diagnostic = Field(description="Original diagnostic model.")
    normalized_message: str = Field(
        description="Diagnostic message after fixed-rule normalization."
    )
    semantic_anchor: str = Field(
        description="Enclosing symbol anchor, shifted-line fallback, or <deleted>."
    )
    drifted_start_line: int = Field(
        ge=0, description="Baseline line after applying PatchSummary drift."
    )
    fingerprint: str = Field(
        description="Stable fingerprint over path/code/normalized message/semantic anchor."
    )


class DiagnosticsDelta(BaseModel):
    """Three-way diagnostics delta between a baseline and post-patch result."""

    new: list[DeltaDiagnostic] = Field(
        default_factory=list, description="Diagnostics present only after the patch."
    )
    resolved: list[DeltaDiagnostic] = Field(
        default_factory=list, description="Baseline diagnostics absent after the patch."
    )
    unchanged: list[DeltaDiagnostic] = Field(
        default_factory=list, description="Diagnostics present before and after."
    )
    severity_delta: dict[str, int] = Field(
        default_factory=lambda: {severity: 0 for severity in DIAGNOSTIC_SEVERITIES}
    )
    partial: bool = Field(
        default=False,
        description="True when verification is usable but incomplete/non-blocking.",
    )
    provider_error: ToolError | None = Field(
        default=None,
        description="Safe provider error when diagnostics are unavailable.",
    )

    @property
    def new_diagnostics(self) -> list[Diagnostic]:
        return [item.diagnostic for item in self.new]

    @property
    def resolved_diagnostics(self) -> list[Diagnostic]:
        return [item.diagnostic for item in self.resolved]

    @property
    def unchanged_diagnostics(self) -> list[Diagnostic]:
        return [item.diagnostic for item in self.unchanged]

    @property
    def has_new_errors(self) -> bool:
        return any(
            item.diagnostic.severity == DiagnosticSeverity.ERROR for item in self.new
        )

    @property
    def has_new_warnings(self) -> bool:
        return any(
            item.diagnostic.severity == DiagnosticSeverity.WARNING for item in self.new
        )

    @classmethod
    def provider_partial(cls, error: ToolError) -> "DiagnosticsDelta":
        return cls(partial=True, provider_error=error)


def compute_delta(
    baseline: BaselineSnapshot | list[Diagnostic],
    after: list[Diagnostic],
    patch_summary: PatchSummary | None = None,
    *,
    baseline_symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
    after_symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
) -> DiagnosticsDelta:
    """Compute a semantic diagnostics delta without raw line/message comparison."""
    with trace_span(
        "code_intel.verifier.compute_delta",
        _diagnostic_trace_input_metadata(baseline, after),
    ) as span:
        delta = _compute_delta_untraced(
            baseline,
            after,
            patch_summary,
            baseline_symbols_by_path=baseline_symbols_by_path,
            after_symbols_by_path=after_symbols_by_path,
        )
        span.add_metadata(_delta_trace_metadata(delta))
        return delta


def _compute_delta_untraced(
    baseline: BaselineSnapshot | list[Diagnostic],
    after: list[Diagnostic],
    patch_summary: PatchSummary | None = None,
    *,
    baseline_symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
    after_symbols_by_path: Mapping[str, Iterable[Symbol]] | None = None,
) -> DiagnosticsDelta:
    summary = patch_summary or PatchSummary()
    if isinstance(baseline, BaselineSnapshot):
        baseline_diagnostics = baseline.diagnostics
        baseline_symbols = _symbol_map(
            baseline_symbols_by_path or baseline.symbols_by_path
        )
    else:
        baseline_diagnostics = list(baseline)
        baseline_symbols = _symbol_map(baseline_symbols_by_path or {})

    after_symbols = _symbol_map(after_symbols_by_path or {})
    after_symbol_names = (
        _symbol_names_by_path(after_symbols)
        if after_symbols_by_path is not None
        else None
    )

    baseline_items = [
        _anchor_diagnostic(
            diagnostic,
            baseline_symbols,
            summary,
            after_symbol_names=after_symbol_names,
            baseline_side=True,
        )
        for diagnostic in baseline_diagnostics
    ]
    after_items = [
        _anchor_diagnostic(
            diagnostic,
            after_symbols,
            PatchSummary(),
            after_symbol_names=None,
            baseline_side=False,
        )
        for diagnostic in after
    ]

    after_by_fingerprint: dict[str, list[DeltaDiagnostic]] = {}
    for item in after_items:
        after_by_fingerprint.setdefault(item.fingerprint, []).append(item)

    resolved: list[DeltaDiagnostic] = []
    unchanged: list[DeltaDiagnostic] = []
    for baseline_item in baseline_items:
        if baseline_item.semantic_anchor == _DELETED_ANCHOR:
            resolved.append(baseline_item)
            continue
        matches = after_by_fingerprint.get(baseline_item.fingerprint, [])
        if matches:
            unchanged.append(matches.pop(0))
            if not matches:
                _ = after_by_fingerprint.pop(baseline_item.fingerprint, None)
        else:
            resolved.append(baseline_item)

    new = [item for matches in after_by_fingerprint.values() for item in matches]
    severity_delta = _severity_delta(new, resolved)
    partial = any(item.diagnostic.severity != DiagnosticSeverity.ERROR for item in new)
    return DiagnosticsDelta(
        new=new,
        resolved=resolved,
        unchanged=unchanged,
        severity_delta=severity_delta,
        partial=partial,
    )


def _diagnostic_trace_input_metadata(
    baseline: BaselineSnapshot | list[Diagnostic],
    after: list[Diagnostic],
) -> dict[str, object]:
    diagnostics = list(
        baseline.diagnostics if isinstance(baseline, BaselineSnapshot) else baseline
    ) + list(after)
    return {
        "paths": [diagnostic.path for diagnostic in diagnostics],
        "diagnostic_messages": [diagnostic.message for diagnostic in diagnostics],
    }


def _delta_trace_metadata(delta: DiagnosticsDelta) -> dict[str, object]:
    diagnostics = (
        delta.new_diagnostics + delta.resolved_diagnostics + delta.unchanged_diagnostics
    )
    metadata: dict[str, object] = {
        "result_count": len(diagnostics),
        "truncated": delta.partial,
        "paths": [diagnostic.path for diagnostic in diagnostics],
        "diagnostic_templates": [
            item.normalized_message
            for item in delta.new + delta.resolved + delta.unchanged
        ],
    }
    if delta.provider_error is not None:
        metadata["error"] = delta.provider_error
    return metadata


def _symbol_map(
    symbols_by_path: Mapping[str, Iterable[Symbol]],
) -> dict[str, list[Symbol]]:
    return {
        validate_workspace_relative_path(path): list(symbols)
        for path, symbols in symbols_by_path.items()
    }


def _symbol_names_by_path(
    symbols_by_path: Mapping[str, list[Symbol]],
) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for path, symbols in symbols_by_path.items():
        path_names: set[str] = set()
        for symbol in symbols:
            path_names.add(symbol.name)
            if symbol.qualified_name is not None:
                path_names.add(symbol.qualified_name)
        names[path] = path_names
    return names


def _anchor_diagnostic(
    diagnostic: Diagnostic,
    symbols_by_path: Mapping[str, list[Symbol]],
    patch_summary: PatchSummary,
    *,
    after_symbol_names: Mapping[str, set[str]] | None,
    baseline_side: bool,
) -> DeltaDiagnostic:
    drifted_line = patch_summary.shifted_line(
        diagnostic.path, diagnostic.range.start_line
    )
    anchor, symbol_name = _semantic_anchor(
        symbols_by_path.get(diagnostic.path, []), diagnostic.path, diagnostic.range
    )
    if baseline_side and symbol_name is not None and after_symbol_names is not None:
        if symbol_name not in after_symbol_names.get(diagnostic.path, set()):
            anchor = _DELETED_ANCHOR
    semantic_anchor = anchor or f"<line:{drifted_line}>"
    normalized_message = _normalize_message(diagnostic.message)
    fingerprint = _diagnostic_fingerprint(
        diagnostic, normalized_message, semantic_anchor
    )
    return DeltaDiagnostic(
        diagnostic=diagnostic,
        normalized_message=normalized_message,
        semantic_anchor=semantic_anchor,
        drifted_start_line=drifted_line,
        fingerprint=fingerprint,
    )


def _semantic_anchor(
    symbols: list[Symbol], path: str, source_range: Range
) -> tuple[str | None, str | None]:
    candidates = [
        symbol
        for symbol in symbols
        if symbol.path == path and range_contains(symbol.range, source_range)
    ]
    if not candidates:
        return None, None
    selected = min(candidates, key=lambda symbol: range_size(symbol.range))
    identity = selected.qualified_name or selected.name
    return identity, identity


def _diagnostic_fingerprint(
    diagnostic: Diagnostic, normalized_message: str, semantic_anchor: str
) -> str:
    return stable_json_hash(
        {
            "path": diagnostic.path,
            "code": diagnostic.code or "",
            "message_normalized": normalized_message,
            "semantic_anchor": semantic_anchor,
        }
    )


def _severity_delta(
    new: list[DeltaDiagnostic], resolved: list[DeltaDiagnostic]
) -> dict[str, int]:
    delta = {severity: 0 for severity in DIAGNOSTIC_SEVERITIES}
    for item in new:
        delta[item.diagnostic.severity.value] += 1
    for item in resolved:
        delta[item.diagnostic.severity.value] -= 1
    return delta


__all__ = [
    "BaselineCache",
    "BaselineKey",
    "BaselineSnapshot",
    "CallSource",
    "DeltaDiagnostic",
    "DiagnosticsDelta",
    "PatchLineEdit",
    "PatchSummary",
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
    "_normalize_message",
    "normalize_message",
]
