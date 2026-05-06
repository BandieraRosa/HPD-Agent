"""Pydantic contracts shared by the code intelligence subsystem."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_workspace_relative_path(path: str) -> str:
    """Validate workspace-relative paths that use forward slashes."""
    if not path:
        raise ValueError("path must not be empty")
    if path.startswith("/"):
        raise ValueError("path must be workspace-relative")
    if "\\" in path:
        raise ValueError("path must use forward slashes")
    first_segment = path.split("/", 1)[0]
    if (
        len(first_segment) >= 2
        and first_segment[1] == ":"
        and first_segment[0].isalpha()
    ):
        raise ValueError("path must be workspace-relative")
    if any(segment in {"", ".", ".."} for segment in path.split("/")):
        raise ValueError("path must not contain empty, current, or parent segments")
    return path


class Range(BaseModel):
    """0-based half-open source range [start, end)."""

    start_line: int = Field(ge=0, description="0-based inclusive start line.")
    start_col: int = Field(ge=0, description="0-based inclusive start column.")
    end_line: int = Field(ge=0, description="0-based exclusive end line.")
    end_col: int = Field(ge=0, description="0-based exclusive end column.")

    @model_validator(mode="after")
    def validate_half_open_order(self) -> "Range":
        if (self.end_line, self.end_col) < (self.start_line, self.start_col):
            raise ValueError("range end must not be before range start")
        return self


class Location(BaseModel):
    """Workspace-relative source location."""

    path: str = Field(description="Workspace-relative path using forward slashes.")
    range: Range = Field(
        description="0-based half-open source range for this location."
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)


class SymbolKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    VARIABLE = "variable"
    IMPORT = "import"
    EXPORT = "export"
    MODULE = "module"
    NAMESPACE = "namespace"
    ENUM = "enum"
    ENUM_MEMBER = "enum_member"


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Symbol(BaseModel):
    """Indexed symbol identity, location, summary, and provenance."""

    id: str = Field(
        default="",
        description="Workspace-global stable symbol ID generated from language, path, kind, name, position, and file hash.",
    )
    name: str = Field(description="Simple symbol name, such as 'login'.")
    qualified_name: str | None = Field(
        default=None,
        description="Qualified symbol name, such as 'UserService.login'.",
    )
    kind: SymbolKind = Field(
        description="Symbol kind using lowercase English enum values."
    )
    language: str = Field(
        description="Source language, such as 'python' or 'typescript'."
    )
    path: str = Field(description="Workspace-relative path using forward slashes.")
    range: Range = Field(
        description="Full symbol range, including body when available."
    )
    selection_range: Range | None = Field(
        default=None,
        description="Range for the symbol name selection, if available.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Parent symbol ID for nested symbols, if any.",
    )
    signature: str | None = Field(
        default=None,
        description="Symbol signature text, if available.",
    )
    doc: str | None = Field(
        default=None,
        description="Docstring or leading comment text, if available.",
    )
    source: str = Field(description="Provider source that produced this symbol.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Provider confidence in this symbol, from 0.0 to 1.0.",
    )
    file_hash: str = Field(
        description="File hash captured when the symbol was extracted."
    )
    index_version: str = Field(description="Extraction/indexing rules version.")
    stale: bool = Field(
        default=False,
        description="Whether callers have marked this symbol as stale.",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)

    @model_validator(mode="after")
    def generate_stable_id(self) -> "Symbol":
        identity_name = self.qualified_name or self.name
        identity_range = self.selection_range or self.range
        raw = ":".join(
            (
                self.language,
                self.path,
                self.kind.value,
                identity_name,
                str(identity_range.start_line),
                str(identity_range.start_col),
                self.file_hash,
            )
        )
        self.id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return self


class Diagnostic(BaseModel):
    """Static analysis diagnostic reported for a source range."""

    path: str = Field(description="Workspace-relative path using forward slashes.")
    range: Range = Field(
        description="0-based half-open source range for this diagnostic."
    )
    severity: DiagnosticSeverity = Field(description="Diagnostic severity level.")
    message: str = Field(description="Human-readable diagnostic message.")
    code: str | None = Field(
        default=None, description="Provider-specific diagnostic code."
    )
    source: str = Field(description="Diagnostic provider source, such as 'pyright'.")
    fingerprint: str = Field(
        description="Stable diagnostic fingerprint for delta comparison."
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)


class ToolError(BaseModel):
    """LLM-facing tool error with English code and Chinese guidance text."""

    code: str = Field(description="Machine-readable English error code.")
    message: str = Field(description="Chinese human-readable error message.")
    hint: str | None = Field(
        default=None, description="Chinese recovery hint for the LLM."
    )


class ToolMeta(BaseModel):
    """Metadata attached to every tool result envelope."""

    elapsed_ms: int = Field(
        default=0, ge=0, description="Tool elapsed time in milliseconds."
    )
    truncated: bool = Field(
        default=False, description="Whether the result was truncated by budget."
    )
    more_available: bool = Field(
        default=False, description="Whether additional results are available."
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Provider sources used to produce the result.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Machine-readable metadata flags produced while resolving the request.",
    )


T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """Generic success/error envelope returned by code intelligence tools."""

    ok: bool = Field(description="Whether the tool call completed successfully.")
    data: T | None = Field(
        default=None, description="Typed tool result data when ok is true."
    )
    error: ToolError | None = Field(
        default=None, description="Typed tool error when ok is false."
    )
    meta: ToolMeta = Field(default_factory=ToolMeta, description="Execution metadata.")


class CodeContext(BaseModel):
    """Extracted code context around a target symbol."""

    target_symbol: Symbol = Field(description="Symbol that the context describes.")
    signature: str | None = Field(
        default=None, description="Target symbol signature, if included."
    )
    body: str | None = Field(
        default=None, description="Target symbol body text, if included."
    )
    parents: list[Symbol] = Field(
        default_factory=list,
        description="Outer symbols such as enclosing classes or modules.",
    )
    imports: list[str] = Field(
        default_factory=list, description="Relevant import lines."
    )
    nearby_symbols: list[Symbol] = Field(
        default_factory=list,
        description="Nearby symbols from the same file.",
    )
    truncated: bool = Field(
        default=False, description="Whether context was truncated by budget."
    )


class HoverInfo(BaseModel):
    """LSP-style hover information for a code target."""

    contents: str = Field(description="Hover contents shown for the target.")
    range: Range | None = Field(
        default=None, description="Source range covered by the hover, if any."
    )
    source: str | None = Field(
        default=None, description="Provider source that produced the hover."
    )


__all__ = [
    "CodeContext",
    "Diagnostic",
    "DiagnosticSeverity",
    "HoverInfo",
    "Location",
    "Range",
    "Symbol",
    "SymbolKind",
    "ToolError",
    "ToolMeta",
    "ToolResult",
]
