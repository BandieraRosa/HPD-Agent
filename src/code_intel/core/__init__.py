"""Pure data contracts and provider interfaces for code intelligence."""

from .anchors import CodeTarget, TextAnchor
from .models import (
    CodeContext,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    Range,
    Symbol,
    SymbolKind,
    ToolError,
    ToolMeta,
    ToolResult,
)

__all__ = [
    "CodeContext",
    "CodeTarget",
    "Diagnostic",
    "DiagnosticSeverity",
    "HoverInfo",
    "Location",
    "Range",
    "Symbol",
    "SymbolKind",
    "TextAnchor",
    "ToolError",
    "ToolMeta",
    "ToolResult",
]
