"""Capability and provider contracts for code intelligence."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, Field

from .anchors import CodeTarget
from .models import CodeContext, Diagnostic, HoverInfo, Location, Symbol, SymbolKind


class Capability(str, Enum):
    """Routable code intelligence capabilities."""

    OUTLINE = "outline"
    SYMBOL_SEARCH = "symbol_search"
    CONTEXT_EXTRACT = "context_extract"
    DEFINITION = "definition"
    REFERENCES = "references"
    HOVER = "hover"
    DIAGNOSTICS = "diagnostics"
    DOCUMENT_SYMBOLS = "document_symbols"
    RENAME = "rename"
    TEXT_SEARCH = "text_search"


class ProviderStatus(str, Enum):
    """Machine-readable provider health state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ConfidenceClass(str, Enum):
    """Discrete routing confidence class, not a continuous score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContextPart(str, Enum):
    """Parts of context that a context provider may include."""

    SIGNATURE = "signature"
    BODY = "body"
    PARENTS = "parents"
    IMPORTS = "imports"
    NEARBY = "nearby"


class ProviderHealth(BaseModel):
    """Provider health data used by later kernel fallback decisions."""

    status: ProviderStatus = Field(description="Machine-readable provider health state.")
    health_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("health_score", "score"),
        description="Health score used as a routing tie-breaker, from 0.0 to 1.0.",
    )
    message: str | None = Field(default=None, description="Optional machine/log-facing health detail.")

    @property
    def score(self) -> float:
        """Alias for callers that use the shorter score name."""
        return self.health_score


@runtime_checkable
class Provider(Protocol):
    """Base provider contract; capability methods are declared separately."""

    name: str
    capabilities: set[Capability]
    languages: set[str]

    async def supports(self, capability: Capability, language: str) -> bool: ...

    async def health(self) -> ProviderHealth: ...


@runtime_checkable
class OutlineProvider(Protocol):
    async def outline(self, path: str) -> list[Symbol]: ...


@runtime_checkable
class SymbolSearchProvider(Protocol):
    async def search_symbols(
        self,
        query: str,
        kind: SymbolKind | None = None,
        limit: int = 20,
    ) -> list[Symbol]: ...


@runtime_checkable
class ContextExtractProvider(Protocol):
    async def extract_context(
        self,
        target: CodeTarget,
        include: set[ContextPart],
        max_tokens: int,
    ) -> CodeContext: ...


@runtime_checkable
class DefinitionProvider(Protocol):
    async def goto_definition(self, target: CodeTarget) -> list[Location]: ...


@runtime_checkable
class ReferenceProvider(Protocol):
    async def find_references(self, target: CodeTarget) -> list[Location]: ...


@runtime_checkable
class HoverProvider(Protocol):
    async def hover(self, target: CodeTarget) -> HoverInfo | None: ...


@runtime_checkable
class DiagnosticsProvider(Protocol):
    async def diagnostics(self, path: str) -> list[Diagnostic]: ...


__all__ = [
    "Capability",
    "ConfidenceClass",
    "ContextExtractProvider",
    "ContextPart",
    "DefinitionProvider",
    "DiagnosticsProvider",
    "HoverProvider",
    "OutlineProvider",
    "Provider",
    "ProviderHealth",
    "ProviderStatus",
    "ReferenceProvider",
    "SymbolSearchProvider",
]
