"""Deterministic fake syntax provider for kernel and tool tests."""

from __future__ import annotations

from src.code_intel.core import (
    Capability,
    ConfidenceClass,
    Location,
    ProviderHealth,
    ProviderStatus,
    Range,
    Symbol,
    SymbolKind,
)

PYTHON_FAKE_PATH = "src/fake/service.py"
TYPESCRIPT_FAKE_PATH = "src/fake/client.ts"
_FAKE_FILE_HASH = "fakehash00000000"
_INDEX_VERSION = "fake-v1"
_SUPPORTED_LANGUAGES = {"python", "typescript"}


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(
        start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col
    )


def _symbol(
    *,
    name: str,
    qualified_name: str,
    kind: SymbolKind,
    language: str,
    path: str,
    symbol_range: Range,
    selection_range: Range,
    signature: str | None,
    doc: str | None,
    confidence: float,
    parent_id: str | None = None,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language=language,
        path=path,
        range=symbol_range,
        selection_range=selection_range,
        parent_id=parent_id,
        signature=signature,
        doc=doc,
        source="fake_syntax",
        confidence=confidence,
        file_hash=_FAKE_FILE_HASH,
        index_version=_INDEX_VERSION,
    )


def fake_symbols() -> list[Symbol]:
    """Return a fresh deterministic fake symbol table."""
    service = _symbol(
        name="FakeService",
        qualified_name="FakeService",
        kind=SymbolKind.CLASS,
        language="python",
        path=PYTHON_FAKE_PATH,
        symbol_range=_range(2, 0, 13, 0),
        selection_range=_range(2, 6, 2, 17),
        signature="class FakeService:",
        doc="Deterministic service used by code intelligence tests.",
        confidence=0.86,
    )
    run = _symbol(
        name="run",
        qualified_name="FakeService.run",
        kind=SymbolKind.METHOD,
        language="python",
        path=PYTHON_FAKE_PATH,
        symbol_range=_range(5, 4, 9, 25),
        selection_range=_range(5, 8, 5, 11),
        signature="def run(self, value: str) -> str",
        doc="Return a deterministic fake response.",
        confidence=0.88,
        parent_id=service.id,
    )
    helper = _symbol(
        name="helper",
        qualified_name="helper",
        kind=SymbolKind.FUNCTION,
        language="python",
        path=PYTHON_FAKE_PATH,
        symbol_range=_range(16, 0, 18, 20),
        selection_range=_range(16, 4, 16, 10),
        signature="def helper(value: str) -> str",
        doc="Normalize a fake value.",
        confidence=0.82,
    )
    client = _symbol(
        name="fakeClient",
        qualified_name="fakeClient",
        kind=SymbolKind.FUNCTION,
        language="typescript",
        path=TYPESCRIPT_FAKE_PATH,
        symbol_range=_range(3, 0, 7, 1),
        selection_range=_range(3, 16, 3, 26),
        signature="function fakeClient(value: string): string",
        doc="TypeScript fake client used by integration tests.",
        confidence=0.8,
    )
    return [service, run, helper, client]


_TEXT_MATCHES = [
    ("FakeService", Location(path=PYTHON_FAKE_PATH, range=_range(2, 6, 2, 17))),
    (
        "return fake response",
        Location(path=PYTHON_FAKE_PATH, range=_range(8, 8, 8, 33)),
    ),
    ("helper", Location(path=PYTHON_FAKE_PATH, range=_range(16, 4, 16, 10))),
    ("fakeClient", Location(path=TYPESCRIPT_FAKE_PATH, range=_range(3, 16, 3, 26))),
]


class FakeSyntaxProvider:
    """Self-contained syntax-like provider with stable fake data."""

    def __init__(
        self,
        *,
        name: str = "fake_syntax",
        languages: set[str] | None = None,
        health: ProviderHealth | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[Capability] = {
            Capability.OUTLINE,
            Capability.SYMBOL_SEARCH,
            Capability.DOCUMENT_SYMBOLS,
            Capability.TEXT_SEARCH,
        }
        self.languages: set[str] = set(languages or _SUPPORTED_LANGUAGES)
        self._health: ProviderHealth = health or ProviderHealth(
            status=ProviderStatus.HEALTHY, health_score=1.0
        )

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        return self._health

    async def confidence_for(
        self, capability: Capability, _language: str
    ) -> ConfidenceClass:
        if capability == Capability.TEXT_SEARCH:
            return ConfidenceClass.LOW
        return ConfidenceClass.MEDIUM

    async def outline(self, path: str) -> list[Symbol]:
        return [symbol for symbol in fake_symbols() if symbol.path == path]

    async def document_symbols(self, path: str) -> list[Symbol]:
        return await self.outline(path)

    async def search_symbols(
        self,
        query: str,
        kind: SymbolKind | None = None,
        limit: int = 20,
    ) -> list[Symbol]:
        normalized_query = query.casefold()
        matches = [
            symbol
            for symbol in fake_symbols()
            if (kind is None or symbol.kind == kind)
            and (
                normalized_query in symbol.name.casefold()
                or normalized_query in (symbol.qualified_name or "").casefold()
            )
        ]
        return matches[:limit]

    async def text_search(
        self, query: str, path: str | None = None, limit: int = 20
    ) -> list[Location]:
        normalized_query = query.casefold()
        matches = [
            location
            for text, location in _TEXT_MATCHES
            if normalized_query in text.casefold()
            and (path is None or location.path == path)
        ]
        return matches[:limit]


__all__ = [
    "FakeSyntaxProvider",
    "PYTHON_FAKE_PATH",
    "TYPESCRIPT_FAKE_PATH",
    "fake_symbols",
]
