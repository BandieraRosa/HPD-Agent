"""Deterministic fake semantic provider for kernel and tool tests."""

from __future__ import annotations

from src.code_intel.core import (
    Capability,
    CodeContext,
    CodeTarget,
    ConfidenceClass,
    ContextPart,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    ProviderHealth,
    ProviderStatus,
    Range,
    Symbol,
    SymbolNotFound,
)

from .syntax import PYTHON_FAKE_PATH, fake_symbols

_SUPPORTED_LANGUAGES = {"python", "typescript"}


def _range(start_line: int, start_col: int, end_line: int, end_col: int) -> Range:
    return Range(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col)


_BODY_BY_QUALIFIED_NAME = {
    "FakeService": "class FakeService:\n    def run(self, value: str) -> str:\n        return helper(value)",
    "FakeService.run": "def run(self, value: str) -> str:\n    return helper(value)",
    "helper": "def helper(value: str) -> str:\n    return f'fake:{value}'",
}

_IMPORTS_BY_PATH = {
    PYTHON_FAKE_PATH: ["from __future__ import annotations"],
}

_REFERENCES_BY_QUALIFIED_NAME = {
    "FakeService": [
        Location(path=PYTHON_FAKE_PATH, range=_range(2, 6, 2, 17)),
        Location(path=PYTHON_FAKE_PATH, range=_range(21, 10, 21, 21)),
    ],
    "FakeService.run": [
        Location(path=PYTHON_FAKE_PATH, range=_range(5, 8, 5, 11)),
        Location(path=PYTHON_FAKE_PATH, range=_range(22, 18, 22, 21)),
    ],
    "helper": [
        Location(path=PYTHON_FAKE_PATH, range=_range(8, 15, 8, 21)),
        Location(path=PYTHON_FAKE_PATH, range=_range(16, 4, 16, 10)),
    ],
}


class FakeSemanticProvider:
    """Self-contained semantic-like provider with stable fake data."""

    def __init__(
        self,
        *,
        name: str = "fake_semantic",
        languages: set[str] | None = None,
        health: ProviderHealth | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[Capability] = {
            Capability.CONTEXT_EXTRACT,
            Capability.DEFINITION,
            Capability.REFERENCES,
            Capability.HOVER,
            Capability.DIAGNOSTICS,
        }
        self.languages: set[str] = set(languages or _SUPPORTED_LANGUAGES)
        self._health: ProviderHealth = health or ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        return self._health

    async def confidence_for(self, _capability: Capability, _language: str) -> ConfidenceClass:
        return ConfidenceClass.HIGH

    async def extract_context(
        self,
        target: CodeTarget,
        include: set[ContextPart],
        max_tokens: int,
    ) -> CodeContext:
        symbol = self._resolve_symbol(target)
        qualified_name = symbol.qualified_name or symbol.name
        return CodeContext(
            target_symbol=symbol,
            signature=symbol.signature if ContextPart.SIGNATURE in include else None,
            body=_BODY_BY_QUALIFIED_NAME.get(qualified_name) if ContextPart.BODY in include else None,
            parents=self._parents_for(symbol) if ContextPart.PARENTS in include else [],
            imports=list(_IMPORTS_BY_PATH.get(symbol.path, [])) if ContextPart.IMPORTS in include else [],
            nearby_symbols=self._nearby_symbols(symbol) if ContextPart.NEARBY in include else [],
            truncated=max_tokens < 64,
        )

    async def goto_definition(self, target: CodeTarget) -> list[Location]:
        symbol = self._resolve_symbol(target)
        return [Location(path=symbol.path, range=symbol.selection_range or symbol.range)]

    async def find_references(self, target: CodeTarget) -> list[Location]:
        symbol = self._resolve_symbol(target)
        qualified_name = symbol.qualified_name or symbol.name
        return list(_REFERENCES_BY_QUALIFIED_NAME.get(qualified_name, []))

    async def hover(self, target: CodeTarget) -> HoverInfo | None:
        symbol = self._resolve_symbol(target)
        signature = symbol.signature or symbol.name
        doc = f" — {symbol.doc}" if symbol.doc else ""
        return HoverInfo(
            contents=f"({symbol.kind.value}) {signature}{doc}",
            range=symbol.selection_range,
            source=self.name,
        )

    async def diagnostics(self, path: str) -> list[Diagnostic]:
        if path != PYTHON_FAKE_PATH:
            return []
        return [
            Diagnostic(
                path=PYTHON_FAKE_PATH,
                range=_range(24, 4, 24, 15),
                severity=DiagnosticSeverity.WARNING,
                message="Fake verifier warning for deterministic tests.",
                code="fake-warning",
                source=self.name,
                fingerprint="fake-semantic-warning-1",
            )
        ]

    async def verify(self, path: str) -> list[Diagnostic]:
        return await self.diagnostics(path)

    def _resolve_symbol(self, target: CodeTarget) -> Symbol:
        symbols = fake_symbols()
        if target.symbol_id is not None:
            for symbol in symbols:
                if symbol.id == target.symbol_id:
                    return symbol

        if target.anchor is not None:
            for symbol in symbols:
                if symbol.path != target.anchor.path:
                    continue
                if target.anchor.symbol_name is not None and symbol.name == target.anchor.symbol_name:
                    return symbol
                if target.anchor.needle is not None and target.anchor.needle in (symbol.signature or symbol.name):
                    return symbol

        if target.location is not None:
            for symbol in symbols:
                if symbol.path == target.location.path:
                    return symbol

        raise SymbolNotFound("fake target not found")

    @staticmethod
    def _parents_for(symbol: Symbol) -> list[Symbol]:
        if symbol.parent_id is None:
            return []
        return [candidate for candidate in fake_symbols() if candidate.id == symbol.parent_id]

    @staticmethod
    def _nearby_symbols(symbol: Symbol) -> list[Symbol]:
        return [
            candidate
            for candidate in fake_symbols()
            if candidate.path == symbol.path and candidate.id != symbol.id
        ]


__all__ = ["FakeSemanticProvider"]
