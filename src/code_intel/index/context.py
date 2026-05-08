"""Index-backed code context extraction."""

from __future__ import annotations

from pathlib import Path

from src.code_intel.core import (
    CodeContext,
    CodeTarget,
    ContextPart,
    Symbol,
    SymbolKind,
    SymbolNotFound,
    read_source_text,
)
from src.code_intel.core.models import text_for_range

from .resolver import IndexBackedTargetResolver
from .store import SymbolIndexStore

_CONTEXT_SYMBOL_KINDS = {
    SymbolKind.CLASS,
    SymbolKind.INTERFACE,
    SymbolKind.FUNCTION,
    SymbolKind.METHOD,
    SymbolKind.TYPE_ALIAS,
    SymbolKind.ENUM,
    SymbolKind.NAMESPACE,
    SymbolKind.VARIABLE,
}


class IndexBackedCodeContext:
    """Build bounded CodeContext payloads from indexed symbols and source text."""

    def __init__(
        self,
        store: SymbolIndexStore,
        workspace_root: str | Path = ".",
        *,
        resolver: IndexBackedTargetResolver | None = None,
    ) -> None:
        self.store: SymbolIndexStore = store
        self.workspace_root: Path = (
            Path(workspace_root).expanduser().resolve(strict=False)
        )
        self.resolver: IndexBackedTargetResolver = (
            resolver or IndexBackedTargetResolver(store, workspace_root)
        )

    async def extract_context(
        self,
        target: CodeTarget,
        include: set[ContextPart],
        max_tokens: int,
    ) -> tuple[CodeContext, tuple[str, ...]]:
        resolved = await self.resolver.resolve_target(target)
        if resolved.symbol is None:
            raise SymbolNotFound("target location does not map to an indexed symbol")

        symbol = resolved.symbol
        symbols = await self.store.get_symbols(symbol.path)
        symbols_by_id = {item.id: item for item in symbols}
        source = await self._read_text(symbol.path) if _needs_source(include) else None

        context = CodeContext(
            target_symbol=symbol,
            signature=symbol.signature if ContextPart.SIGNATURE in include else None,
            body=(
                text_for_range(source, symbol.range)
                if source is not None and ContextPart.BODY in include
                else None
            ),
            parents=(
                _parents_for(symbol, symbols_by_id)
                if ContextPart.PARENTS in include
                else []
            ),
            imports=(
                _imports_from_source(source)
                if source is not None and ContextPart.IMPORTS in include
                else []
            ),
            nearby_symbols=(
                _nearby_symbols(symbol, symbols)
                if ContextPart.NEARBY in include
                else []
            ),
            truncated=False,
        )
        return _apply_budget(context, max_tokens), resolved.flags

    async def _read_text(self, path: str) -> str:
        return await read_source_text(self.workspace_root, path, "context")


def _needs_source(include: set[ContextPart]) -> bool:
    return bool(include.intersection({ContextPart.BODY, ContextPart.IMPORTS}))


def _parents_for(symbol: Symbol, symbols_by_id: dict[str, Symbol]) -> list[Symbol]:
    parents: list[Symbol] = []
    seen: set[str] = {symbol.id}
    parent_id = symbol.parent_id
    while parent_id is not None and parent_id not in seen:
        seen.add(parent_id)
        parent = symbols_by_id.get(parent_id)
        if parent is None:
            break
        parents.append(parent)
        parent_id = parent.parent_id
    parents.reverse()
    return parents


def _nearby_symbols(symbol: Symbol, symbols: list[Symbol]) -> list[Symbol]:
    candidates = [
        candidate
        for candidate in symbols
        if candidate.id != symbol.id and candidate.kind in _CONTEXT_SYMBOL_KINDS
    ]
    candidates.sort(
        key=lambda candidate: (
            abs(candidate.range.start_line - symbol.range.start_line),
            candidate.range.start_line,
            candidate.range.start_col,
            candidate.name,
        )
    )
    return candidates[:12]


def _imports_from_source(source: str | None) -> list[str]:
    if source is None:
        return []
    imports: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
    return imports


def _apply_budget(context: CodeContext, max_tokens: int) -> CodeContext:
    budget = max(1, max_tokens)
    current = context
    truncated = False
    while _context_tokens(current) > budget:
        truncated = True
        if current.nearby_symbols:
            current = current.model_copy(
                update={"nearby_symbols": current.nearby_symbols[:-1]}
            )
            continue
        if current.body:
            excess_tokens = max(1, _context_tokens(current) - budget)
            new_length = max(0, len(current.body) - (excess_tokens * 4))
            current = current.model_copy(
                update={"body": current.body[:new_length].rstrip()}
            )
            continue
        if current.imports:
            current = current.model_copy(update={"imports": current.imports[:-1]})
            continue
        if current.parents:
            current = current.model_copy(update={"parents": current.parents[:-1]})
            continue
        if current.signature:
            allowed = max(0, budget * 4)
            current = current.model_copy(
                update={"signature": current.signature[:allowed].rstrip()}
            )
            continue
        break
    if truncated:
        current = current.model_copy(update={"truncated": True})
    return current


def _context_tokens(context: CodeContext) -> int:
    parts: list[str] = []
    if context.signature:
        parts.append(context.signature)
    if context.body:
        parts.append(context.body)
    parts.extend(context.imports)
    parts.extend(_symbol_text(symbol) for symbol in context.parents)
    parts.extend(_symbol_text(symbol) for symbol in context.nearby_symbols)
    return sum(_text_tokens(part) for part in parts)


def _symbol_text(symbol: Symbol) -> str:
    return symbol.signature or symbol.qualified_name or symbol.name


def _text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


__all__ = ["IndexBackedCodeContext"]
