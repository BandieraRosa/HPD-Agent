"""Index-backed CodeTarget resolution."""

from __future__ import annotations

import asyncio
import bisect
from dataclasses import dataclass
from pathlib import Path

from src.code_intel.core import CodeTarget, Location, Range, Symbol, SymbolNotFound, TextAnchor
from src.code_intel.core.models import validate_workspace_relative_path

from .store import SymbolIndexStore

SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME = "symbol_id_recovered_via_qualified_name"
_ANCHOR_CONTEXT_CHARS = 500


@dataclass(frozen=True)
class ResolvedTarget:
    """Resolved target location plus optional indexed symbol metadata."""

    location: Location
    symbol: Symbol | None
    source: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NeedleMatch:
    start: int
    end: int


class IndexBackedTargetResolver:
    """Resolve CodeTarget values using the current symbol index and source text anchors."""

    def __init__(self, store: SymbolIndexStore, workspace_root: str | Path = ".") -> None:
        self.store: SymbolIndexStore = store
        self.workspace_root: Path = Path(workspace_root).expanduser().resolve(strict=False)

    async def resolve(self, target: CodeTarget) -> Location | None:
        """Return only the resolved Location for TargetResolver-compatible callers."""
        try:
            return (await self.resolve_target(target)).location
        except SymbolNotFound:
            return None

    async def resolve_target(self, target: CodeTarget) -> ResolvedTarget:
        """Resolve target by symbol_id, then TextAnchor, then explicit Location."""
        await self.store.initialize()

        if target.symbol_id is not None:
            resolved_symbol = await self._resolve_symbol_id(target.symbol_id)
            if resolved_symbol is not None:
                return resolved_symbol

        if target.anchor is not None:
            resolved_anchor = await self._resolve_anchor(target.anchor)
            if resolved_anchor is not None:
                return resolved_anchor

        if target.location is not None:
            symbol = await self._symbol_for_location(target.location)
            return ResolvedTarget(
                location=target.location,
                symbol=symbol,
                source="location",
            )

        raise SymbolNotFound("target did not resolve")

    async def _resolve_symbol_id(self, symbol_id: str) -> ResolvedTarget | None:
        current = await self.store.get_symbol_by_id(symbol_id)
        if current is not None:
            return ResolvedTarget(
                location=_location_for_symbol(current),
                symbol=current,
                source="symbol_id",
            )

        history = await self.store.history_lookup(symbol_id)
        if history is None:
            return None

        recovered = await self.store.get_symbol_by_qualified_name(
            history.path,
            history.qualified_name,
            language=history.language,
        )
        if recovered is None:
            return None

        return ResolvedTarget(
            location=_location_for_symbol(recovered),
            symbol=recovered,
            source="symbol_id_history",
            flags=(SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME,),
        )

    async def _resolve_anchor(self, anchor: TextAnchor) -> ResolvedTarget | None:
        path = validate_workspace_relative_path(anchor.path)
        symbols = await self.store.get_symbols(path)
        name_candidates = _anchor_name_candidates(anchor, symbols)

        if anchor.needle is None:
            return _resolve_symbol_anchor(anchor, name_candidates)

        content = await self._read_text(path)
        matches = _needle_matches(content, anchor)
        if name_candidates:
            matches = _matches_inside_symbols(content, matches, name_candidates)

        if not matches:
            return None

        if anchor.occurrence is not None:
            if anchor.occurrence >= len(matches):
                return None
            selected = matches[anchor.occurrence]
        elif len(matches) > 1:
            raise SymbolNotFound("ambiguous text anchor")
        else:
            selected = matches[0]

        selected_range = range_from_offsets(content, selected.start, selected.end)
        selected_symbol = _innermost_symbol_containing_range(
            name_candidates or symbols,
            selected_range,
        )
        return ResolvedTarget(
            location=Location(path=path, range=selected_range),
            symbol=selected_symbol,
            source="anchor",
        )

    async def _symbol_for_location(self, location: Location) -> Symbol | None:
        symbols = await self.store.get_symbols(location.path)
        return _innermost_symbol_containing_range(symbols, location.range)

    async def _read_text(self, path: str) -> str:
        relative_path = validate_workspace_relative_path(path)
        absolute_path = (self.workspace_root / relative_path).resolve(strict=False)
        try:
            _ = absolute_path.relative_to(self.workspace_root)
        except ValueError as error:
            raise SymbolNotFound("anchor path escaped workspace") from error

        def read_sync() -> str:
            return absolute_path.read_text(encoding="utf-8", errors="replace")

        try:
            return await asyncio.to_thread(read_sync)
        except OSError as error:
            raise SymbolNotFound("anchor source file is unavailable") from error


def _location_for_symbol(symbol: Symbol) -> Location:
    return Location(path=symbol.path, range=symbol.selection_range or symbol.range)


def _anchor_name_candidates(anchor: TextAnchor, symbols: list[Symbol]) -> list[Symbol]:
    if anchor.symbol_name is None:
        return symbols
    return [
        symbol
        for symbol in symbols
        if symbol.name == anchor.symbol_name or symbol.qualified_name == anchor.symbol_name
    ]


def _resolve_symbol_anchor(anchor: TextAnchor, candidates: list[Symbol]) -> ResolvedTarget | None:
    if anchor.symbol_name is None:
        return None
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda symbol: _range_sort_key(symbol.range))
    if anchor.occurrence is not None:
        if anchor.occurrence >= len(ordered):
            return None
        selected = ordered[anchor.occurrence]
    elif len(ordered) > 1:
        raise SymbolNotFound("ambiguous symbol anchor")
    else:
        selected = ordered[0]
    return ResolvedTarget(
        location=_location_for_symbol(selected),
        symbol=selected,
        source="anchor",
    )


def _needle_matches(content: str, anchor: TextAnchor) -> list[_NeedleMatch]:
    needle = anchor.needle
    if needle is None or needle == "":
        return []
    matches: list[_NeedleMatch] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index < 0:
            break
        end = index + len(needle)
        if _surrounding_matches(content, index, end, anchor):
            matches.append(_NeedleMatch(start=index, end=end))
        start = index + max(1, len(needle))
    return matches


def _surrounding_matches(content: str, start: int, end: int, anchor: TextAnchor) -> bool:
    before = content[max(0, start - _ANCHOR_CONTEXT_CHARS) : start]
    after = content[end : min(len(content), end + _ANCHOR_CONTEXT_CHARS)]
    if anchor.surrounding_before is not None and anchor.surrounding_before not in before:
        return False
    if anchor.surrounding_after is not None and anchor.surrounding_after not in after:
        return False
    return True


def _matches_inside_symbols(
    content: str,
    matches: list[_NeedleMatch],
    symbols: list[Symbol],
) -> list[_NeedleMatch]:
    if not matches:
        return []
    return [
        match
        for match in matches
        if _innermost_symbol_containing_range(symbols, range_from_offsets(content, match.start, match.end)) is not None
    ]


def _innermost_symbol_containing_range(symbols: list[Symbol], target_range: Range) -> Symbol | None:
    candidates = [symbol for symbol in symbols if range_contains(symbol.range, target_range)]
    if not candidates:
        return None
    return min(candidates, key=lambda symbol: _range_size(symbol.range))


def range_contains(outer: Range, inner: Range) -> bool:
    return (outer.start_line, outer.start_col) <= (inner.start_line, inner.start_col) and (
        inner.end_line,
        inner.end_col,
    ) <= (outer.end_line, outer.end_col)


def text_for_range(content: str, source_range: Range) -> str:
    start, end = offsets_for_range(content, source_range)
    return content[start:end]


def offsets_for_range(content: str, source_range: Range) -> tuple[int, int]:
    line_starts = _line_starts(content)
    return (
        _position_to_offset(content, line_starts, source_range.start_line, source_range.start_col),
        _position_to_offset(content, line_starts, source_range.end_line, source_range.end_col),
    )


def range_from_offsets(content: str, start: int, end: int) -> Range:
    line_starts = _line_starts(content)
    start_line, start_col = _offset_to_position(line_starts, start)
    end_line, end_col = _offset_to_position(line_starts, end)
    return Range(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col)


def _line_starts(content: str) -> list[int]:
    starts = [0]
    for index, character in enumerate(content):
        if character == "\n":
            starts.append(index + 1)
    return starts


def _position_to_offset(content: str, line_starts: list[int], line: int, column: int) -> int:
    if line >= len(line_starts):
        return len(content)
    line_start = line_starts[line]
    line_end = line_starts[line + 1] if line + 1 < len(line_starts) else len(content)
    return min(line_start + column, line_end)


def _offset_to_position(line_starts: list[int], offset: int) -> tuple[int, int]:
    line = max(0, bisect.bisect_right(line_starts, offset) - 1)
    return line, offset - line_starts[line]


def _range_size(source_range: Range) -> tuple[int, int]:
    return (
        source_range.end_line - source_range.start_line,
        source_range.end_col - source_range.start_col,
    )


def _range_sort_key(source_range: Range) -> tuple[int, int, int, int]:
    return (source_range.start_line, source_range.start_col, source_range.end_line, source_range.end_col)


__all__ = [
    "IndexBackedTargetResolver",
    "ResolvedTarget",
    "SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME",
    "offsets_for_range",
    "range_contains",
    "range_from_offsets",
    "text_for_range",
]
