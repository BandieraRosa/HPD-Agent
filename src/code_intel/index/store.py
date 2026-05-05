"""Async SQLite-backed symbol index store."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Sequence
from typing import Protocol, cast

import aiosqlite

from src.code_intel.core import Range, Symbol, SymbolKind
from src.code_intel.core.models import validate_workspace_relative_path
from src.code_intel.tracing import trace_span

from .invalidation import IndexedFileMetadata

SCHEMA_VERSION = "symbol-index-schema-v1"
_DEFAULT_MAX_SEARCH_RESULTS = 100
_RowValue = str | int | float | bytes | None


class _Row(Protocol):
    def __getitem__(self, key: str | int) -> _RowValue: ...


@dataclass(frozen=True)
class SymbolHistoryEntry:
    """Historical mapping from a symbol ID to its identity tuple."""

    symbol_id: str
    language: str
    path: str
    qualified_name: str
    file_hash: str
    created_at: float
    last_seen_at: float


def default_symbol_index_path(workspace_root: str | Path) -> Path:
    """Return the default per-workspace symbol index path."""
    resolved = Path(workspace_root).expanduser().resolve(strict=False)
    workspace_key = sha256(str(resolved).encode("utf-8")).hexdigest()
    return Path.home() / ".hpagent" / "index" / workspace_key / "symbols.db"


class SymbolIndexStore:
    """Async store for file metadata, symbols, FTS search, and ID history."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        schema_version: str = SCHEMA_VERSION,
        enable_fts: bool = True,
        max_search_results: int = _DEFAULT_MAX_SEARCH_RESULTS,
    ) -> None:
        self.db_path: Path = Path(db_path).expanduser()
        self.schema_version: str = schema_version
        self.enable_fts: bool = enable_fts
        self.max_search_results: int = max(1, max_search_results)
        self.fts_available: bool = False
        self.fts_error: str | None = None
        self._connection: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "SymbolIndexStore":
        await self.initialize()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        await self.close()

    async def initialize(self) -> None:
        """Open the database, initialize pragmas, and create schema."""
        await asyncio.to_thread(self.db_path.parent.mkdir, parents=True, exist_ok=True)
        connection = await self._connect()
        await self._initialize_pragmas(connection)
        await self._create_core_schema(connection)
        await self._initialize_fts(connection)
        await connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def pragma_values(self) -> dict[str, str | int]:
        connection = await self._connect()
        journal_mode = await self._fetch_pragma_text(connection, "journal_mode")
        foreign_keys = await self._fetch_pragma_int(connection, "foreign_keys")
        synchronous = await self._fetch_pragma_int(connection, "synchronous")
        return {
            "journal_mode": journal_mode,
            "foreign_keys": foreign_keys,
            "synchronous": synchronous,
        }

    async def schema_tables(self) -> set[str]:
        connection = await self._connect()
        rows = cast(
            Sequence[_Row],
            await connection.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            ),
        )
        return {str(row[0]) for row in rows}

    async def table_columns(self, table: str) -> list[str]:
        connection = await self._connect()
        rows = cast(
            Sequence[_Row],
            await connection.execute_fetchall(f"PRAGMA table_info({table})"),
        )
        return [str(row[1]) for row in rows]

    async def store_symbols(
        self, file_metadata: CurrentFileForStore, symbols: Sequence[Symbol]
    ) -> None:
        """Replace one file's current symbols while preserving symbol ID history."""
        with trace_span(
            "code_intel.index.store_symbols",
            {"path": file_metadata.path, "language": file_metadata.language},
        ) as span:
            await self._store_symbols_untraced(file_metadata, symbols)
            span.add_metadata({"result_count": len(symbols)})

    async def _store_symbols_untraced(
        self, file_metadata: CurrentFileForStore, symbols: Sequence[Symbol]
    ) -> None:
        path = validate_workspace_relative_path(file_metadata.path)
        connection = await self._connect()
        now = time.time()
        _ = await connection.execute("BEGIN")
        try:
            _ = await connection.execute("DELETE FROM symbols WHERE path = ?", (path,))
            _ = await connection.execute(
                """
                INSERT INTO files (
                    path, language, sha256, mtime, size, indexed_at,
                    grammar_version, query_version, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    language = excluded.language,
                    sha256 = excluded.sha256,
                    mtime = excluded.mtime,
                    size = excluded.size,
                    indexed_at = excluded.indexed_at,
                    grammar_version = excluded.grammar_version,
                    query_version = excluded.query_version,
                    schema_version = excluded.schema_version
                """,
                (
                    path,
                    file_metadata.language,
                    file_metadata.sha256,
                    file_metadata.mtime,
                    file_metadata.size,
                    now,
                    file_metadata.grammar_version,
                    file_metadata.query_version,
                    file_metadata.schema_version,
                ),
            )
            for symbol in symbols:
                await self._insert_symbol(connection, symbol)
                await self._upsert_symbol_history(connection, symbol, now)
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

    async def delete_file(self, path: str) -> None:
        """Delete one file row and cascade current symbols; history is retained."""
        with trace_span("code_intel.index.delete_file", {"path": path}) as span:
            relative_path = validate_workspace_relative_path(path)
            connection = await self._connect()
            _ = await connection.execute(
                "DELETE FROM files WHERE path = ?", (relative_path,)
            )
            await connection.commit()
            span.add_metadata({"result_count": 0})

    async def get_file(self, path: str) -> IndexedFileMetadata | None:
        with trace_span("code_intel.index.get_file", {"path": path}) as span:
            relative_path = validate_workspace_relative_path(path)
            connection = await self._connect()
            row = await _fetchone(
                connection,
                """
                SELECT path, language, sha256, mtime, size, indexed_at,
                       grammar_version, query_version, schema_version
                FROM files
                WHERE path = ?
                """,
                (relative_path,),
            )
            if row is None:
                span.add_metadata({"cache_hit": False, "result_count": 0})
                return None
            span.add_metadata({"cache_hit": True, "result_count": 1})
            return _indexed_file_from_row(row)

    async def list_files(self) -> list[IndexedFileMetadata]:
        with trace_span("code_intel.index.list_files") as span:
            connection = await self._connect()
            rows = cast(
                Sequence[_Row],
                await connection.execute_fetchall("""
                    SELECT path, language, sha256, mtime, size, indexed_at,
                           grammar_version, query_version, schema_version
                    FROM files
                    ORDER BY path
                    """),
            )
            files = [_indexed_file_from_row(row) for row in rows]
            span.add_metadata({"result_count": len(files)})
            return files

    async def get_symbols(self, path: str | None = None) -> list[Symbol]:
        metadata = {"path": path} if path is not None else None
        with trace_span("code_intel.index.get_symbols", metadata) as span:
            connection = await self._connect()
            if path is None:
                rows = cast(
                    Sequence[_Row],
                    await connection.execute_fetchall(
                        f"{_SYMBOL_SELECT_SQL} ORDER BY path, start_line, start_col, name"
                    ),
                )
            else:
                relative_path = validate_workspace_relative_path(path)
                rows = cast(
                    Sequence[_Row],
                    await connection.execute_fetchall(
                        f"{_SYMBOL_SELECT_SQL} WHERE path = ? ORDER BY start_line, start_col, name",
                        (relative_path,),
                    ),
                )
            symbols = [_symbol_from_row(row) for row in rows]
            span.add_metadata(
                {"result_count": len(symbols), "cache_hit": bool(symbols)}
            )
            return symbols

    async def get_symbol_by_id(self, symbol_id: str) -> Symbol | None:
        """Return the current symbol row for a symbol ID, if it is still indexed."""
        with trace_span("code_intel.index.get_symbol_by_id") as span:
            if not symbol_id:
                span.add_metadata({"cache_hit": False, "result_count": 0})
                return None
            connection = await self._connect()
            row = await _fetchone(
                connection,
                f"{_SYMBOL_SELECT_SQL} WHERE id = ?",
                (symbol_id,),
            )
            if row is None:
                span.add_metadata({"cache_hit": False, "result_count": 0})
                return None
            span.add_metadata({"cache_hit": True, "result_count": 1})
            return _symbol_from_row(row)

    async def get_symbol_by_qualified_name(
        self,
        path: str,
        qualified_name: str,
        *,
        language: str | None = None,
    ) -> Symbol | None:
        """Return the current symbol matching a path and qualified-name identity."""
        relative_path = validate_workspace_relative_path(path)
        if not qualified_name:
            return None
        connection = await self._connect()
        language_clause = "AND language = ?" if language is not None else ""
        parameters: list[object] = [relative_path, qualified_name, qualified_name]
        if language is not None:
            parameters.append(language)
        with trace_span(
            "code_intel.index.get_symbol_by_qualified_name",
            {"path": relative_path, "language": language or ""},
        ) as span:
            row = await _fetchone(
                connection,
                f"""
                {_SYMBOL_SELECT_SQL}
                WHERE path = ?
                  AND (qualified_name = ? OR (qualified_name IS NULL AND name = ?))
                  {language_clause}
                ORDER BY stale ASC, start_line, start_col, name
                LIMIT 1
                """,
                tuple(parameters),
            )
            if row is None:
                span.add_metadata({"cache_hit": False, "result_count": 0})
                return None
            span.add_metadata({"cache_hit": True, "result_count": 1})
            return _symbol_from_row(row)

    async def search_symbols(
        self,
        query: str,
        *,
        kind: SymbolKind | None = None,
        limit: int = 20,
    ) -> list[Symbol]:
        """Search symbols with FTS when available, otherwise bounded LIKE fallback."""
        requested_limit = self._bounded_limit(limit)
        if query == "" or requested_limit <= 0:
            return []
        with trace_span("code_intel.index.search_symbols") as span:
            if self.fts_available:
                try:
                    results = await self._search_symbols_fts(
                        query, kind=kind, limit=requested_limit
                    )
                    span.add_metadata(
                        {
                            "result_count": len(results),
                            "cache_hit": True,
                            "truncated": len(results) >= requested_limit,
                        }
                    )
                    return results
                except Exception as error:
                    self.fts_available = False
                    self.fts_error = str(error)
            results = await self._search_symbols_like(
                query, kind=kind, limit=requested_limit
            )
            span.add_metadata(
                {
                    "result_count": len(results),
                    "cache_hit": False,
                    "truncated": len(results) >= requested_limit,
                }
            )
            return results

    async def history_lookup(self, symbol_id: str) -> SymbolHistoryEntry | None:
        connection = await self._connect()
        row = await _fetchone(
            connection,
            """
            SELECT symbol_id, language, path, qualified_name, file_hash, created_at, last_seen_at
            FROM symbol_id_history
            WHERE symbol_id = ?
            """,
            (symbol_id,),
        )
        with trace_span("code_intel.index.history_lookup") as span:
            if row is None:
                span.add_metadata({"cache_hit": False, "result_count": 0})
                return None
            entry = SymbolHistoryEntry(
                symbol_id=str(row["symbol_id"]),
                language=str(row["language"]),
                path=str(row["path"]),
                qualified_name=str(row["qualified_name"]),
                file_hash=str(row["file_hash"]),
                created_at=_to_float(row["created_at"]),
                last_seen_at=_to_float(row["last_seen_at"]),
            )
            span.add_metadata(
                {
                    "cache_hit": True,
                    "result_count": 1,
                    "path": entry.path,
                    "language": entry.language,
                }
            )
            return entry

    async def history_entries(self) -> list[SymbolHistoryEntry]:
        connection = await self._connect()
        rows = cast(
            Sequence[_Row],
            await connection.execute_fetchall("""
                SELECT symbol_id, language, path, qualified_name, file_hash, created_at, last_seen_at
                FROM symbol_id_history
                ORDER BY path, qualified_name, file_hash, symbol_id
                """),
        )
        with trace_span("code_intel.index.history_entries") as span:
            entries = [
                SymbolHistoryEntry(
                    symbol_id=str(row["symbol_id"]),
                    language=str(row["language"]),
                    path=str(row["path"]),
                    qualified_name=str(row["qualified_name"]),
                    file_hash=str(row["file_hash"]),
                    created_at=_to_float(row["created_at"]),
                    last_seen_at=_to_float(row["last_seen_at"]),
                )
                for row in rows
            ]
            span.add_metadata({"result_count": len(entries)})
            return entries

    async def _connect(self) -> aiosqlite.Connection:
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
        return self._connection

    async def _initialize_pragmas(self, connection: aiosqlite.Connection) -> None:
        _ = await connection.execute("PRAGMA journal_mode=WAL")
        _ = await connection.execute("PRAGMA synchronous=NORMAL")
        _ = await connection.execute("PRAGMA foreign_keys=ON")

    async def _create_core_schema(self, connection: aiosqlite.Connection) -> None:
        _ = await connection.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                indexed_at REAL NOT NULL,
                grammar_version TEXT NOT NULL,
                query_version TEXT NOT NULL,
                schema_version TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                language TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT,
                kind TEXT NOT NULL,
                parent_id TEXT REFERENCES symbols(id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
                start_line INTEGER NOT NULL,
                start_col INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                end_col INTEGER NOT NULL,
                sel_start_line INTEGER,
                sel_start_col INTEGER,
                sel_end_line INTEGER,
                sel_end_col INTEGER,
                signature TEXT,
                doc TEXT,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                file_sha256 TEXT NOT NULL,
                index_version TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_qname ON symbols(qualified_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
            CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

            CREATE TABLE IF NOT EXISTS symbol_id_history (
                symbol_id TEXT PRIMARY KEY,
                language TEXT NOT NULL,
                path TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_symbol_id_history_identity
                ON symbol_id_history(language, path, qualified_name, file_hash);
            CREATE INDEX IF NOT EXISTS idx_symbol_id_history_path_qname
                ON symbol_id_history(path, qualified_name);
            """)

    async def _initialize_fts(self, connection: aiosqlite.Connection) -> None:
        if not self.enable_fts:
            self.fts_available = False
            self.fts_error = "FTS disabled for this store"
            return
        try:
            _ = await connection.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                    name, qualified_name, signature, path,
                    content='symbols', content_rowid='rowid'
                );

                CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                    INSERT INTO symbols_fts(rowid, name, qualified_name, signature, path)
                    VALUES (new.rowid, new.name, new.qualified_name, new.signature, new.path);
                END;

                CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, signature, path)
                    VALUES('delete', old.rowid, old.name, old.qualified_name, old.signature, old.path);
                END;

                CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
                    INSERT INTO symbols_fts(symbols_fts, rowid, name, qualified_name, signature, path)
                    VALUES('delete', old.rowid, old.name, old.qualified_name, old.signature, old.path);
                    INSERT INTO symbols_fts(rowid, name, qualified_name, signature, path)
                    VALUES (new.rowid, new.name, new.qualified_name, new.signature, new.path);
                END;
                """)
            self.fts_available = True
            self.fts_error = None
        except Exception as error:
            self.fts_available = False
            self.fts_error = str(error)
            await self._drop_fts_objects(connection)

    async def _drop_fts_objects(self, connection: aiosqlite.Connection) -> None:
        try:
            _ = await connection.executescript("""
                DROP TRIGGER IF EXISTS symbols_ai;
                DROP TRIGGER IF EXISTS symbols_ad;
                DROP TRIGGER IF EXISTS symbols_au;
                DROP TABLE IF EXISTS symbols_fts;
                """)
        except Exception:
            return

    async def _insert_symbol(
        self, connection: aiosqlite.Connection, symbol: Symbol
    ) -> None:
        selection = symbol.selection_range
        _ = await connection.execute(
            """
            INSERT INTO symbols (
                id, path, language, name, qualified_name, kind, parent_id,
                start_line, start_col, end_line, end_col,
                sel_start_line, sel_start_col, sel_end_line, sel_end_col,
                signature, doc, source, confidence, file_sha256, index_version, stale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.id,
                symbol.path,
                symbol.language,
                symbol.name,
                symbol.qualified_name,
                symbol.kind.value,
                symbol.parent_id,
                symbol.range.start_line,
                symbol.range.start_col,
                symbol.range.end_line,
                symbol.range.end_col,
                selection.start_line if selection is not None else None,
                selection.start_col if selection is not None else None,
                selection.end_line if selection is not None else None,
                selection.end_col if selection is not None else None,
                symbol.signature,
                symbol.doc,
                symbol.source,
                symbol.confidence,
                symbol.file_hash,
                symbol.index_version,
                1 if symbol.stale else 0,
            ),
        )

    async def _upsert_symbol_history(
        self, connection: aiosqlite.Connection, symbol: Symbol, seen_at: float
    ) -> None:
        qualified_name = symbol.qualified_name or symbol.name
        _ = await connection.execute(
            """
            INSERT INTO symbol_id_history (
                symbol_id, language, path, qualified_name, file_hash, created_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol_id) DO UPDATE SET
                language = excluded.language,
                path = excluded.path,
                qualified_name = excluded.qualified_name,
                file_hash = excluded.file_hash,
                last_seen_at = excluded.last_seen_at
            """,
            (
                symbol.id,
                symbol.language,
                symbol.path,
                qualified_name,
                symbol.file_hash,
                seen_at,
                seen_at,
            ),
        )

    async def _search_symbols_fts(
        self,
        query: str,
        *,
        kind: SymbolKind | None,
        limit: int,
    ) -> list[Symbol]:
        connection = await self._connect()
        where_kind = "AND s.kind = ?" if kind is not None else ""
        params: list[str | int] = [_fts_query(query)]
        if kind is not None:
            params.append(kind.value)
        params.append(limit)
        rows = cast(
            Sequence[_Row],
            await connection.execute_fetchall(
                f"""
                SELECT s.id, s.path, s.language, s.name, s.qualified_name, s.kind, s.parent_id,
                       s.start_line, s.start_col, s.end_line, s.end_col,
                       s.sel_start_line, s.sel_start_col, s.sel_end_line, s.sel_end_col,
                       s.signature, s.doc, s.source, s.confidence, s.file_sha256, s.index_version, s.stale
                FROM symbols_fts
                JOIN symbols AS s ON s.rowid = symbols_fts.rowid
                WHERE symbols_fts MATCH ? {where_kind}
                ORDER BY s.qualified_name IS NULL, s.qualified_name, s.name, s.path
                LIMIT ?
                """,
                tuple(params),
            ),
        )
        return [_symbol_from_row(row) for row in rows]

    async def _search_symbols_like(
        self,
        query: str,
        *,
        kind: SymbolKind | None,
        limit: int,
    ) -> list[Symbol]:
        connection = await self._connect()
        pattern = f"%{_escape_like(query)}%"
        where_kind = "AND kind = ?" if kind is not None else ""
        escape = "\\"
        params: list[str | int] = [
            pattern,
            escape,
            pattern,
            escape,
            pattern,
            escape,
            pattern,
            escape,
        ]
        if kind is not None:
            params.append(kind.value)
        params.append(limit)
        rows = cast(
            Sequence[_Row],
            await connection.execute_fetchall(
                f"""
                {_SYMBOL_SELECT_SQL}
                WHERE (
                    name LIKE ? ESCAPE ?
                    OR COALESCE(qualified_name, '') LIKE ? ESCAPE ?
                    OR COALESCE(signature, '') LIKE ? ESCAPE ?
                    OR path LIKE ? ESCAPE ?
                ) {where_kind}
                ORDER BY qualified_name IS NULL, qualified_name, name, path
                LIMIT ?
                """,
                tuple(params),
            ),
        )
        return [_symbol_from_row(row) for row in rows]

    def _bounded_limit(self, limit: int) -> int:
        return max(0, min(limit, self.max_search_results))

    @staticmethod
    async def _fetch_pragma_text(connection: aiosqlite.Connection, name: str) -> str:
        row = await _fetchone(connection, f"PRAGMA {name}")
        if row is None:
            return ""
        return str(row[0])

    @staticmethod
    async def _fetch_pragma_int(connection: aiosqlite.Connection, name: str) -> int:
        row = await _fetchone(connection, f"PRAGMA {name}")
        if row is None:
            return 0
        return _to_int(row[0])


@dataclass(frozen=True)
class CurrentFileForStore:
    """File metadata accepted by store_symbols."""

    path: str
    language: str
    sha256: str
    mtime: float
    size: int
    grammar_version: str
    query_version: str
    schema_version: str = SCHEMA_VERSION


_SYMBOL_SELECT_SQL = """
SELECT id, path, language, name, qualified_name, kind, parent_id,
       start_line, start_col, end_line, end_col,
       sel_start_line, sel_start_col, sel_end_line, sel_end_col,
       signature, doc, source, confidence, file_sha256, index_version, stale
FROM symbols
"""


async def _fetchone(
    connection: aiosqlite.Connection,
    sql: str,
    parameters: Sequence[object] = (),
) -> _Row | None:
    async with connection.execute(sql, parameters) as cursor:
        return cast(_Row | None, await cursor.fetchone())


def _required_row_value(value: _RowValue) -> str | int | float | bytes:
    if value is None:
        raise ValueError("unexpected null value in symbol index row")
    return value


def _to_int(value: _RowValue) -> int:
    return int(_required_row_value(value))


def _to_float(value: _RowValue) -> float:
    return float(_required_row_value(value))


def _indexed_file_from_row(row: _Row) -> IndexedFileMetadata:
    return IndexedFileMetadata(
        path=str(row["path"]),
        language=str(row["language"]),
        sha256=str(row["sha256"]),
        mtime=_to_float(row["mtime"]),
        size=_to_int(row["size"]),
        indexed_at=_to_float(row["indexed_at"]),
        grammar_version=str(row["grammar_version"]),
        query_version=str(row["query_version"]),
        schema_version=str(row["schema_version"]),
    )


def _symbol_from_row(row: _Row) -> Symbol:
    selection = _range_or_none(
        row, "sel_start_line", "sel_start_col", "sel_end_line", "sel_end_col"
    )
    return Symbol(
        id=str(row["id"]),
        name=str(row["name"]),
        qualified_name=cast(str | None, row["qualified_name"]),
        kind=SymbolKind(str(row["kind"])),
        language=str(row["language"]),
        path=str(row["path"]),
        range=Range(
            start_line=_to_int(row["start_line"]),
            start_col=_to_int(row["start_col"]),
            end_line=_to_int(row["end_line"]),
            end_col=_to_int(row["end_col"]),
        ),
        selection_range=selection,
        parent_id=cast(str | None, row["parent_id"]),
        signature=cast(str | None, row["signature"]),
        doc=cast(str | None, row["doc"]),
        source=str(row["source"]),
        confidence=_to_float(row["confidence"]),
        file_hash=str(row["file_sha256"]),
        index_version=str(row["index_version"]),
        stale=bool(_to_int(row["stale"])),
    )


def _range_or_none(
    row: _Row, start_line: str, start_col: str, end_line: str, end_col: str
) -> Range | None:
    if (
        row[start_line] is None
        or row[start_col] is None
        or row[end_line] is None
        or row[end_col] is None
    ):
        return None
    return Range(
        start_line=_to_int(row[start_line]),
        start_col=_to_int(row[start_col]),
        end_line=_to_int(row[end_line]),
        end_col=_to_int(row[end_col]),
    )


def _fts_query(query: str) -> str:
    terms = _fts_terms(query)
    if not terms:
        return '""'
    return " ".join(f'"{term}"*' for term in terms)


def _fts_terms(query: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    for char in query:
        if char.isalnum() or char == "_":
            current.append(char)
            continue
        if current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return terms


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


__all__ = [
    "CurrentFileForStore",
    "SCHEMA_VERSION",
    "SymbolHistoryEntry",
    "SymbolIndexStore",
    "default_symbol_index_path",
]
