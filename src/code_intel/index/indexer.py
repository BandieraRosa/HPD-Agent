"""Safe workspace scanner and incremental symbol indexer."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol

from pathspec import PathSpec
from pathspec.pattern import Pattern

from src.code_intel.core import IndexStale, Symbol
from src.code_intel.core.models import validate_workspace_relative_path
from src.code_intel.tracing import trace_span

from .invalidation import (
    CurrentFileMetadata,
    InvalidationAction,
    InvalidationDecision,
    InvalidationReason,
    decide_file_invalidation,
)
from .store import CurrentFileForStore, SCHEMA_VERSION, SymbolIndexStore, default_symbol_index_path

_BINARY_CHUNK_SIZE = 4096
_DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000
_MAX_IGNORE_FILE_SIZE_BYTES = 256_000
_ALWAYS_IGNORED_NAMES = {".git"}
_SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_SECRET_NAME_PARTS = {"secret", "secrets", "credential", "credentials", "token", "apikey", "api_key"}
_EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}


class SymbolExtractor(Protocol):
    """Provider-free callable used by SymbolIndexer to extract file symbols."""

    def __call__(self, workspace_root: Path, path: str, language: str) -> Sequence[Symbol] | Awaitable[Sequence[Symbol]]:
        """Return symbols for one workspace-relative path."""
        ...


@dataclass(frozen=True)
class ScanStatus:
    """Per-path status emitted by workspace scanning and indexing."""

    path: str
    status: str
    reason: str


@dataclass(frozen=True)
class WorkspaceIndexResult:
    """Summary of one index pass."""

    indexed: int = 0
    rebuilt: int = 0
    reused: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: int = 0
    rebuild_all: bool = False
    fts_available: bool = False
    statuses: list[ScanStatus] = field(default_factory=list)


@dataclass(frozen=True)
class _CandidateFile:
    path: str
    absolute_path: Path
    language: str
    sha256: str
    mtime: float
    size: int

    def current_metadata(self, *, grammar_version: str, query_version: str, schema_version: str) -> CurrentFileMetadata:
        return CurrentFileMetadata(
            path=self.path,
            language=self.language,
            sha256=self.sha256,
            mtime=self.mtime,
            size=self.size,
            grammar_version=grammar_version,
            query_version=query_version,
            schema_version=schema_version,
        )

    def store_metadata(self, *, grammar_version: str, query_version: str, schema_version: str) -> CurrentFileForStore:
        return CurrentFileForStore(
            path=self.path,
            language=self.language,
            sha256=self.sha256,
            mtime=self.mtime,
            size=self.size,
            grammar_version=grammar_version,
            query_version=query_version,
            schema_version=schema_version,
        )


class SymbolIndexer:
    """Incrementally scan a workspace and populate a SymbolIndexStore."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        extractor: SymbolExtractor,
        store: SymbolIndexStore | None = None,
        db_path: str | Path | None = None,
        grammar_version: str,
        query_version: str,
        schema_version: str = SCHEMA_VERSION,
        max_file_size_bytes: int = _DEFAULT_MAX_FILE_SIZE_BYTES,
        languages_by_extension: dict[str, str] | None = None,
    ) -> None:
        self.workspace_root: Path = Path(workspace_root).expanduser().resolve(strict=False)
        self.extractor: SymbolExtractor = extractor
        self.store: SymbolIndexStore = store or SymbolIndexStore(db_path or default_symbol_index_path(self.workspace_root))
        self.grammar_version: str = grammar_version
        self.query_version: str = query_version
        self.schema_version: str = schema_version
        self.max_file_size_bytes: int = max(1, max_file_size_bytes)
        self.languages_by_extension: dict[str, str] = dict(languages_by_extension or _EXTENSION_TO_LANGUAGE)

    async def index_workspace(self) -> WorkspaceIndexResult:
        """Scan safe candidates and rebuild only stale file entries."""
        with trace_span("code_intel.index.index_workspace") as span:
            result = await self._index_workspace_untraced()
            span.add_metadata({"result_count": result.indexed, "cache_hit": result.reused > 0})
            return result

    async def _index_workspace_untraced(self) -> WorkspaceIndexResult:
        await self.store.initialize()
        candidates, scan_statuses = await asyncio.to_thread(self._scan_candidates_sync)
        candidate_by_path = {candidate.path: candidate for candidate in candidates}
        stored_files = await self.store.list_files()
        stored_by_path = {file.path: file for file in stored_files}
        statuses: list[ScanStatus] = list(scan_statuses)
        deleted = 0

        for previous in stored_files:
            if previous.path not in candidate_by_path:
                await self.store.delete_file(previous.path)
                deleted += 1
                statuses.append(ScanStatus(path=previous.path, status="deleted", reason=InvalidationReason.DELETED.value))

        decisions: dict[str, InvalidationDecision] = {}
        rebuild_all = False
        for path, candidate in candidate_by_path.items():
            current = candidate.current_metadata(
                grammar_version=self.grammar_version,
                query_version=self.query_version,
                schema_version=self.schema_version,
            )
            decision = decide_file_invalidation(stored_by_path.get(path), current)
            decisions[path] = decision
            rebuild_all = rebuild_all or decision.action == InvalidationAction.REBUILD_ALL

        rebuilt = 0
        reused = 0
        indexed = 0
        errors = 0
        for candidate in candidates:
            decision = decisions[candidate.path]
            should_rebuild = rebuild_all or decision.action == InvalidationAction.REBUILD
            reason = decision.reason.value
            if rebuild_all and decision.action == InvalidationAction.REUSE:
                reason = InvalidationAction.REBUILD_ALL.value
            if not should_rebuild:
                reused += 1
                statuses.append(ScanStatus(path=candidate.path, status="reused", reason=reason))
                continue
            try:
                metadata = candidate.store_metadata(
                    grammar_version=self.grammar_version,
                    query_version=self.query_version,
                    schema_version=self.schema_version,
                )
                symbols = await self._extract_symbols_with_self_heal(candidate, metadata)
                await self.store.store_symbols(metadata, symbols)
                indexed += 1
                rebuilt += 1 if candidate.path in stored_by_path else 0
                statuses.append(ScanStatus(path=candidate.path, status="indexed", reason=reason))
            except Exception as error:
                errors += 1
                statuses.append(ScanStatus(path=candidate.path, status="error", reason=error.__class__.__name__))

        return WorkspaceIndexResult(
            indexed=indexed,
            rebuilt=rebuilt,
            reused=reused,
            deleted=deleted,
            skipped=sum(1 for status in statuses if status.status == "skipped"),
            errors=errors,
            rebuild_all=rebuild_all,
            fts_available=self.store.fts_available,
            statuses=statuses,
        )

    async def index_file(self, path: str) -> InvalidationDecision:
        """Lazily index or reuse one safe file by workspace-relative path."""
        with trace_span("code_intel.index.index_file", {"path": path}) as span:
            decision = await self._index_file_untraced(path)
            span.add_metadata({"cache_hit": decision.should_reuse, "result_count": 0 if decision.should_reuse else 1})
            return decision

    async def _index_file_untraced(self, path: str) -> InvalidationDecision:
        await self.store.initialize()
        relative_path = validate_workspace_relative_path(path)
        candidate, status = await asyncio.to_thread(self._candidate_for_relative_path_sync, relative_path)
        previous = await self.store.get_file(relative_path)
        if candidate is None:
            decision = decide_file_invalidation(previous, None)
            if decision.action == InvalidationAction.DELETE and previous is not None:
                await self.store.delete_file(relative_path)
            return decision
        current = candidate.current_metadata(
            grammar_version=self.grammar_version,
            query_version=self.query_version,
            schema_version=self.schema_version,
        )
        decision = decide_file_invalidation(previous, current)
        if decision.should_rebuild:
            metadata = candidate.store_metadata(
                grammar_version=self.grammar_version,
                query_version=self.query_version,
                schema_version=self.schema_version,
            )
            symbols = await self._extract_symbols_with_self_heal(candidate, metadata)
            await self.store.store_symbols(metadata, symbols)
        if status.status == "skipped" and previous is not None:
            await self.store.delete_file(relative_path)
        return decision

    async def _extract_symbols_with_self_heal(
        self,
        candidate: _CandidateFile,
        metadata: CurrentFileForStore,
    ) -> list[Symbol]:
        with trace_span(
            "code_intel.index.extract_symbols",
            {"path": candidate.path, "language": candidate.language},
        ) as span:
            try:
                symbols = await self._extract_symbols(candidate.path, candidate.language)
            except IndexStale:
                await self.store.delete_file(metadata.path)
                symbols = await self._extract_symbols(candidate.path, candidate.language)
            span.add_metadata({"result_count": len(symbols)})
            return symbols

    async def _extract_symbols(self, path: str, language: str) -> list[Symbol]:
        result = self.extractor(self.workspace_root, path, language)
        if inspect.isawaitable(result):
            result = await result
        return list(result)

    def _scan_candidates_sync(self) -> tuple[list[_CandidateFile], list[ScanStatus]]:
        spec = self._load_ignore_spec()
        candidates: list[_CandidateFile] = []
        statuses: list[ScanStatus] = []
        if not self.workspace_root.is_dir():
            return candidates, statuses

        stack = [self.workspace_root]
        while stack:
            directory = stack.pop()
            try:
                children = sorted(directory.iterdir(), key=lambda child: child.name)
            except OSError:
                continue
            directories: list[Path] = []
            for child in children:
                if child.is_symlink():
                    relative = self._lexical_relative_path(child)
                    if relative is not None:
                        statuses.append(ScanStatus(relative, "skipped", "symlink"))
                    continue
                relative_path = self._relative_path(child)
                if relative_path is None:
                    continue
                if child.is_dir():
                    if self._is_ignored(relative_path, is_dir=True, spec=spec):
                        statuses.append(ScanStatus(relative_path, "skipped", "ignored"))
                    else:
                        directories.append(child)
                    continue
                if not child.is_file():
                    continue
                candidate, status = self._candidate_for_file(child, relative_path, spec)
                if candidate is None:
                    statuses.append(status)
                else:
                    candidates.append(candidate)
            stack.extend(reversed(directories))
        return candidates, statuses

    def _candidate_for_relative_path_sync(self, path: str) -> tuple[_CandidateFile | None, ScanStatus]:
        spec = self._load_ignore_spec()
        lexical_path = self.workspace_root / path
        if lexical_path.is_symlink():
            return None, ScanStatus(path=path, status="skipped", reason="symlink")
        absolute_path = lexical_path.resolve(strict=False)
        try:
            _ = absolute_path.relative_to(self.workspace_root)
        except ValueError:
            return None, ScanStatus(path=path, status="skipped", reason="out_of_root")
        if not absolute_path.is_file():
            return None, ScanStatus(path=path, status="skipped", reason="missing")
        return self._candidate_for_file(absolute_path, path, spec)

    def _candidate_for_file(
        self,
        absolute_path: Path,
        relative_path: str,
        spec: PathSpec[Pattern],
    ) -> tuple[_CandidateFile | None, ScanStatus]:
        if self._is_ignored(relative_path, is_dir=False, spec=spec):
            return None, ScanStatus(relative_path, "skipped", "ignored")
        if self._is_secret_like(relative_path):
            return None, ScanStatus(relative_path, "skipped", "secret")
        language = self._language_for_path(relative_path)
        if language is None:
            return None, ScanStatus(relative_path, "skipped", "unsupported_language")
        try:
            stat = absolute_path.stat()
        except OSError:
            return None, ScanStatus(relative_path, "skipped", "unreadable")
        if stat.st_size > self.max_file_size_bytes:
            return None, ScanStatus(relative_path, "skipped", "oversized")
        digest, binary, oversized = self._hash_text_file(absolute_path)
        if binary:
            return None, ScanStatus(relative_path, "skipped", "binary")
        if oversized:
            return None, ScanStatus(relative_path, "skipped", "oversized")
        if digest is None:
            return None, ScanStatus(relative_path, "skipped", "unreadable")
        return (
            _CandidateFile(
                path=relative_path,
                absolute_path=absolute_path,
                language=language,
                sha256=digest,
                mtime=stat.st_mtime,
                size=stat.st_size,
            ),
            ScanStatus(relative_path, "candidate", "ok"),
        )

    def _load_ignore_spec(self) -> PathSpec[Pattern]:
        gitignore = self.workspace_root / ".gitignore"
        try:
            if not gitignore.is_file() or gitignore.stat().st_size > _MAX_IGNORE_FILE_SIZE_BYTES:
                return PathSpec.from_lines("gitignore", [])
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return PathSpec.from_lines("gitignore", [])
        return PathSpec.from_lines("gitignore", lines)

    def _relative_path(self, path: Path) -> str | None:
        try:
            relative = path.resolve(strict=False).relative_to(self.workspace_root)
        except (OSError, ValueError):
            return None
        return self._validate_relative(relative.as_posix())

    def _lexical_relative_path(self, path: Path) -> str | None:
        try:
            relative = path.relative_to(self.workspace_root)
        except ValueError:
            return None
        return self._validate_relative(relative.as_posix())

    @staticmethod
    def _validate_relative(relative_path: str) -> str | None:
        if not relative_path or relative_path == ".":
            return None
        try:
            return validate_workspace_relative_path(relative_path)
        except ValueError:
            return None

    def _is_ignored(self, relative_path: str, *, is_dir: bool, spec: PathSpec[Pattern]) -> bool:
        parts = set(PurePosixPath(relative_path).parts)
        if parts.intersection(_ALWAYS_IGNORED_NAMES):
            return True
        if spec.match_file(relative_path):
            return True
        return is_dir and spec.match_file(f"{relative_path}/")

    def _is_secret_like(self, relative_path: str) -> bool:
        path = PurePosixPath(relative_path)
        name = path.name.casefold()
        stem = path.stem.casefold()
        compact = stem.replace("-", "_")
        return (
            name in _SECRET_FILE_NAMES
            or path.suffix.casefold() in _SECRET_SUFFIXES
            or compact in _SECRET_NAME_PARTS
            or any(part in compact for part in _SECRET_NAME_PARTS)
        )

    def _language_for_path(self, relative_path: str) -> str | None:
        return self.languages_by_extension.get(PurePosixPath(relative_path).suffix.casefold())

    def _hash_text_file(self, path: Path) -> tuple[str | None, bool, bool]:
        digest = hashlib.sha256()
        total = 0
        try:
            with path.open("rb") as handle:
                first_chunk = handle.read(min(_BINARY_CHUNK_SIZE, self.max_file_size_bytes))
                if b"\0" in first_chunk:
                    return None, True, False
                digest.update(first_chunk)
                total += len(first_chunk)
                while True:
                    chunk = handle.read(_BINARY_CHUNK_SIZE)
                    if not chunk:
                        break
                    if b"\0" in chunk:
                        return None, True, False
                    total += len(chunk)
                    if total > self.max_file_size_bytes:
                        return None, False, True
                    digest.update(chunk)
        except OSError:
            return None, False, False
        return digest.hexdigest(), False, False


__all__ = [
    "ScanStatus",
    "SymbolExtractor",
    "SymbolIndexer",
    "WorkspaceIndexResult",
]
