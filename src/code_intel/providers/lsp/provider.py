"""Kernel-routed semantic provider backed by LSP clients."""

from __future__ import annotations

import asyncio
import contextvars
import re
import stat
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar, cast, runtime_checkable

from lsprotocol import types as lsp_types

from src.code_intel.core import (
    Capability,
    CodeTarget,
    ConfidenceClass,
    Diagnostic,
    HoverInfo,
    Location,
    LSPTimeout,
    ProviderHealth,
    ProviderStatus,
    ProviderUnavailable,
    Range,
    Symbol,
)
from src.code_intel.core.models import validate_workspace_relative_path

from .manager import LSPManager, LSPServerHandle
from .registry import LanguageServerSpec

T = TypeVar("T")

_LANGUAGE_BY_EXTENSION = {
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
_ROUTED_CAPABILITIES = {
    Capability.DEFINITION,
    Capability.REFERENCES,
    Capability.HOVER,
    Capability.DIAGNOSTICS,
    Capability.DOCUMENT_SYMBOLS,
}
_NO_ROUTED_CAPABILITIES_MESSAGE = "语言服务器未提供可用的代码智能能力。"
_DIAGNOSTICS_READ_FAILURE_MESSAGE = "无法读取文件，无法同步 LSP 诊断。"
_DEFAULT_MAX_SYNC_FILE_SIZE_BYTES = 1_000_000
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DECLARATION_IDENTIFIER_PATTERNS = (
    re.compile(r"\basync\s+def\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bfrom\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"\bimport\s+([A-Za-z_][A-Za-z0-9_]*)"),
)
_SEMANTIC_POSITION_KEYWORDS = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "case",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "match",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


class _HealthState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    PERMANENTLY_UNHEALTHY = "permanently_unhealthy"


@runtime_checkable
class _DiagnosticsWaitClient(Protocol):
    def diagnostics_version(self, path: str) -> int: ...

    async def wait_for_diagnostics(
        self,
        path: str,
        *,
        timeout: float,
        after_version: int | None = None,
        document_version: int | None = None,
    ) -> bool: ...


@dataclass(frozen=True)
class _DocumentSnapshot:
    resolved_path: Path
    mtime_ns: int
    size: int


@dataclass(frozen=True)
class _WorkspaceDocument:
    content: str
    snapshot: _DocumentSnapshot


@dataclass(frozen=True)
class _PendingDiagnosticsSync:
    document_version: int
    after_publish_version: int | None


class _ProviderLSPClient(Protocol):
    async def goto_definition(
        self, path: str, *, line: int, character: int
    ) -> list[Location]: ...

    async def find_references(
        self,
        path: str,
        *,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[Location]: ...

    async def hover(
        self, path: str, *, line: int, character: int
    ) -> HoverInfo | None: ...

    async def document_symbols(self, path: str) -> list[Symbol]: ...

    async def diagnostics(self, path: str) -> list[Diagnostic]: ...

    async def did_open(
        self, path: str, *, language_id: str, text: str, version: int
    ) -> None: ...

    async def did_change(self, path: str, *, text: str, version: int) -> None: ...

    async def did_save(self, path: str, *, text: str | None = None) -> None: ...

    async def is_running(self) -> bool: ...


class LSPProvider:
    """Dynamic semantic provider that exposes only server-negotiated LSP capabilities."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        *,
        manager: LSPManager | None = None,
        specs: Iterable[LanguageServerSpec] | None = None,
        name: str = "lsp",
        languages: set[str] | None = None,
        max_sync_file_size_bytes: int = _DEFAULT_MAX_SYNC_FILE_SIZE_BYTES,
    ) -> None:
        self.name: str = name
        self.manager: LSPManager = manager or LSPManager(workspace_root, specs=specs)
        self.languages: set[str] = set(languages or self.manager.languages)
        self.capabilities: set[Capability] = set()
        self._capabilities_by_language: dict[str, set[Capability]] = {}
        self._health_states: dict[str, _HealthState] = {}
        self._health_messages: dict[str, str] = {}
        self._restart_failures: dict[str, int] = {}
        self._diagnostics_cache: dict[str, list[Diagnostic]] = {}
        self._diagnostics_pending_syncs: dict[str, _PendingDiagnosticsSync] = {}
        self._document_versions: dict[str, int] = {}
        self._document_snapshots: dict[str, _DocumentSnapshot] = {}
        self._document_locks: dict[str, asyncio.Lock] = {}
        self._open_documents: set[str] = set()
        self.max_sync_file_size_bytes: int = max(1, max_sync_file_size_bytes)
        self._language_context: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar(
                f"lsp_provider_language_{id(self)}",
                default=None,
            )
        )

    async def supports(self, capability: Capability, language: str) -> bool:
        _ = self._language_context.set(language)
        if language not in self.languages or capability not in _ROUTED_CAPABILITIES:
            return False
        state = self._health_states.get(language, _HealthState.HEALTHY)
        if state in {_HealthState.UNHEALTHY, _HealthState.PERMANENTLY_UNHEALTHY}:
            cached = self._capabilities_by_language.get(language, set())
            if not cached:
                raise ProviderUnavailable(
                    self._health_messages.get(language, "语言服务器不可用。")
                )
            return capability in cached
        capabilities = await self._ensure_capabilities(language)
        return capability in capabilities

    async def health(self) -> ProviderHealth:
        language = self._language_context.get()
        if language is not None and language in self.languages:
            return await self.check_health(language)
        if not self._health_states:
            return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)
        if any(
            state == _HealthState.PERMANENTLY_UNHEALTHY
            for state in self._health_states.values()
        ):
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                health_score=0.0,
                message="permanently_unhealthy",
            )
        if any(
            state == _HealthState.UNHEALTHY for state in self._health_states.values()
        ):
            return ProviderHealth(
                status=ProviderStatus.DEGRADED, health_score=0.4, message="unhealthy"
            )
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def confidence_for(
        self, capability: Capability, language: str
    ) -> ConfidenceClass:
        if (
            language in self.languages
            and capability in self._capabilities_by_language.get(language, set())
        ):
            return ConfidenceClass.HIGH
        return ConfidenceClass.LOW

    async def goto_definition(self, target: CodeTarget) -> list[Location]:
        location = self._target_location(target)
        language = self._language_for_path(location.path)
        _ = self._language_context.set(language)
        client = await self._client_for_language(language, Capability.DEFINITION)
        line, character = await self._prepare_semantic_request(
            location, language, client
        )
        return await self._with_unhealthy_on_unavailable(
            language,
            client.goto_definition(
                location.path,
                line=line,
                character=character,
            ),
        )

    async def find_references(self, target: CodeTarget) -> list[Location]:
        location = self._target_location(target)
        language = self._language_for_path(location.path)
        _ = self._language_context.set(language)
        client = await self._client_for_language(language, Capability.REFERENCES)
        line, character = await self._prepare_semantic_request(
            location, language, client
        )
        return await self._with_unhealthy_on_unavailable(
            language,
            client.find_references(
                location.path,
                line=line,
                character=character,
                include_declaration=True,
            ),
        )

    async def hover(self, target: CodeTarget) -> HoverInfo | None:
        location = self._target_location(target)
        language = self._language_for_path(location.path)
        _ = self._language_context.set(language)
        client = await self._client_for_language(language, Capability.HOVER)
        line, character = await self._prepare_semantic_request(
            location, language, client
        )
        return await self._with_unhealthy_on_unavailable(
            language,
            client.hover(
                location.path,
                line=line,
                character=character,
            ),
        )

    async def document_symbols(self, path: str) -> list[Symbol]:
        relative_path = self._validate_path(path)
        language = self._language_for_path(relative_path)
        _ = self._language_context.set(language)
        client = await self._client_for_language(language, Capability.DOCUMENT_SYMBOLS)
        return await self._with_unhealthy_on_unavailable(
            language, client.document_symbols(relative_path)
        )

    async def diagnostics(self, path: str) -> list[Diagnostic]:
        relative_path = self._validate_path(path)
        language = self._language_for_path(relative_path)
        _ = self._language_context.set(language)
        async with self._document_lock(relative_path):
            document = await self._read_workspace_document(relative_path)
            client = await self._client_for_language(language, Capability.DIAGNOSTICS)
            after_version = self._diagnostics_version(client, relative_path)
            pending_sync = self._diagnostics_pending_syncs.get(relative_path)
            if relative_path not in self._open_documents:
                document_version = await self._did_open_with_client(
                    language, client, relative_path, document.content
                )
                self._document_snapshots[relative_path] = document.snapshot
                await self._wait_for_diagnostics_snapshot(
                    client,
                    relative_path,
                    after_version=after_version,
                    document_version=document_version,
                )
                _ = self._diagnostics_pending_syncs.pop(relative_path, None)
            elif self._document_snapshots.get(relative_path) != document.snapshot:
                document_version = await self._did_change_with_client(
                    language, client, relative_path, document.content
                )
                self._document_snapshots[relative_path] = document.snapshot
                await self._wait_for_diagnostics_snapshot(
                    client,
                    relative_path,
                    after_version=after_version,
                    document_version=document_version,
                )
                _ = self._diagnostics_pending_syncs.pop(relative_path, None)
            elif pending_sync is not None:
                await self._wait_for_diagnostics_snapshot(
                    client,
                    relative_path,
                    after_version=pending_sync.after_publish_version,
                    document_version=pending_sync.document_version,
                )
                _ = self._diagnostics_pending_syncs.pop(relative_path, None)
            diagnostics = await self._with_unhealthy_on_unavailable(
                language, client.diagnostics(relative_path)
            )
            self._diagnostics_cache[relative_path] = diagnostics
            return list(diagnostics)

    async def notify_did_open(self, path: str, content: str) -> None:
        relative_path = self._validate_path(path)
        language = self._language_for_path(relative_path)
        _ = self._language_context.set(language)
        async with self._document_lock(relative_path):
            client = await self._client_for_language(language, Capability.DIAGNOSTICS)
            after_publish_version = self._diagnostics_version(client, relative_path)
            document_version = await self._did_open_with_client(
                language, client, relative_path, content
            )
            self._mark_pending_diagnostics_sync(
                relative_path,
                document_version=document_version,
                after_publish_version=after_publish_version,
            )
            snapshot = await self._try_workspace_document_snapshot(relative_path)
            if snapshot is not None:
                self._document_snapshots[relative_path] = snapshot

    async def notify_did_change(self, path: str, new_content: str) -> None:
        relative_path = self._validate_path(path)
        language = self._language_for_path(relative_path)
        _ = self._language_context.set(language)
        async with self._document_lock(relative_path):
            client = await self._client_for_language(language, Capability.DIAGNOSTICS)
            if relative_path not in self._open_documents:
                after_publish_version = self._diagnostics_version(client, relative_path)
                document_version = await self._did_open_with_client(
                    language, client, relative_path, new_content
                )
                self._mark_pending_diagnostics_sync(
                    relative_path,
                    document_version=document_version,
                    after_publish_version=after_publish_version,
                )
            after_publish_version = self._diagnostics_version(client, relative_path)
            document_version = await self._did_change_with_client(
                language, client, relative_path, new_content
            )
            self._mark_pending_diagnostics_sync(
                relative_path,
                document_version=document_version,
                after_publish_version=after_publish_version,
            )
            snapshot = await self._try_workspace_document_snapshot(relative_path)
            if snapshot is not None:
                self._document_snapshots[relative_path] = snapshot
            else:
                _ = self._document_snapshots.pop(relative_path, None)

    async def notify_did_save(self, path: str, content: str | None = None) -> None:
        relative_path = self._validate_path(path)
        language = self._language_for_path(relative_path)
        _ = self._language_context.set(language)
        async with self._document_lock(relative_path):
            client = await self._client_for_language(language, Capability.DIAGNOSTICS)
            if relative_path not in self._open_documents and content is not None:
                after_publish_version = self._diagnostics_version(client, relative_path)
                document_version = await self._did_open_with_client(
                    language, client, relative_path, content
                )
                self._mark_pending_diagnostics_sync(
                    relative_path,
                    document_version=document_version,
                    after_publish_version=after_publish_version,
                )
                snapshot = await self._try_workspace_document_snapshot(relative_path)
                if snapshot is not None:
                    self._document_snapshots[relative_path] = snapshot
            await self._with_unhealthy_on_unavailable(
                language, client.did_save(relative_path, text=content)
            )

    async def restart(self, language: str) -> ProviderHealth:
        """Explicit restart hook; a successful restart clears permanent unhealthy state."""
        _ = self._language_context.set(language)
        if language not in self.languages:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                health_score=0.0,
                message=f"unsupported language: {language}",
            )
        self._forget_documents_for_language(language)
        try:
            handle = await self.manager.restart(language)
        except ProviderUnavailable as error:
            self._record_restart_failure(language, error)
            return self._health_for_language(language)
        except Exception:
            self._record_restart_failure(
                language, ProviderUnavailable("语言服务器重启失败。")
            )
            return self._health_for_language(language)
        capabilities = self._record_capabilities(language, handle.capabilities)
        if not capabilities:
            return self._health_for_language(language)
        self._health_states[language] = _HealthState.HEALTHY
        _ = self._health_messages.pop(language, None)
        self._restart_failures[language] = 0
        return self._health_for_language(language)

    async def check_health(self, language: str) -> ProviderHealth:
        """Probe an initialized language server and update provider health state."""
        state = self._health_states.get(language, _HealthState.HEALTHY)
        if state == _HealthState.PERMANENTLY_UNHEALTHY:
            return self._health_for_language(language)
        if state == _HealthState.UNHEALTHY:
            return self._health_for_language(language)
        try:
            running = await self.manager.check_health(language)
        except ProviderUnavailable as error:
            self._mark_unhealthy(language, error)
            return self._health_for_language(language)
        except Exception:
            self._mark_unhealthy(
                language, ProviderUnavailable("语言服务器健康检查失败。")
            )
            return self._health_for_language(language)
        if not running:
            self._mark_unhealthy(
                language, ProviderUnavailable("language server is not running")
            )
            return self._health_for_language(language)
        handle = self._initialized_handle_for_language(language)
        if handle is not None:
            capabilities = self._capabilities_by_language.get(language)
            if capabilities is None:
                capabilities = self._record_capabilities(language, handle.capabilities)
            if not capabilities:
                self._mark_unhealthy(
                    language, ProviderUnavailable(_NO_ROUTED_CAPABILITIES_MESSAGE)
                )
                return self._health_for_language(language)
        self._health_states[language] = _HealthState.HEALTHY
        _ = self._health_messages.pop(language, None)
        return self._health_for_language(language)

    async def shutdown(self, language: str | None = None) -> None:
        """Explicit shutdown hook exposed above the manager lifecycle."""
        await self.manager.shutdown(language)
        if language is None:
            self._open_documents.clear()
            self._document_versions.clear()
            self._document_snapshots.clear()
            self._diagnostics_cache.clear()
            self._diagnostics_pending_syncs.clear()
            return
        self._forget_documents_for_language(language)

    async def _client_for_language(
        self, language: str, capability: Capability
    ) -> _ProviderLSPClient:
        await self._ensure_operation_supported(language, capability)
        handle = await self._ensure_handle(language)
        return cast(_ProviderLSPClient, cast(object, handle.client))

    async def _ensure_operation_supported(
        self, language: str, capability: Capability
    ) -> None:
        state = self._health_states.get(language, _HealthState.HEALTHY)
        if state in {_HealthState.UNHEALTHY, _HealthState.PERMANENTLY_UNHEALTHY}:
            raise ProviderUnavailable(
                self._health_messages.get(language, "语言服务器不可用。")
            )
        capabilities = await self._ensure_capabilities(language)
        if capability not in capabilities:
            raise ProviderUnavailable(f"语言服务器不支持 {capability.value}。")

    async def _ensure_capabilities(self, language: str) -> set[Capability]:
        cached = self._capabilities_by_language.get(language)
        if (
            cached is not None
            and self._health_states.get(language, _HealthState.HEALTHY)
            == _HealthState.HEALTHY
        ):
            if not cached:
                error = ProviderUnavailable(_NO_ROUTED_CAPABILITIES_MESSAGE)
                self._mark_unhealthy(language, error)
                raise error
            return cached
        handle = await self._ensure_handle(language)
        capabilities = self._record_capabilities(language, handle.capabilities)
        if not capabilities:
            raise ProviderUnavailable(_NO_ROUTED_CAPABILITIES_MESSAGE)
        return capabilities

    async def _ensure_handle(self, language: str) -> LSPServerHandle:
        try:
            handle = await self.manager.ensure_client(language)
        except ProviderUnavailable as error:
            self._mark_unhealthy(language, error)
            raise
        except Exception:
            error = ProviderUnavailable("语言服务器不可用。")
            self._mark_unhealthy(language, error)
            raise error from None
        self._health_states[language] = _HealthState.HEALTHY
        _ = self._health_messages.pop(language, None)
        _ = self._restart_failures.setdefault(language, 0)
        return handle

    async def _with_unhealthy_on_unavailable(
        self, language: str, operation: Awaitable[T]
    ) -> T:
        try:
            return await operation
        except ProviderUnavailable as error:
            self._mark_unhealthy(language, error)
            raise
        except LSPTimeout:
            raise

    async def _did_open_with_client(
        self,
        language: str,
        client: _ProviderLSPClient,
        relative_path: str,
        content: str,
    ) -> int:
        version = self._next_document_version(relative_path)
        await self._with_unhealthy_on_unavailable(
            language,
            client.did_open(
                relative_path, language_id=language, text=content, version=version
            ),
        )
        self._open_documents.add(relative_path)
        return version

    async def _did_change_with_client(
        self,
        language: str,
        client: _ProviderLSPClient,
        relative_path: str,
        content: str,
    ) -> int:
        version = self._next_document_version(relative_path)
        await self._with_unhealthy_on_unavailable(
            language, client.did_change(relative_path, text=content, version=version)
        )
        return version

    async def _prepare_semantic_request(
        self,
        location: Location,
        language: str,
        client: _ProviderLSPClient,
    ) -> tuple[int, int]:
        relative_path = self._validate_path(location.path)
        fallback = (location.range.start_line, location.range.start_col)
        async with self._document_lock(relative_path):
            try:
                document = await self._read_workspace_document(relative_path)
            except ProviderUnavailable:
                return fallback
            if relative_path not in self._open_documents:
                after_publish_version = self._diagnostics_version(client, relative_path)
                document_version = await self._did_open_with_client(
                    language, client, relative_path, document.content
                )
                self._document_snapshots[relative_path] = document.snapshot
                self._mark_pending_diagnostics_sync(
                    relative_path,
                    document_version=document_version,
                    after_publish_version=after_publish_version,
                )
            elif self._document_snapshots.get(relative_path) != document.snapshot:
                after_publish_version = self._diagnostics_version(client, relative_path)
                document_version = await self._did_change_with_client(
                    language, client, relative_path, document.content
                )
                self._document_snapshots[relative_path] = document.snapshot
                self._mark_pending_diagnostics_sync(
                    relative_path,
                    document_version=document_version,
                    after_publish_version=after_publish_version,
                )
            return self._semantic_position_from_content(
                document.content, location.range
            )

    @classmethod
    def _semantic_position_from_content(
        cls, content: str, target_range: Range
    ) -> tuple[int, int]:
        fallback = (target_range.start_line, target_range.start_col)
        lines = content.splitlines()
        if target_range.start_line >= len(lines):
            return fallback
        line = lines[target_range.start_line]
        start_col = min(target_range.start_col, len(line))
        if target_range.end_line == target_range.start_line:
            end_col = min(max(target_range.end_col, start_col), len(line))
            if end_col <= start_col:
                end_col = len(line)
        else:
            end_col = len(line)
        offset = cls._semantic_identifier_offset(line[start_col:end_col])
        if offset is None:
            return fallback
        return (target_range.start_line, start_col + offset)

    @staticmethod
    def _semantic_identifier_offset(segment: str) -> int | None:
        for pattern in _DECLARATION_IDENTIFIER_PATTERNS:
            match = pattern.search(segment)
            if match is not None:
                return match.start(1)
        for match in _IDENTIFIER_PATTERN.finditer(segment):
            if match.group(0) not in _SEMANTIC_POSITION_KEYWORDS:
                return match.start()
        return None

    def _mark_pending_diagnostics_sync(
        self,
        relative_path: str,
        *,
        document_version: int,
        after_publish_version: int | None,
    ) -> None:
        self._diagnostics_pending_syncs[relative_path] = _PendingDiagnosticsSync(
            document_version=document_version,
            after_publish_version=after_publish_version,
        )

    def _diagnostics_version(
        self, client: _ProviderLSPClient, relative_path: str
    ) -> int | None:
        if not isinstance(client, _DiagnosticsWaitClient):
            return None
        return client.diagnostics_version(relative_path)

    async def _wait_for_diagnostics_snapshot(
        self,
        client: _ProviderLSPClient,
        relative_path: str,
        *,
        after_version: int | None,
        document_version: int,
    ) -> None:
        if not isinstance(client, _DiagnosticsWaitClient):
            return
        timeout = max(0.0, self.manager.request_timeout_seconds)
        refreshed = await client.wait_for_diagnostics(
            relative_path,
            timeout=timeout,
            after_version=after_version,
            document_version=document_version,
        )
        if not refreshed:
            raise LSPTimeout("LSP diagnostics did not refresh before timeout")

    async def _read_workspace_document(self, relative_path: str) -> _WorkspaceDocument:
        try:
            return await asyncio.to_thread(
                self._read_workspace_document_sync, relative_path
            )
        except ProviderUnavailable:
            raise
        except Exception:
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None

    def _read_workspace_document_sync(self, relative_path: str) -> _WorkspaceDocument:
        try:
            root = self.manager.workspace_root.resolve(strict=False)
            resolved_path = (self.manager.workspace_root / relative_path).resolve(
                strict=True
            )
            _ = resolved_path.relative_to(root)
            file_stat = resolved_path.stat()
        except (OSError, ValueError):
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None
        if not stat.S_ISREG(file_stat.st_mode):
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None
        if file_stat.st_size > self.max_sync_file_size_bytes:
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None
        try:
            with resolved_path.open("rb") as handle:
                raw = handle.read(self.max_sync_file_size_bytes + 1)
            if len(raw) > self.max_sync_file_size_bytes:
                raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE)
            content = raw.decode("utf-8")
            file_stat = resolved_path.stat()
        except (OSError, UnicodeError):
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None
        if file_stat.st_size > self.max_sync_file_size_bytes:
            raise ProviderUnavailable(_DIAGNOSTICS_READ_FAILURE_MESSAGE) from None
        return _WorkspaceDocument(
            content=content,
            snapshot=_DocumentSnapshot(
                resolved_path=resolved_path,
                mtime_ns=file_stat.st_mtime_ns,
                size=file_stat.st_size,
            ),
        )

    async def _try_workspace_document_snapshot(
        self, relative_path: str
    ) -> _DocumentSnapshot | None:
        try:
            document = await self._read_workspace_document(relative_path)
        except ProviderUnavailable:
            return None
        return document.snapshot

    def _document_lock(self, relative_path: str) -> asyncio.Lock:
        return self._document_locks.setdefault(relative_path, asyncio.Lock())

    def _initialized_handle_for_language(self, language: str) -> LSPServerHandle | None:
        for handle in self.manager.handles:
            if handle.spec.language == language:
                return handle
        return None

    def _forget_documents_for_language(self, language: str) -> None:
        tracked_paths = (
            set(self._open_documents)
            | set(self._document_versions)
            | set(self._document_snapshots)
            | set(self._diagnostics_cache)
            | set(self._diagnostics_pending_syncs)
        )
        for tracked_path in tracked_paths:
            if self._language_for_path(tracked_path) != language:
                continue
            self._open_documents.discard(tracked_path)
            _ = self._document_versions.pop(tracked_path, None)
            _ = self._document_snapshots.pop(tracked_path, None)
            _ = self._diagnostics_cache.pop(tracked_path, None)
            _ = self._diagnostics_pending_syncs.pop(tracked_path, None)

    def _record_capabilities(
        self, language: str, server_capabilities: lsp_types.ServerCapabilities
    ) -> set[Capability]:
        capabilities = self._capabilities_from_server(server_capabilities)
        self._capabilities_by_language[language] = capabilities
        self.capabilities = (
            set().union(*self._capabilities_by_language.values())
            if self._capabilities_by_language
            else set()
        )
        if not capabilities:
            self._mark_unhealthy(
                language, ProviderUnavailable(_NO_ROUTED_CAPABILITIES_MESSAGE)
            )
        return capabilities

    def _mark_unhealthy(self, language: str, error: ProviderUnavailable) -> None:
        self._health_states[language] = _HealthState.UNHEALTHY
        detail = error.detail or "语言服务器不可用。"
        self._health_messages[language] = f"unhealthy: {detail}"

    def _record_restart_failure(
        self, language: str, error: ProviderUnavailable
    ) -> None:
        failures = self._restart_failures.get(language, 0) + 1
        self._restart_failures[language] = failures
        detail = error.detail or "语言服务器重启失败。"
        if failures >= 3:
            self._health_states[language] = _HealthState.PERMANENTLY_UNHEALTHY
            self._health_messages[language] = f"permanently_unhealthy: {detail}"
        else:
            self._health_states[language] = _HealthState.UNHEALTHY
            self._health_messages[language] = f"unhealthy: {detail}"

    def _health_for_language(self, language: str) -> ProviderHealth:
        state = self._health_states.get(language, _HealthState.HEALTHY)
        if state == _HealthState.PERMANENTLY_UNHEALTHY:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                health_score=0.0,
                message=self._health_messages.get(language, "permanently_unhealthy"),
            )
        if state == _HealthState.UNHEALTHY:
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                health_score=0.0,
                message=self._health_messages.get(language, "unhealthy"),
            )
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    @classmethod
    def _capabilities_from_server(
        cls, server_capabilities: lsp_types.ServerCapabilities
    ) -> set[Capability]:
        capabilities: set[Capability] = set()
        if cls._server_capability_enabled(server_capabilities.definition_provider):
            capabilities.add(Capability.DEFINITION)
        if cls._server_capability_enabled(server_capabilities.references_provider):
            capabilities.add(Capability.REFERENCES)
        if cls._server_capability_enabled(server_capabilities.hover_provider):
            capabilities.add(Capability.HOVER)
        if cls._server_capability_enabled(server_capabilities.document_symbol_provider):
            capabilities.add(Capability.DOCUMENT_SYMBOLS)
        if cls._server_capability_enabled(
            server_capabilities.diagnostic_provider
        ) or cls._text_sync_enabled(server_capabilities.text_document_sync):
            capabilities.add(Capability.DIAGNOSTICS)
        return capabilities

    @staticmethod
    def _server_capability_enabled(value: object) -> bool:
        return value is not None and value is not False

    @staticmethod
    def _text_sync_enabled(value: object) -> bool:
        if value is None or value == lsp_types.TextDocumentSyncKind.None_:
            return False
        if isinstance(value, lsp_types.TextDocumentSyncOptions):
            return bool(
                value.open_close
                or value.change not in (None, lsp_types.TextDocumentSyncKind.None_)
                or value.save not in (None, False)
            )
        return True

    @staticmethod
    def _target_location(target: CodeTarget) -> Location:
        if target.location is None:
            raise ProviderUnavailable("LSP 语义查询需要明确的文件位置。")
        return target.location

    def _next_document_version(self, path: str) -> int:
        version = self._document_versions.get(path, 0) + 1
        self._document_versions[path] = version
        return version

    def _language_for_path(self, path: str) -> str:
        relative_path = self._validate_path(path)
        language = _LANGUAGE_BY_EXTENSION.get(
            PurePosixPath(relative_path).suffix.casefold()
        )
        if language is None or language not in self.languages:
            raise ProviderUnavailable("当前文件没有可用的 LSP 语言服务器。")
        return language

    @staticmethod
    def _validate_path(path: str) -> str:
        try:
            return validate_workspace_relative_path(path)
        except ValueError:
            raise ProviderUnavailable("invalid LSP workspace path") from None


__all__ = ["LSPProvider"]
