"""Lifecycle manager for lazily started LSP clients."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lsprotocol import types as lsp_types

from src.code_intel.core import CodeIntelError, ProviderUnavailable

from .client import LSPClient
from .registry import LanguageServerSpec, language_server_specs
from .transport import LSPTransport

LSPManagerKey = tuple[str, str, tuple[str, ...], str]
_DEFAULT_IDLE_TIMEOUT_SECONDS = 600.0
_DEFAULT_DETECT_TIMEOUT_SECONDS = 5.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0


class ManagedLSPClient(Protocol):
    """Client methods the manager needs for lifecycle ownership."""

    async def initialize(
        self,
        *,
        root_uri: str | None = None,
        initialization_options: object | None = None,
    ) -> lsp_types.InitializeResult: ...

    async def shutdown(self) -> None: ...

    async def close(self) -> None: ...

    async def is_running(self) -> bool: ...


class LSPClientFactory(Protocol):
    """Factory used by tests to inject mock clients without spawning real servers."""

    def __call__(
        self,
        spec: LanguageServerSpec,
        workspace_root: Path,
        command: Sequence[str],
        *,
        request_timeout_seconds: float,
    ) -> ManagedLSPClient: ...


@dataclass
class LSPServerHandle:
    """Initialized client plus cached initialize metadata."""

    key: LSPManagerKey
    spec: LanguageServerSpec
    client: ManagedLSPClient
    initialize_result: lsp_types.InitializeResult
    capabilities: lsp_types.ServerCapabilities
    started_at: float
    last_used_at: float
    idle_timeout_seconds: float

    def touch(self, now: float | None = None) -> None:
        self.last_used_at = time.monotonic() if now is None else now

    def idle_for(self, now: float | None = None) -> float:
        reference_time = time.monotonic() if now is None else now
        return max(0.0, reference_time - self.last_used_at)


def _default_client_factory(
    spec: LanguageServerSpec,
    workspace_root: Path,
    command: Sequence[str],
    *,
    request_timeout_seconds: float,
) -> ManagedLSPClient:
    transport = LSPTransport(
        command, cwd=workspace_root, default_timeout=request_timeout_seconds
    )
    return LSPClient(
        transport,
        workspace_root=workspace_root,
        language=spec.language,
        source=spec.name,
    )


class LSPManager:
    """Cache and supervise initialized LSP clients by workspace/language/command/options."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        *,
        specs: Iterable[LanguageServerSpec] | None = None,
        client_factory: LSPClientFactory | None = None,
        idle_timeout_seconds: float = _DEFAULT_IDLE_TIMEOUT_SECONDS,
        request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.workspace_root: Path = (
            Path(workspace_root).expanduser().resolve(strict=False)
        )
        self._specs: dict[str, LanguageServerSpec] = language_server_specs(specs)
        self._client_factory: LSPClientFactory = (
            client_factory or _default_client_factory
        )
        self.idle_timeout_seconds: float = max(0.0, idle_timeout_seconds)
        self.request_timeout_seconds: float = request_timeout_seconds
        self._handles: dict[LSPManagerKey, LSPServerHandle] = {}
        self._startup_locks: dict[LSPManagerKey, asyncio.Lock] = {}

    @property
    def languages(self) -> set[str]:
        """Languages with registered server specs."""
        return set(self._specs)

    @property
    def handles(self) -> tuple[LSPServerHandle, ...]:
        """Return initialized handles for diagnostics/tests."""
        return tuple(self._handles.values())

    def spec_for(self, language: str) -> LanguageServerSpec:
        try:
            return self._specs[language]
        except KeyError:
            raise ProviderUnavailable(f"未注册 {language} 的语言服务器。") from None

    def key_for(self, language: str) -> LSPManagerKey:
        spec = self.spec_for(language)
        return self.key_for_spec(spec)

    def key_for_spec(self, spec: LanguageServerSpec) -> LSPManagerKey:
        command = tuple(spec.launch_command)
        return (
            str(self.workspace_root),
            spec.language,
            command,
            self._init_options_hash(spec.init_options),
        )

    async def ensure_client(self, language: str) -> LSPServerHandle:
        """Return an initialized handle, starting the server lazily if needed."""
        spec = self.spec_for(language)
        key = self.key_for_spec(spec)
        handle = self._handles.get(key)
        if handle is not None:
            handle.touch()
            return handle

        lock = self._startup_locks.setdefault(key, asyncio.Lock())
        async with lock:
            handle = self._handles.get(key)
            if handle is not None:
                handle.touch()
                return handle

            await self.detect_server(language)

            client = self._client_factory(
                spec,
                self.workspace_root,
                tuple(spec.launch_command),
                request_timeout_seconds=self.request_timeout_seconds,
            )
            started_at = time.monotonic()
            try:
                initialize_result = await client.initialize(
                    root_uri=self.workspace_root.as_uri(),
                    initialization_options=spec.init_options or None,
                )
            except CodeIntelError:
                await self._close_after_failed_start(client)
                raise
            except Exception:
                await self._close_after_failed_start(client)
                raise ProviderUnavailable("语言服务器初始化失败。") from None

            handle = LSPServerHandle(
                key=key,
                spec=spec,
                client=client,
                initialize_result=initialize_result,
                capabilities=initialize_result.capabilities,
                started_at=started_at,
                last_used_at=started_at,
                idle_timeout_seconds=self.idle_timeout_seconds,
            )
            self._handles[key] = handle
            return handle

    async def capabilities_for(self, language: str) -> lsp_types.ServerCapabilities:
        """Return cached initialize capabilities, starting the server if needed."""
        handle = await self.ensure_client(language)
        return handle.capabilities

    async def initialize_result_for(self, language: str) -> lsp_types.InitializeResult:
        """Return the cached initialize result for a language."""
        handle = await self.ensure_client(language)
        return handle.initialize_result

    async def detect_server(self, language: str) -> None:
        """Run the full configured detect command before starting a server."""
        await self._run_detect_command(self.spec_for(language))

    async def check_health(self, language: str) -> bool:
        """Return whether an already initialized server for a language is still running."""
        spec = self.spec_for(language)
        handle = self._handles.get(self.key_for_spec(spec))
        if handle is None:
            return True
        try:
            running = await handle.client.is_running()
        except Exception:
            return False
        if running:
            handle.touch()
        return running

    async def restart(self, language: str) -> LSPServerHandle:
        """Explicitly restart one language server and return its new initialized handle."""
        await self.shutdown(language)
        return await self.ensure_client(language)

    async def shutdown(self, language: str | None = None) -> None:
        """Gracefully shutdown one language server, or all initialized servers."""
        if language is None:
            keys = list(self._handles)
        else:
            spec = self.spec_for(language)
            keys = [self.key_for_spec(spec)]
        for key in keys:
            handle = self._handles.pop(key, None)
            if handle is None:
                continue
            await self._shutdown_client(handle.client)

    async def shutdown_idle(self, now: float | None = None) -> int:
        """Shutdown handles whose tracked idle time exceeded the configured timeout."""
        reference_time = time.monotonic() if now is None else now
        idle_keys = [
            key
            for key, handle in self._handles.items()
            if handle.idle_timeout_seconds >= 0.0
            and handle.idle_for(reference_time) >= handle.idle_timeout_seconds
        ]
        for key in idle_keys:
            handle = self._handles.pop(key, None)
            if handle is not None:
                await self._shutdown_client(handle.client)
        return len(idle_keys)

    async def _shutdown_client(self, client: ManagedLSPClient) -> None:
        try:
            await client.shutdown()
            return
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            return

    @staticmethod
    async def _close_after_failed_start(client: ManagedLSPClient) -> None:
        try:
            await client.close()
        except Exception:
            return

    @staticmethod
    def _init_options_hash(init_options: dict[str, object]) -> str:
        payload = json.dumps(
            init_options,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    async def _run_detect_command(self, spec: LanguageServerSpec) -> None:
        command = tuple(spec.detect_command)
        if not command:
            raise ProviderUnavailable(
                self._detect_failure_message(spec, "检测命令为空")
            )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root),
            )
        except OSError:
            raise ProviderUnavailable(
                self._detect_failure_message(spec, "检测命令无法执行")
            ) from None

        try:
            _stdout, _stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=_DEFAULT_DETECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            _ = await process.wait()
            raise ProviderUnavailable(
                self._detect_failure_message(spec, "检测命令超时")
            ) from None

        if process.returncode != 0:
            raise ProviderUnavailable(
                self._detect_failure_message(
                    spec, f"检测命令退出码 {process.returncode}"
                )
            )

    @staticmethod
    def _detect_failure_message(spec: LanguageServerSpec, reason: str) -> str:
        command_text = " ".join(spec.detect_command) or spec.name
        return f"缺少语言服务器 {spec.name}；{reason}：{command_text}；请先安装：{spec.install_hint}"


__all__ = [
    "LSPClientFactory",
    "LSPManager",
    "LSPManagerKey",
    "LSPServerHandle",
    "ManagedLSPClient",
]
