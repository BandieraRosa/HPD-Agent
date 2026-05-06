"""CodeIntel runtime lifecycle owner for REPL and commands."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from src.code_intel import CodeIntelKernel
from src.code_intel.config import (
    CodeIntelConfig,
    CodeIntelConfigError,
    code_intel_index_db_path,
    load_code_intel_config,
)
from src.code_intel.core import Symbol
from src.code_intel.index import SymbolIndexer, SymbolIndexStore
from src.code_intel.providers.text_search import TextSearchProvider
from src.code_intel.providers.tree_sitter import (
    TREE_SITTER_INDEX_VERSION,
    TREE_SITTER_QUERY_VERSION,
    TreeSitterProvider,
)
from src.code_intel.tools.runtime import get_code_intel_kernel, set_code_intel_kernel


@dataclass
class CodeIntelRuntimeStatus:
    """Visible runtime state for startup diagnostics and commands."""

    initialized: bool = False
    enabled: bool = True
    workspace_root: Path | None = None
    db_path: Path | None = None
    symbol_index_attached: bool = False
    index_build_scheduled: bool = False
    index_build_running: bool = False
    index_build_completed: bool = False
    index_build_failed: bool = False
    lsp_provider_registered: bool = False
    lsp_prewarm_scheduled: bool = False
    warnings: list[str] = field(default_factory=list)


class CodeIntelRuntime:
    """Own CodeIntel kernel providers, symbol index, and background work."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        *,
        config: CodeIntelConfig | None = None,
        config_error: CodeIntelConfigError | None = None,
        kernel: CodeIntelKernel | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve(strict=False)
        if config is None:
            loaded = load_code_intel_config()
            config = loaded.config
            config_error = loaded.error
        self.config = config
        self.config_error = config_error
        self.kernel = kernel or CodeIntelKernel(workspace_root=self.workspace_root)
        self.status = CodeIntelRuntimeStatus(
            enabled=config.enabled,
            workspace_root=self.workspace_root,
            db_path=code_intel_index_db_path(self.workspace_root, config),
        )
        self.symbol_store: SymbolIndexStore | None = None
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._owns_global_kernel = False
        self._initialized = False

    async def initialize(self) -> CodeIntelRuntimeStatus:
        """Perform cheap startup work and schedule optional background work."""
        if self._initialized:
            return self.status

        if self.config_error is not None:
            self._warn(self.config_error.format())
        if self.config.enabled:
            self._register_static_providers()
            await self._attach_existing_index()
            self._schedule_startup_work()
        self._set_global_kernel()
        self.status.initialized = True
        self._initialized = True
        return self.status

    async def build_symbol_index(self) -> object:
        """Build the workspace symbol index now and attach it in-place."""
        db_path = self._db_path()
        provider = TreeSitterProvider(self.workspace_root)
        store = SymbolIndexStore(db_path)

        async def extract_symbols(
            workspace_root: Path, path: str, language: str
        ) -> Sequence[Symbol]:
            _ = workspace_root, language
            return await provider.outline(path)

        indexer = SymbolIndexer(
            self.workspace_root,
            extractor=extract_symbols,
            store=store,
            grammar_version=TREE_SITTER_INDEX_VERSION,
            query_version=TREE_SITTER_QUERY_VERSION,
            max_file_size_bytes=self.config.index.max_file_size_bytes,
        )
        try:
            result = await indexer.index_workspace()
        except Exception:
            await store.close()
            raise
        await self._replace_symbol_store(store)
        self.status.index_build_completed = True
        self.status.index_build_failed = False
        return result

    async def clear_symbol_index(self) -> int:
        """Close, detach, and remove symbol index files for this workspace."""
        if self.symbol_store is not None:
            await self.symbol_store.close()
            self.symbol_store = None
        self.kernel.attach_symbol_index(None, self.workspace_root)
        self.status.symbol_index_attached = False
        removed = 0
        db_path = self._db_path()
        for candidate in (
            db_path,
            Path(str(db_path) + "-wal"),
            Path(str(db_path) + "-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.unlink()
                    removed += 1
            except OSError as error:
                self._warn(
                    f"代码索引清理失败: {candidate} ({error.__class__.__name__})"
                )
        return removed

    async def close(self) -> None:
        """Cancel background work and release owned resources."""
        tasks = list(self._background_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    self._warn(f"后台任务结束异常: {result.__class__.__name__}")
        self._background_tasks.clear()

        if self.symbol_store is not None:
            try:
                await self.symbol_store.close()
            finally:
                self.symbol_store = None
        if self._owns_global_kernel and get_code_intel_kernel() is self.kernel:
            set_code_intel_kernel(None)
            self._owns_global_kernel = False

    def _register_static_providers(self) -> None:
        if self.config.providers.text_search:
            try:
                self.kernel.register_provider(
                    TextSearchProvider(
                        self.workspace_root,
                        max_file_size_bytes=self.config.index.max_file_size_bytes,
                    )
                )
            except Exception as error:
                self._warn(
                    f"text_search provider 初始化失败: {error.__class__.__name__}"
                )
        if self.config.providers.tree_sitter:
            try:
                self.kernel.register_provider(TreeSitterProvider(self.workspace_root))
            except Exception as error:
                self._warn(
                    f"tree_sitter provider 初始化失败: {error.__class__.__name__}"
                )

    async def _attach_existing_index(self) -> None:
        db_path = self._db_path()
        if not db_path.exists():
            return
        store = SymbolIndexStore(db_path)
        try:
            await store.initialize()
        except Exception as error:
            await store.close()
            self._warn(f"代码索引加载失败: {error.__class__.__name__} - {error}")
            return
        await self._replace_symbol_store(store)

    async def _replace_symbol_store(self, store: SymbolIndexStore) -> None:
        previous = self.symbol_store
        self.symbol_store = store
        self.kernel.attach_symbol_index(store, self.workspace_root)
        self.status.symbol_index_attached = True
        if previous is not None and previous is not store:
            await previous.close()

    def _schedule_startup_work(self) -> None:
        db_path = self._db_path()
        if self.config.index.auto_build_on_startup and not db_path.exists():
            self.status.index_build_scheduled = True
            self._track_task(asyncio.create_task(self._background_index_build()))

    async def _background_index_build(self) -> None:
        self.status.index_build_running = True
        try:
            _ = await self.build_symbol_index()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.status.index_build_failed = True
            self._warn(f"后台代码索引构建失败: {error.__class__.__name__} - {error}")
        finally:
            self.status.index_build_running = False

    def _track_task(self, task: asyncio.Task[object]) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _set_global_kernel(self) -> None:
        set_code_intel_kernel(self.kernel)
        self._owns_global_kernel = True

    def _db_path(self) -> Path:
        return cast(Path, self.status.db_path)

    def _warn(self, message: str) -> None:
        self.status.warnings.append(message)


async def create_code_intel_runtime(
    workspace_root: str | Path = ".",
) -> CodeIntelRuntime:
    """Create and initialize the default CodeIntel runtime."""
    runtime = CodeIntelRuntime(workspace_root)
    await runtime.initialize()
    return runtime


__all__ = ["CodeIntelRuntime", "CodeIntelRuntimeStatus", "create_code_intel_runtime"]
