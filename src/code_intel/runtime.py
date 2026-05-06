"""CodeIntel runtime lifecycle owner for REPL and commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.code_intel import CodeIntelKernel
from src.code_intel.config import (
    CodeIntelConfig,
    CodeIntelConfigError,
    code_intel_index_db_path,
    load_code_intel_config,
)
from src.code_intel.providers.text_search import TextSearchProvider
from src.code_intel.providers.tree_sitter import TreeSitterProvider
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
    """Own CodeIntel kernel providers and global lifecycle binding."""

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
        self._owns_global_kernel = False
        self._initialized = False

    async def initialize(self) -> CodeIntelRuntimeStatus:
        """Perform cheap startup work and bind the global CodeIntel kernel."""
        if self._initialized:
            return self.status

        if self.config_error is not None:
            self._warn(self.config_error.format())
        if self.config.enabled:
            self._register_static_providers()
        self._set_global_kernel()
        self.status.initialized = True
        self._initialized = True
        return self.status

    async def close(self) -> None:
        """Release owned global bindings."""
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

    def _set_global_kernel(self) -> None:
        set_code_intel_kernel(self.kernel)
        self._owns_global_kernel = True

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
