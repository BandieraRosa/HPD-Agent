"""Path validation helpers shared by LSP provider/client boundaries."""

from __future__ import annotations

from src.code_intel.core import ProviderUnavailable
from src.code_intel.core.models import validate_workspace_relative_path


def validate_lsp_workspace_path(path: str) -> str:
    """Validate an LSP workspace-relative path and map failures to ProviderUnavailable."""
    try:
        return validate_workspace_relative_path(path)
    except ValueError:
        raise ProviderUnavailable("invalid LSP workspace path") from None


__all__ = ["validate_lsp_workspace_path"]
