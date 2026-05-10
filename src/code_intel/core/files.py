"""Filesystem helpers shared inside code_intel."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .errors import SymbolNotFound
from .models import validate_workspace_relative_path


async def read_source_text(workspace_root: Path, path: str, context: str) -> str:
    """Validate a workspace-relative path and read source text safely."""
    relative_path = validate_workspace_relative_path(path)
    absolute_path = (workspace_root / relative_path).resolve(strict=False)
    try:
        _ = absolute_path.relative_to(workspace_root)
    except ValueError as error:
        raise SymbolNotFound(f"{context} path escaped workspace") from error

    def read_sync() -> str:
        return absolute_path.read_text(encoding="utf-8", errors="replace")

    try:
        return await asyncio.to_thread(read_sync)
    except OSError as error:
        raise SymbolNotFound(f"{context} source file is unavailable") from error


__all__ = ["read_source_text"]
