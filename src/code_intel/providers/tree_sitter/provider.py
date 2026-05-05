"""Async Tree-sitter syntax provider for document outlines."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.code_intel.core import (
    Capability,
    ConfidenceClass,
    ProviderHealth,
    ProviderStatus,
    Symbol,
)

from .parser import SUPPORTED_LANGUAGES, TreeSitterParser


class TreeSitterProvider:
    """Real Tree-sitter provider for syntax-only document symbols."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        *,
        name: str = "tree_sitter",
        languages: set[str] | None = None,
        parser: TreeSitterParser | None = None,
        health: ProviderHealth | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[Capability] = {
            Capability.OUTLINE,
            Capability.DOCUMENT_SYMBOLS,
        }
        self.languages: set[str] = set(languages or SUPPORTED_LANGUAGES)
        self.workspace_root: Path = (
            Path(workspace_root).expanduser().resolve(strict=False)
        )
        self._parser: TreeSitterParser = parser or TreeSitterParser()
        self._configured_health: ProviderHealth | None = health

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        if self._configured_health is not None:
            return self._configured_health
        if not self.workspace_root.is_dir():
            return ProviderHealth(
                status=ProviderStatus.UNAVAILABLE,
                health_score=0.0,
                message="workspace root is not a directory",
            )
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def confidence_for(
        self, capability: Capability, _language: str
    ) -> ConfidenceClass:
        if capability in self.capabilities:
            return ConfidenceClass.MEDIUM
        return ConfidenceClass.LOW

    async def outline(self, path: str) -> list[Symbol]:
        return await asyncio.to_thread(self._outline_sync, path)

    async def document_symbols(self, path: str) -> list[Symbol]:
        return await self.outline(path)

    def _outline_sync(self, path: str) -> list[Symbol]:
        return self._parser.extract_symbols(self.workspace_root, path)


__all__ = ["TreeSitterProvider"]
