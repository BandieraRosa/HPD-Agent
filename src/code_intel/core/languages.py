"""Shared language resolution helpers for code_intel providers and tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath

DEFAULT_LANGUAGE = "python"
LANGUAGE_BY_EXTENSION: dict[str, str] = {
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
TOOL_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    **LANGUAGE_BY_EXTENSION,
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}
SUPPORTED_CODE_LANGUAGES = frozenset(LANGUAGE_BY_EXTENSION.values())


def language_for_path(
    path: str,
    languages_by_extension: Mapping[str, str] = LANGUAGE_BY_EXTENSION,
) -> str | None:
    """Return the configured language for a path extension, if supported."""
    return languages_by_extension.get(PurePosixPath(path).suffix.casefold())


def language_for_path_or_default(
    path: str | None,
    *,
    default: str = DEFAULT_LANGUAGE,
    languages_by_extension: Mapping[str, str] = LANGUAGE_BY_EXTENSION,
) -> str:
    """Return a path language while preserving callers that fall back to Python."""
    if path is None:
        return default
    return language_for_path(path, languages_by_extension) or default


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_BY_EXTENSION",
    "SUPPORTED_CODE_LANGUAGES",
    "TOOL_LANGUAGE_BY_EXTENSION",
    "language_for_path",
    "language_for_path_or_default",
]
