"""Safe pathspec-backed workspace text search provider."""

from __future__ import annotations

import asyncio
import re
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import ClassVar

from pathspec import PathSpec
from pathspec.pattern import Pattern

from src.code_intel.core import (
    Capability,
    CodeIntelError,
    ConfidenceClass,
    Location,
    ProviderHealth,
    ProviderStatus,
    ProviderUnavailable,
    Range,
    ToolMeta,
)
from src.code_intel.core.models import validate_workspace_relative_path

_BINARY_CHUNK_SIZE = 4096
_DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000
_DEFAULT_MAX_RESULTS = 1000
_MAX_GITIGNORE_SIZE_BYTES = 256_000
_ALWAYS_IGNORED_NAMES = {".git"}
_SUPPORTED_LANGUAGES = {
    "python",
    "typescript",
    "javascript",
    "go",
    "rust",
    "java",
    "kotlin",
    "ruby",
    "php",
    "csharp",
    "c",
    "cpp",
    "text",
}


class InvalidSearchPattern(CodeIntelError):
    """Regex validation failure exposed as a safe Chinese ToolError."""

    code: ClassVar[str] = "invalid_regex"
    message: ClassVar[str] = "正则表达式无效。"
    hint: ClassVar[str | None] = "请检查括号、转义字符或改用 literal 文本搜索。"


class InvalidSearchPath(CodeIntelError):
    """Path validation failure exposed as a safe Chinese ToolError."""

    code: ClassVar[str] = "invalid_search_path"
    message: ClassVar[str] = "搜索路径必须位于工作区内。"
    hint: ClassVar[str | None] = "请使用不含绝对路径、反斜杠或 .. 片段的工作区相对路径。"


class TextSearchLocations(list[Location]):
    """Core Location results with optional ToolMeta for kernel propagation."""

    def __init__(self, locations: Iterable[Location] = (), *, tool_meta: ToolMeta | None = None) -> None:
        super().__init__(locations)
        self.tool_meta: ToolMeta = tool_meta or ToolMeta()


class TextSearchProvider:
    """Pure-Python workspace scanner for literal and regex text search."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        name: str = "text_search",
        languages: set[str] | None = None,
        max_file_size_bytes: int = _DEFAULT_MAX_FILE_SIZE_BYTES,
        max_results: int = _DEFAULT_MAX_RESULTS,
        health: ProviderHealth | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[Capability] = {Capability.TEXT_SEARCH}
        self.languages: set[str] = set(languages or _SUPPORTED_LANGUAGES)
        self.workspace_root: Path = Path(workspace_root).expanduser().resolve(strict=False)
        self.max_file_size_bytes: int = max(1, max_file_size_bytes)
        self.max_results: int = max(1, max_results)
        self._configured_health: ProviderHealth | None = health
        self._gitignore_spec: PathSpec[Pattern] = self._load_gitignore_spec()

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and (language in self.languages or "*" in self.languages)

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

    async def confidence_for(self, capability: Capability, _language: str) -> ConfidenceClass:
        if capability == Capability.TEXT_SEARCH:
            return ConfidenceClass.HIGH
        return ConfidenceClass.LOW

    async def text_search(
        self,
        query: str,
        path: str | None = None,
        limit: int = 20,
        *,
        max_results: int | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> list[Location]:
        return await asyncio.to_thread(
            self._text_search_sync,
            query,
            path,
            limit,
            max_results,
            regex,
            case_sensitive,
        )

    def _text_search_sync(
        self,
        query: str,
        path: str | None,
        limit: int,
        max_results: int | None,
        regex: bool,
        case_sensitive: bool,
    ) -> TextSearchLocations:
        if not self.workspace_root.is_dir():
            raise ProviderUnavailable("workspace root is not available")
        if query == "":
            return TextSearchLocations()

        pattern = self._compile_pattern(query, regex=regex, case_sensitive=case_sensitive)
        result_limit = self._requested_limit(limit, max_results)
        if result_limit <= 0:
            return TextSearchLocations()

        locations: list[Location] = []
        meta = ToolMeta()
        for file_path in self._iter_candidate_files(path):
            text, oversized = self._read_text_file(file_path)
            if oversized:
                meta.truncated = True
                continue
            if text is None:
                continue

            for match in pattern.finditer(text):
                if len(locations) >= result_limit:
                    meta.more_available = True
                    return TextSearchLocations(locations, tool_meta=meta)
                locations.append(self._location_for_span(file_path, text, match.span()))

        return TextSearchLocations(locations, tool_meta=meta)

    def _load_gitignore_spec(self) -> PathSpec[Pattern]:
        gitignore = self.workspace_root / ".gitignore"
        try:
            if not gitignore.is_file() or gitignore.stat().st_size > _MAX_GITIGNORE_SIZE_BYTES:
                return PathSpec.from_lines("gitignore", [])
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return PathSpec.from_lines("gitignore", [])
        return PathSpec.from_lines("gitignore", lines)

    @staticmethod
    def _compile_pattern(query: str, *, regex: bool, case_sensitive: bool) -> re.Pattern[str]:
        flags = 0 if case_sensitive else re.IGNORECASE
        source = query if regex else re.escape(query)
        try:
            return re.compile(source, flags)
        except re.error as error:
            raise InvalidSearchPattern(str(error)) from error

    def _requested_limit(self, limit: int, max_results: int | None) -> int:
        requested = max_results if max_results is not None else limit
        return max(0, min(requested, self.max_results))

    def _iter_candidate_files(self, requested_path: str | None) -> Iterator[Path]:
        start = self._start_path(requested_path)
        if start is None:
            return
        if start.is_file():
            if self._is_searchable_file(start):
                yield start
            return
        if not start.is_dir():
            return

        stack = [start]
        while stack:
            directory = stack.pop()
            try:
                children = sorted(directory.iterdir(), key=lambda child: child.name)
            except OSError:
                continue

            directories: list[Path] = []
            for child in children:
                if child.is_symlink():
                    continue
                relative_path = self._relative_path(child)
                if relative_path is None:
                    continue
                if child.is_dir():
                    if not self._is_ignored(relative_path, is_dir=True):
                        directories.append(child)
                elif child.is_file() and not self._is_ignored(relative_path, is_dir=False):
                    yield child
            stack.extend(reversed(directories))

    def _start_path(self, requested_path: str | None) -> Path | None:
        if requested_path is None:
            return self.workspace_root
        relative_path = self._validate_requested_path(requested_path)
        target = (self.workspace_root / relative_path).resolve(strict=False)
        if not self._is_inside_workspace(target) or target.is_symlink():
            return None
        relative_target = self._relative_path(target)
        if relative_target is not None and self._is_ignored(relative_target, is_dir=target.is_dir()):
            return None
        return target

    @staticmethod
    def _validate_requested_path(path: str) -> str:
        try:
            return validate_workspace_relative_path(path)
        except ValueError as error:
            raise InvalidSearchPath(str(error)) from error

    def _relative_path(self, path: Path) -> str | None:
        try:
            relative = path.resolve(strict=False).relative_to(self.workspace_root)
        except (OSError, ValueError):
            return None
        relative_path = relative.as_posix()
        if not relative_path or relative_path == ".":
            return None
        try:
            return validate_workspace_relative_path(relative_path)
        except ValueError:
            return None

    def _is_inside_workspace(self, path: Path) -> bool:
        try:
            _ = path.relative_to(self.workspace_root)
        except ValueError:
            return False
        return True

    def _is_ignored(self, relative_path: str, *, is_dir: bool) -> bool:
        path_parts = set(PurePosixPath(relative_path).parts)
        if path_parts.intersection(_ALWAYS_IGNORED_NAMES):
            return True
        if self._gitignore_spec.match_file(relative_path):
            return True
        return is_dir and self._gitignore_spec.match_file(f"{relative_path}/")

    def _is_searchable_file(self, path: Path) -> bool:
        if path.is_symlink() or not self._is_inside_workspace(path.resolve(strict=False)):
            return False
        relative_path = self._relative_path(path)
        return relative_path is not None and not self._is_ignored(relative_path, is_dir=False)

    def _read_text_file(self, path: Path) -> tuple[str | None, bool]:
        try:
            file_size = path.stat().st_size
        except OSError:
            return None, False
        if file_size > self.max_file_size_bytes:
            return None, True

        try:
            with path.open("rb") as handle:
                first_chunk = handle.read(min(_BINARY_CHUNK_SIZE, self.max_file_size_bytes))
                if b"\0" in first_chunk:
                    return None, False
                remainder = handle.read(max(0, self.max_file_size_bytes - len(first_chunk)) + 1)
        except OSError:
            return None, False

        raw = first_chunk + remainder
        if len(raw) > self.max_file_size_bytes:
            return None, True
        return raw.decode("utf-8", errors="replace"), False

    def _location_for_span(self, path: Path, text: str, span: tuple[int, int]) -> Location:
        start_offset, end_offset = span
        line_starts = self._line_starts(text)
        start_line, start_col = self._line_col_for_offset(line_starts, start_offset)
        end_line, end_col = self._line_col_for_offset(line_starts, end_offset)
        relative_path = self._relative_path(path)
        if relative_path is None:
            raise InvalidSearchPath("matched path escaped workspace")
        return Location(
            path=relative_path,
            range=Range(start_line=start_line, start_col=start_col, end_line=end_line, end_col=end_col),
        )

    @staticmethod
    def _line_starts(text: str) -> list[int]:
        return [0] + [index + 1 for index, char in enumerate(text) if char == "\n"]

    @staticmethod
    def _line_col_for_offset(line_starts: list[int], offset: int) -> tuple[int, int]:
        line = max(0, bisect_right(line_starts, offset) - 1)
        return line, offset - line_starts[line]


__all__ = [
    "InvalidSearchPath",
    "InvalidSearchPattern",
    "TextSearchLocations",
    "TextSearchProvider",
]
