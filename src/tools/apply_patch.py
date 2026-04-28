# pyright: reportUnknownVariableType=false
"""Safer patch tool scaffold.

This module defines the LangChain tool interface for a safer patch tool. The
v1 parser, validator, dry-run planner, and filesystem application logic will be
filled in by later tasks.
"""

from dataclasses import dataclass, replace
import re
from typing import Literal

from langchain_core.tools import tool


BEGIN_MARKER = "*** Begin Patch"
END_MARKER = "*** End Patch"
EOF_MARKER = "\\ No newline at end of file"

NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
INVALID_PATCH = "INVALID_PATCH"
INVALID_ENVELOPE = "INVALID_ENVELOPE"
MALFORMED_SECTION = "MALFORMED_SECTION"
MALFORMED_HUNK = "MALFORMED_HUNK"
DUPLICATE_FILE_OPERATION = "DUPLICATE_FILE_OPERATION"
UNSAFE_PATH = "UNSAFE_PATH"
APPLY_FAILED = "APPLY_FAILED"

ERROR_CODES = (
    NOT_IMPLEMENTED,
    INVALID_PATCH,
    INVALID_ENVELOPE,
    MALFORMED_SECTION,
    MALFORMED_HUNK,
    DUPLICATE_FILE_OPERATION,
    UNSAFE_PATH,
    APPLY_FAILED,
)

OperationKind = Literal["add", "update", "delete"]
HunkLineKind = Literal["context", "add", "delete"]

_SECTION_RE = re.compile(r"^\*\*\* (Add File|Update File|Delete File): (.+)$")
_HUNK_RE = re.compile(r"^@@ -([0-9]+),([0-9]+) \+([0-9]+),([0-9]+) @@$")


@dataclass(frozen=True)
class PatchHunkLine:
    kind: HunkLineKind
    content: str
    no_newline_at_end: bool = False


@dataclass(frozen=True)
class PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]


@dataclass(frozen=True)
class PatchOperation:
    kind: OperationKind
    path: str
    added_lines: tuple[PatchHunkLine, ...] = ()
    hunks: tuple[PatchHunk, ...] = ()


@dataclass(frozen=True)
class PatchDocument:
    operations: tuple[PatchOperation, ...]


class PatchError(ValueError):
    """Patch parser error with a stable machine-readable code."""

    code: str
    message: str
    line: int | None

    def __init__(self, code: str, message: str, *, line: int | None = None) -> None:
        self.code = code
        self.message = message
        self.line = line
        super().__init__(self.display_message)

    @property
    def display_message(self) -> str:
        if self.line is None:
            return self.message
        return f"{self.message} (line {self.line})"

    def to_error_result(self) -> str:
        return _format_error(self.code, self.display_message)


def _format_error(code: str, detail: str) -> str:
    return f"[Error] {code}: {detail}"


def parse_patch_text(patch_text: str) -> PatchDocument:
    """Parse a custom v1 patch envelope into typed operations."""
    lines = patch_text.splitlines()
    if not lines or lines[0] != BEGIN_MARKER:
        raise PatchError(INVALID_ENVELOPE, "patch must start with the begin marker", line=1)

    end_index = _find_end_marker(lines)
    for offset, trailing_line in enumerate(lines[end_index + 1 :], start=end_index + 2):
        if trailing_line.strip():
            raise PatchError(
                INVALID_ENVELOPE,
                "only whitespace is allowed after the end marker",
                line=offset,
            )

    body = lines[1:end_index]
    operations: list[PatchOperation] = []
    seen_paths: set[str] = set()
    index = 0

    while index < len(body):
        line = body[index]
        line_number = index + 2
        if _is_blank_separator(line):
            index += 1
            continue

        section_match = _SECTION_RE.match(line)
        if section_match is None:
            raise PatchError(MALFORMED_SECTION, "expected a file operation section", line=line_number)

        section_name = section_match.group(1)
        raw_path = section_match.group(2)
        if not raw_path.strip():
            raise PatchError(MALFORMED_SECTION, "section path must not be empty", line=line_number)
        if raw_path in seen_paths:
            raise PatchError(
                DUPLICATE_FILE_OPERATION,
                "patch contains repeated file operation path",
                line=line_number,
            )
        seen_paths.add(raw_path)

        index += 1
        if section_name == "Add File":
            operation, index = _parse_add_file(raw_path, body, index)
        elif section_name == "Update File":
            operation, index = _parse_update_file(raw_path, body, index)
        else:
            operation, index = _parse_delete_file(raw_path, body, index)
        operations.append(operation)

    if not operations:
        raise PatchError(INVALID_ENVELOPE, "patch must contain at least one file operation")

    return PatchDocument(operations=tuple(operations))


def _find_end_marker(lines: list[str]) -> int:
    for index, line in enumerate(lines[1:], start=1):
        if line == END_MARKER:
            return index
    raise PatchError(INVALID_ENVELOPE, "patch is missing the end marker")


def _parse_add_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    added_lines: list[PatchHunkLine] = []

    for index in range(body_start, body_end):
        line = lines[index]
        if _is_blank_separator(line) or not line.startswith("+"):
            raise PatchError(
                MALFORMED_SECTION,
                "Add File body lines must start with '+'",
                line=index + 2,
            )
        added_lines.append(PatchHunkLine(kind="add", content=line[1:]))

    return (
        PatchOperation(kind="add", path=path, added_lines=tuple(added_lines)),
        end_index,
    )


def _parse_update_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    if body_start == body_end:
        raise PatchError(MALFORMED_SECTION, "Update File requires at least one hunk")

    hunks: list[PatchHunk] = []
    index = body_start
    while index < body_end:
        line = lines[index]
        if _is_blank_separator(line):
            index += 1
            continue
        if _HUNK_RE.match(line) is None:
            raise PatchError(MALFORMED_HUNK, "expected a well-formed hunk header", line=index + 2)
        hunk, index = _parse_hunk(lines, index, body_end)
        hunks.append(hunk)

    if not hunks:
        raise PatchError(MALFORMED_SECTION, "Update File requires at least one hunk")

    return PatchOperation(kind="update", path=path, hunks=tuple(hunks)), end_index


def _parse_delete_file(
    path: str,
    lines: list[str],
    start_index: int,
) -> tuple[PatchOperation, int]:
    end_index = _find_next_section(lines, start_index)
    body_start, body_end = _trim_blank_separators(lines, start_index, end_index)
    if body_start != body_end:
        raise PatchError(MALFORMED_SECTION, "Delete File section does not accept a body", line=body_start + 2)
    return PatchOperation(kind="delete", path=path), end_index


def _parse_hunk(lines: list[str], header_index: int, section_end: int) -> tuple[PatchHunk, int]:
    header = lines[header_index]
    match = _HUNK_RE.match(header)
    if match is None:
        raise PatchError(MALFORMED_HUNK, "malformed hunk header", line=header_index + 2)

    old_start = int(match.group(1))
    old_count = int(match.group(2))
    new_start = int(match.group(3))
    new_count = int(match.group(4))
    old_seen = 0
    new_seen = 0
    hunk_lines: list[PatchHunkLine] = []
    last_body_line_index: int | None = None
    index = header_index + 1

    while index < section_end:
        line = lines[index]
        if line == EOF_MARKER:
            if not hunk_lines or last_body_line_index != index - 1:
                raise PatchError(
                    MALFORMED_HUNK,
                    "EOF marker must immediately follow a hunk body line",
                    line=index + 2,
                )
            hunk_lines[-1] = replace(hunk_lines[-1], no_newline_at_end=True)
            index += 1
            continue

        if old_seen == old_count and new_seen == new_count:
            break

        if not line or line[0] not in " +-":
            raise PatchError(
                MALFORMED_HUNK,
                "hunk body lines must start with space, '-', or '+'",
                line=index + 2,
            )

        prefix = line[0]
        content = line[1:]
        if prefix == " ":
            old_seen += 1
            new_seen += 1
            kind: HunkLineKind = "context"
        elif prefix == "-":
            old_seen += 1
            kind = "delete"
        else:
            new_seen += 1
            kind = "add"

        if old_seen > old_count or new_seen > new_count:
            raise PatchError(MALFORMED_HUNK, "hunk body exceeds header counts", line=index + 2)

        hunk_lines.append(PatchHunkLine(kind=kind, content=content))
        last_body_line_index = index
        index += 1

    if old_seen != old_count or new_seen != new_count:
        raise PatchError(MALFORMED_HUNK, "hunk body does not match header counts", line=header_index + 2)

    return (
        PatchHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=tuple(hunk_lines),
        ),
        index,
    )


def _find_next_section(lines: list[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        if lines[index].startswith("*** "):
            return index
        index += 1
    return index


def _trim_blank_separators(lines: list[str], start_index: int, end_index: int) -> tuple[int, int]:
    while start_index < end_index and _is_blank_separator(lines[start_index]):
        start_index += 1
    while end_index > start_index and _is_blank_separator(lines[end_index - 1]):
        end_index -= 1
    return start_index, end_index


def _is_blank_separator(line: str) -> bool:
    return not line.strip()


def _placeholder_result(patch_text: str, dry_run: bool) -> str:
    mode = "dry-run" if dry_run else "apply"
    detail = (
        f"apply_patch {mode} mode is not implemented yet; "
        + f"received {len(patch_text)} characters of patch text."
    )
    return _format_error(NOT_IMPLEMENTED, detail)


@tool
def apply_patch(patch_text: str, dry_run: bool = False) -> str:
    """Apply a patch safely to the repository filesystem.

    Args:
        patch_text: Patch text to validate and apply in a future implementation.
        dry_run: If True, future versions will report planned changes only.
    """
    return _placeholder_result(patch_text=patch_text, dry_run=dry_run)


__all__ = [
    "apply_patch",
    "parse_patch_text",
    "PatchDocument",
    "PatchOperation",
    "PatchHunk",
    "PatchHunkLine",
    "PatchError",
    "NOT_IMPLEMENTED",
    "INVALID_PATCH",
    "INVALID_ENVELOPE",
    "MALFORMED_SECTION",
    "MALFORMED_HUNK",
    "DUPLICATE_FILE_OPERATION",
    "UNSAFE_PATH",
    "APPLY_FAILED",
    "ERROR_CODES",
]
