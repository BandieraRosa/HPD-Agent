"""Deterministic invalidation decisions for the symbol index."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InvalidationAction(str, Enum):
    """Index action selected for a file metadata comparison."""

    REUSE = "reuse"
    REBUILD = "rebuild"
    DELETE = "delete"
    REBUILD_ALL = "rebuild_all"


class InvalidationReason(str, Enum):
    """Machine-readable reason for an invalidation decision."""

    UNCHANGED = "unchanged"
    NEW_FILE = "new_file"
    DELETED = "deleted"
    LANGUAGE_CHANGED = "language_changed"
    CONTENT_HASH_CHANGED = "content_hash_changed"
    GRAMMAR_VERSION_CHANGED = "grammar_version_changed"
    QUERY_VERSION_CHANGED = "query_version_changed"
    SCHEMA_VERSION_CHANGED = "schema_version_changed"


@dataclass(frozen=True)
class IndexedFileMetadata:
    """Metadata persisted for an indexed workspace file."""

    path: str
    language: str
    sha256: str
    mtime: float
    size: int
    indexed_at: float
    grammar_version: str
    query_version: str
    schema_version: str


@dataclass(frozen=True)
class CurrentFileMetadata:
    """Current filesystem metadata used for correctness decisions."""

    path: str
    language: str
    sha256: str
    mtime: float
    size: int
    grammar_version: str
    query_version: str
    schema_version: str
    exists: bool = True


@dataclass(frozen=True)
class InvalidationDecision:
    """Action and reason returned by invalidation checks."""

    action: InvalidationAction
    reason: InvalidationReason

    @property
    def should_reuse(self) -> bool:
        return self.action == InvalidationAction.REUSE

    @property
    def should_rebuild(self) -> bool:
        return self.action in {
            InvalidationAction.REBUILD,
            InvalidationAction.REBUILD_ALL,
        }


def decide_file_invalidation(
    previous: IndexedFileMetadata | None,
    current: CurrentFileMetadata | None,
) -> InvalidationDecision:
    """Compare stored and current metadata without trusting mtime for correctness."""
    if current is None or not current.exists:
        if previous is None:
            return InvalidationDecision(
                InvalidationAction.REUSE, InvalidationReason.UNCHANGED
            )
        return InvalidationDecision(
            InvalidationAction.DELETE, InvalidationReason.DELETED
        )

    if previous is None:
        return InvalidationDecision(
            InvalidationAction.REBUILD, InvalidationReason.NEW_FILE
        )

    if previous.schema_version != current.schema_version:
        return InvalidationDecision(
            InvalidationAction.REBUILD_ALL, InvalidationReason.SCHEMA_VERSION_CHANGED
        )
    if previous.grammar_version != current.grammar_version:
        return InvalidationDecision(
            InvalidationAction.REBUILD_ALL, InvalidationReason.GRAMMAR_VERSION_CHANGED
        )
    if previous.query_version != current.query_version:
        return InvalidationDecision(
            InvalidationAction.REBUILD_ALL, InvalidationReason.QUERY_VERSION_CHANGED
        )
    if previous.language != current.language:
        return InvalidationDecision(
            InvalidationAction.REBUILD, InvalidationReason.LANGUAGE_CHANGED
        )
    if previous.sha256 != current.sha256:
        return InvalidationDecision(
            InvalidationAction.REBUILD, InvalidationReason.CONTENT_HASH_CHANGED
        )
    return InvalidationDecision(InvalidationAction.REUSE, InvalidationReason.UNCHANGED)


__all__ = [
    "CurrentFileMetadata",
    "IndexedFileMetadata",
    "InvalidationAction",
    "InvalidationDecision",
    "InvalidationReason",
    "decide_file_invalidation",
]
