"""Async symbol index package."""

from .context import IndexBackedCodeContext
from .indexer import ScanStatus, SymbolExtractor, SymbolIndexer, WorkspaceIndexResult
from .invalidation import (
    CurrentFileMetadata,
    IndexedFileMetadata,
    InvalidationAction,
    InvalidationDecision,
    InvalidationReason,
    decide_file_invalidation,
)
from .resolver import (
    IndexBackedTargetResolver,
    ResolvedTarget,
    SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME,
)
from .store import (
    CurrentFileForStore,
    SCHEMA_VERSION,
    SymbolHistoryEntry,
    SymbolIndexStore,
    default_symbol_index_path,
)

__all__ = [
    "CurrentFileForStore",
    "CurrentFileMetadata",
    "IndexBackedCodeContext",
    "IndexBackedTargetResolver",
    "IndexedFileMetadata",
    "InvalidationAction",
    "InvalidationDecision",
    "InvalidationReason",
    "SCHEMA_VERSION",
    "ScanStatus",
    "SymbolExtractor",
    "SYMBOL_ID_RECOVERED_VIA_QUALIFIED_NAME",
    "SymbolHistoryEntry",
    "SymbolIndexStore",
    "ResolvedTarget",
    "SymbolIndexer",
    "WorkspaceIndexResult",
    "decide_file_invalidation",
    "default_symbol_index_path",
]
