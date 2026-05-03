"""Language Server Protocol provider package."""

from .client import LSPClient, LSP_SYMBOL_CONFIDENCE, LSP_SYMBOL_INDEX_VERSION, LSP_SYMBOL_SOURCE
from .manager import LSPClientFactory, LSPManager, LSPManagerKey, LSPServerHandle, ManagedLSPClient
from .provider import LSPProvider
from .registry import (
    DEFAULT_LANGUAGE_SERVER_SPECS,
    LanguageServerSpec,
    default_language_server_specs,
    language_server_specs,
)
from .transport import LSPTransport, NotificationHandler

__all__ = [
    "DEFAULT_LANGUAGE_SERVER_SPECS",
    "LSPClient",
    "LSPClientFactory",
    "LSPManager",
    "LSPManagerKey",
    "LSPProvider",
    "LSPServerHandle",
    "LSPTransport",
    "LSP_SYMBOL_CONFIDENCE",
    "LSP_SYMBOL_INDEX_VERSION",
    "LSP_SYMBOL_SOURCE",
    "LanguageServerSpec",
    "ManagedLSPClient",
    "NotificationHandler",
    "default_language_server_specs",
    "language_server_specs",
]
