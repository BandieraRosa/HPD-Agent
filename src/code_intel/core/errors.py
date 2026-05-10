"""Typed code intelligence exceptions and ToolError mapping."""

from __future__ import annotations

from typing import ClassVar

from .models import ToolError


class CodeIntelError(Exception):
    """Base exception that maps provider/internal failures to safe tool errors."""

    code: ClassVar[str] = "code_intel_error"
    message: ClassVar[str] = "代码智能服务发生错误。"
    hint: ClassVar[str | None] = "请稍后重试，或使用 read_file 查看源码。"

    def __init__(self, detail: str | None = None) -> None:
        self.detail: str | None = detail
        super().__init__(detail or self.message)

    def to_tool_error(self) -> ToolError:
        """Return the LLM-facing error without leaking raw provider details."""
        return ToolError(code=self.code, message=self.message, hint=self.hint)


class LanguageNotSupported(CodeIntelError):
    code: ClassVar[str] = "unsupported_language"
    message: ClassVar[str] = "当前语言不受支持。"
    hint: ClassVar[str | None] = "请跳过本文件，或使用 read_file 直接查看源码。"


class ProviderUnavailable(CodeIntelError):
    code: ClassVar[str] = "provider_unavailable"
    message: ClassVar[str] = "代码智能提供方暂时不可用。"
    hint: ClassVar[str | None] = "请稍后重试，或改用基础文件读取和搜索。"


class IndexStale(CodeIntelError):
    code: ClassVar[str] = "index_stale"
    message: ClassVar[str] = "代码索引已过期。"
    hint: ClassVar[str | None] = "请等待系统重建索引后重试。"


class SymbolNotFound(CodeIntelError):
    code: ClassVar[str] = "symbol_not_found"
    message: ClassVar[str] = "未找到指定符号。"
    hint: ClassVar[str | None] = "symbol 已被修改或删除，请用 code_search 重定位。"


class LSPTimeout(CodeIntelError):
    code: ClassVar[str] = "lsp_timeout"
    message: ClassVar[str] = "语言服务器响应超时。"
    hint: ClassVar[str | None] = "可降级到 tree-sitter，或稍后重试。"


__all__ = [
    "CodeIntelError",
    "IndexStale",
    "LSPTimeout",
    "LanguageNotSupported",
    "ProviderUnavailable",
    "SymbolNotFound",
]
