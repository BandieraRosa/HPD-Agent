"""Contracts for safe code_intel trace metadata redaction."""

from __future__ import annotations

from src.code_intel.core import ProviderUnavailable
from src.code_intel.tracing import ALLOWED_TRACE_FIELDS, redact, safe_error_metadata


def test_trace_metadata_redaction() -> None:
    """trace metadata 必须通过白名单过滤，不能泄漏源码或绝对路径。"""
    raw = {
        "elapsed_ms": 12,
        "provider_name": "fake_provider",
        "capability": "diagnostics",
        "language": "python",
        "cache_hit": True,
        "fallback_chain": ["lsp", "tree_sitter"],
        "result_count": 2,
        "truncated": False,
        "path": "src/app.py",
        "paths": ["src/app.py", "/home/user/project/src/secret.py", "../escape.py"],
        "source_body": "def leak_secret():\n    return TOKEN",
        "symbol_signature": "def leak_secret(token: str) -> str",
        "lsp_request": {"textDocument": {"uri": "file:///home/user/project/src/secret.py"}},
        "diagnostic_messages": ["Cannot find name 'TOP_SECRET' at line 42"],
        "error": {"code": "provider_unavailable", "message": "raw /home/user traceback"},
    }

    metadata = redact(raw, schema=ALLOWED_TRACE_FIELDS)

    assert metadata["elapsed_ms"] == 12
    assert metadata["provider_name"] == "fake_provider"
    assert metadata["path"] == "src/app.py"
    assert metadata["paths"] == ["src/app.py"]
    assert metadata["fallback_chain"] == ["lsp", "tree_sitter"]
    assert metadata["diagnostic_templates"] == ["Cannot find name <quoted> at line <line>"]
    assert metadata["error"] == {"code": "provider_unavailable", "message": "代码智能服务发生错误。"}

    serialized = repr(metadata)
    assert "source_body" not in metadata
    assert "symbol_signature" not in metadata
    assert "lsp_request" not in metadata
    assert "TOP_SECRET" not in serialized
    assert "/home/user" not in serialized
    assert "def leak_secret" not in serialized


def test_safe_error_metadata_uses_chinese_generic_tool_error() -> None:
    metadata = safe_error_metadata(ProviderUnavailable("raw provider traceback /tmp/secret"))

    assert metadata == {"error": {"code": "provider_unavailable", "message": "代码智能提供方暂时不可用。"}}
