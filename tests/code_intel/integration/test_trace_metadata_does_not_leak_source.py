"""Regression tests proving trace metadata cannot leak source content."""

from __future__ import annotations

import json

from src.code_intel.tracing import trace_span
from src.core.observability import TraceRecord, get_tracer


def _start_trace() -> None:
    tracer = get_tracer()
    _ = tracer.end_trace()
    _ = tracer.start_trace(query="redaction leak test", session_id="test")


def _end_trace() -> TraceRecord:
    record = get_tracer().end_trace()
    assert record is not None
    return record


def test_trace_metadata_does_not_leak_source_body_paths_or_lsp_payloads() -> None:
    _start_trace()
    with trace_span(
        "code_intel.test.redaction",
        {
            "provider_name": "lsp",
            "capability": "diagnostics",
            "language": "python",
            "path": "/home/user/project/src/secret.py",
            "paths": ["src/safe.py", "/home/user/project/src/secret.py"],
            "source_body": "def secret():\n    return API_TOKEN",
            "symbol_signature": "def secret(api_key: str) -> str",
            "hover_contents": "api_key: TOP_SECRET",
            "lsp_request": {"params": {"text": "API_TOKEN", "uri": "file:///home/user/project/src/secret.py"}},
            "lsp_response": {"contents": "def secret(api_key: str) -> str"},
            "diagnostic_messages": ["Argument of type 'TOP_SECRET' cannot be assigned to parameter 'api_key' on line 7"],
            "error": {"code": "provider_unavailable", "message": "Traceback /home/user/project/src/secret.py API_TOKEN"},
        },
    ):
        pass
    record = _end_trace()

    metadata = record.spans[0].metadata
    serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    assert metadata["paths"] == ["src/safe.py"]
    assert "path" not in metadata
    assert metadata["error"] == {"code": "provider_unavailable", "message": "代码智能服务发生错误。"}
    assert metadata["diagnostic_templates"] == [
        "Argument of type <quoted> cannot be assigned to parameter <quoted> on line <line>"
    ]
    assert "API_TOKEN" not in serialized
    assert "TOP_SECRET" not in serialized
    assert "api_key" not in serialized
    assert "/home/user" not in serialized
    assert "def secret" not in serialized
    assert "lsp_request" not in serialized
    assert "lsp_response" not in serialized
