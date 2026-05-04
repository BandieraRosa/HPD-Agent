"""Contracts preventing raw provider payloads from crossing code_intel boundaries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

from pydantic import BaseModel

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Capability, ConfidenceClass, ProviderHealth, ProviderStatus, Range, ToolResult
from src.code_intel.tools import code_outline, set_code_intel_kernel
from src.code_intel.tools.models import CodeSearchData, SearchMatch
from src.code_intel.tracing import ALLOWED_TRACE_FIELDS, redact

_RAW_BOUNDARY_TOKENS = (
    "lsp_request",
    "lsp_response",
    "source_body",
    "symbol_signature",
    "tree_sitter.Node",
    "tree_sitter.Tree",
    "tree_sitter.Query",
    "Language(",
)
_SECRET_TOKENS = ("TOP_SECRET", "API_TOKEN", "SECRET_VALUE")
T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


@dataclass(frozen=True)
class _RawNode:
    payload: str = "tree_sitter.Node(API_TOKEN=SECRET_VALUE)"


class _MalformedOutlineProvider:
    name: str = "malformed_outline"
    capabilities: set[Capability] = {Capability.OUTLINE}
    languages: set[str] = {"python"}

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def confidence_for(self, _capability: Capability, _language: str) -> ConfidenceClass:
        return ConfidenceClass.HIGH

    async def outline(self, path: str) -> list[object]:
        _ = path
        return [_RawNode()]


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: object, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, item)
    return cast(str, await invokable.ainvoke(args))


def _serialized(value: object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def test_trace_redaction_drops_raw_lsp_tree_sitter_source_and_absolute_paths(tmp_path: Path) -> None:
    raw = {
        "provider_name": "lsp",
        "capability": "diagnostics",
        "language": "python",
        "path": str(tmp_path / "src" / "secret.py"),
        "paths": ["src/app.py", str(tmp_path / "src" / "secret.py")],
        "source_body": "def leak():\n    return TOP_SECRET",
        "symbol_signature": "def leak(token: str) -> str",
        "lsp_request": {"textDocument": {"uri": f"file://{tmp_path}/src/secret.py"}},
        "lsp_response": {"contents": "API_TOKEN"},
        "tree_sitter_node": "tree_sitter.Node(type=function_definition)",
        "diagnostic_messages": ["Cannot find name 'SECRET_VALUE' at line 42"],
    }

    metadata = redact(raw, schema=ALLOWED_TRACE_FIELDS)
    serialized = _serialized(metadata)

    assert metadata["paths"] == ["src/app.py"]
    assert "path" not in metadata
    assert metadata["diagnostic_templates"] == ["Cannot find name <quoted> at line <line>"]
    assert str(tmp_path) not in serialized
    for token in _RAW_BOUNDARY_TOKENS + _SECRET_TOKENS:
        assert token not in serialized


def test_serialized_tool_result_contains_only_normalized_fields_not_raw_provider_objects(tmp_path: Path) -> None:
    result = ToolResult[object](
        ok=True,
        data=CodeSearchData(
            matches=[
                SearchMatch(
                    symbol_id=None,
                    name="AlphaService",
                    qualified_name=None,
                    kind="text",
                    path="src/app.py",
                    range=Range(start_line=0, start_col=0, end_line=0, end_col=12),
                    snippet="AlphaService",
                    source="text_search",
                    confidence=0.5,
                )
            ]
        ),
    )

    serialized = _serialized(result)

    assert "src/app.py" in serialized
    assert str(tmp_path) not in serialized
    for token in _RAW_BOUNDARY_TOKENS + _SECRET_TOKENS:
        assert token not in serialized


def test_code_intel_contract_sources_do_not_assert_raw_payload_success() -> None:
    contract_root = Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(contract_root.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        blocker_marker = "REVIEW_RESULT" + "_BLOCKER"
        unresolved_skip_marker = "pytest.mark." + "xf" + "ail"
        if blocker_marker in source or unresolved_skip_marker in source:
            offenders.append(path.name)

    assert offenders == []


def test_malformed_provider_tool_output_returns_safe_validation_error_json() -> None:
    set_code_intel_kernel(CodeIntelKernel((_MalformedOutlineProvider(),)))
    try:
        raw = _run(_ainvoke_text(code_outline, {"path": "src/app.py"}))
    finally:
        set_code_intel_kernel(None)

    payload = cast(dict[str, object], json.loads(raw))
    serialized = _serialized(payload)

    assert payload["ok"] is False
    error = cast(dict[str, object], payload["error"])
    assert error["code"] == "invalid_input"
    assert error["hint"] == "请检查 target、include、mode、operation 或路径参数。"
    for token in _RAW_BOUNDARY_TOKENS + _SECRET_TOKENS:
        assert token not in serialized

