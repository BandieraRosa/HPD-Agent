"""Tests for the async LangChain code_intel tools with explicit fake providers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine, Generator
from typing import Protocol, TypeVar, cast

from langchain_core.tools import BaseTool


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...

import pytest

from src.code_intel import CodeIntelKernel
from src.code_intel.providers.fake import PYTHON_FAKE_PATH, create_fake_providers, fake_symbols
from src.code_intel.core import Capability, ConfidenceClass, Diagnostic, ProviderHealth, ProviderStatus, ProviderUnavailable
from src.code_intel.verifier import clear_baseline_cache
from src.code_intel.tools import (
    code_context,
    code_outline,
    code_search,
    code_semantic,
    code_verify,
    set_code_intel_kernel,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _payload(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(raw))


def _data(raw: str) -> dict[str, object]:
    payload = _payload(raw)
    assert payload["ok"] is True
    return cast(dict[str, object], payload["data"])


@pytest.fixture(autouse=True)
def explicit_fake_kernel() -> Generator[None, None, None]:
    clear_baseline_cache()
    set_code_intel_kernel(CodeIntelKernel(create_fake_providers()))
    yield
    set_code_intel_kernel(None)
    clear_baseline_cache()


def test_default_kernel_remains_provider_free_until_explicitly_injected() -> None:
    set_code_intel_kernel(None)

    raw = _run(_ainvoke_text(code_outline, {"path": PYTHON_FAKE_PATH}))
    payload = _payload(raw)

    assert payload["ok"] is False
    error = cast(dict[str, object], payload["error"])
    assert error["code"] == "unsupported_language"
    assert "当前语言" in cast(str, error["message"])


def test_code_search_symbol_text_and_mixed_modes_return_tool_result_json() -> None:
    symbol_data = _data(_run(_ainvoke_text(code_search, {"query": "fake", "mode": "symbol", "limit": 10})))
    text_data = _data(_run(_ainvoke_text(code_search, {"query": "fake", "mode": "text", "limit": 10})))
    mixed_data = _data(_run(_ainvoke_text(code_search, {"query": "fake", "mode": "mixed", "limit": 10})))

    symbol_matches = cast(list[dict[str, object]], symbol_data["matches"])
    text_matches = cast(list[dict[str, object]], text_data["matches"])
    mixed_matches = cast(list[dict[str, object]], mixed_data["matches"])

    assert [match["name"] for match in symbol_matches] == ["FakeService", "run", "fakeClient"]
    assert {match["kind"] for match in text_matches} == {"text"}
    assert len(mixed_matches) >= len(symbol_matches)
    assert all(isinstance(match["snippet"], str) and match["snippet"] for match in mixed_matches)


def test_code_outline_returns_file_symbols_and_stable_line_count_estimate() -> None:
    data = _data(_run(_ainvoke_text(code_outline, {"path": PYTHON_FAKE_PATH, "max_depth": 3})))
    symbols = cast(list[dict[str, object]], data["symbols"])

    assert data["path"] == PYTHON_FAKE_PATH
    assert data["language"] == "python"
    assert [symbol["name"] for symbol in symbols] == ["FakeService", "run", "helper"]
    assert data["line_count"] == 19


def test_code_context_accepts_dict_target_and_include_aliases() -> None:
    run_symbol = next(symbol for symbol in fake_symbols() if symbol.qualified_name == "FakeService.run")

    data = _data(
        _run(
            _ainvoke_text(code_context, 
                {
                    "target": {"symbol_id": run_symbol.id},
                    "include": ["signature", "body", "parents", "imports", "nearby_symbols"],
                    "max_tokens": 256,
                }
            )
        )
    )

    target_symbol = cast(dict[str, object], data["target_symbol"])
    parents = cast(list[dict[str, object]], data["parents"])
    nearby_symbols = cast(list[dict[str, object]], data["nearby_symbols"])

    assert target_symbol["qualified_name"] == "FakeService.run"
    assert data["signature"] == "def run(self, value: str) -> str"
    assert "helper(value)" in cast(str, data["body"])
    assert [parent["name"] for parent in parents] == ["FakeService"]
    assert data["imports"] == ["from __future__ import annotations"]
    assert {symbol["name"] for symbol in nearby_symbols} == {"FakeService", "helper"}
    assert data["truncated"] is False


def test_code_semantic_routes_operations_and_groups_references_by_file() -> None:
    run_symbol = next(symbol for symbol in fake_symbols() if symbol.qualified_name == "FakeService.run")
    symbol_target = {"symbol_id": run_symbol.id}
    file_target = {"anchor": {"path": PYTHON_FAKE_PATH, "symbol_name": "FakeService"}}

    definition = _data(_run(_ainvoke_text(code_semantic, {"operation": "definition", "target": symbol_target})))
    references = _data(_run(_ainvoke_text(code_semantic, {"operation": "references", "target": symbol_target})))
    hover = _data(_run(_ainvoke_text(code_semantic, {"operation": "hover", "target": symbol_target})))
    document_symbols = _data(
        _run(_ainvoke_text(code_semantic, {"operation": "document_symbols", "target": file_target, "max_results": 10}))
    )

    definition_locations = cast(list[dict[str, object]], definition["locations"])
    reference_locations = cast(list[dict[str, object]], references["locations"])
    grouped = cast(dict[str, list[dict[str, object]]], references["grouped_by_file"])
    hover_info = cast(dict[str, object], hover["hover"])
    symbols = cast(list[dict[str, object]], document_symbols["document_symbols"])

    assert definition["operation"] == "definition"
    assert definition_locations[0]["path"] == PYTHON_FAKE_PATH
    assert references["operation"] == "references"
    assert len(reference_locations) == 2
    assert list(grouped) == [PYTHON_FAKE_PATH]
    assert len(grouped[PYTHON_FAKE_PATH]) == 2
    assert "def run" in cast(str, hover_info["contents"])
    assert [symbol["name"] for symbol in symbols] == ["FakeService", "run", "helper"]


def test_code_verify_uses_diagnostics_for_explicit_paths_with_agent_call_source() -> None:
    data = _data(
        _run(
            _ainvoke_text(code_verify, 
                {
                    "scope": "file",
                    "paths": [PYTHON_FAKE_PATH],
                    "checks": ["lsp_diagnostics", "tests", "lint"],
                    "baseline": True,
                }
            )
        )
    )

    unchanged = cast(list[dict[str, object]], data["unchanged_diagnostics"])
    skipped = cast(list[dict[str, object]], data["checks_skipped"])

    assert data["ok"] is True
    assert data["call_source"] == "agent"
    assert data["checks_run"] == ["lsp_diagnostics"]
    assert unchanged[0]["code"] == "fake-warning"
    assert unchanged[0]["severity"] == "warning"
    assert data["new_diagnostics"] == []
    assert data["severity_delta"] == {"error": 0, "warning": 0, "info": 0, "hint": 0}
    skipped_checks = {item["check"] for item in skipped}
    assert skipped_checks >= {"tests", "lint"}
    assert "baseline" not in skipped_checks
    assert data["recommended_next_action"] == "proceed"
    assert data["verification_status"] == "success"
    assert isinstance(data["baseline_key"], str) and data["baseline_key"]
    assert data["baseline_refreshed"] is True


def test_code_verify_provider_unavailable_returns_structured_partial() -> None:
    """When the diagnostics provider raises ProviderUnavailable, code_verify
    must return structured partial output, not success and not a traceback.

    We inject a kernel whose provider supports DIAGNOSTICS but raises
    ProviderUnavailable when called, triggering the partial path in code_verify.
    """
    class _FailingFakeProvider:
        name: str = "failing_fake"
        capabilities: set[Capability] = {Capability.DIAGNOSTICS}
        languages: set[str] = {"python"}

        async def supports(self, capability: Capability, language: str) -> bool:
            return capability in self.capabilities and language in self.languages

        async def health(self) -> ProviderHealth:
            return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

        async def confidence_for(self, _capability: Capability, _language: str) -> ConfidenceClass:
            return ConfidenceClass.HIGH

        async def diagnostics(self, path: str) -> list[Diagnostic]:
            _ = path
            raise ProviderUnavailable("diagnostics unavailable for test")

    kernel_failing_diag = CodeIntelKernel((_FailingFakeProvider(),))
    set_code_intel_kernel(kernel_failing_diag)

    raw = _run(_ainvoke_text(code_verify, {"scope": "file", "paths": [PYTHON_FAKE_PATH]}))
    payload = _payload(raw)

    # Tool-level ok should be True (tool didn't crash)
    assert payload["ok"] is True

    data = cast(dict[str, object], payload["data"])

    # Verification status must be partial, not success, not blocked
    assert data["verification_status"] == "partial"
    assert data["recommended_next_action"] == "abort"
    assert data["ok"] is False  # verification-level ok
    assert data["call_source"] == "agent"

    # No diagnostics should be present
    assert data["new_diagnostics"] == []
    assert data["resolved_diagnostics"] == []
    assert data["unchanged_diagnostics"] == []

    # Provider error should be reported
    provider_error = cast(dict[str, object] | None, data.get("provider_error"))
    assert provider_error is not None, "provider_error must be populated"
    assert provider_error["code"] == "provider_unavailable"

    # Checks skipped should include lsp_diagnostics
    checks_skipped = cast(list[dict[str, object]], data["checks_skipped"])
    skipped_check_names = {item["check"] for item in checks_skipped}
    assert "lsp_diagnostics" in skipped_check_names
