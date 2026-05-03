"""Unit tests for CodeIntelKernel provider routing."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

import pytest

from src.code_intel import CodeIntelKernel
from src.code_intel.core import (
    Capability,
    CodeIntelError,
    ConfidenceClass,
    LSPTimeout,
    ProviderHealth,
    ProviderStatus,
    ProviderUnavailable,
)


class RoutingProvider:
    def __init__(
        self,
        *,
        name: str,
        capabilities: set[Capability],
        languages: set[str],
        confidence: ConfidenceClass,
        health_score: float,
        result: str,
        status: ProviderStatus = ProviderStatus.HEALTHY,
        error: Exception | None = None,
    ) -> None:
        self.name: str = name
        self.capabilities: set[Capability] = capabilities
        self.languages: set[str] = languages
        self.confidence_class: ConfidenceClass = confidence
        self._health: ProviderHealth = ProviderHealth(status=status, health_score=health_score)
        self._result: str = result
        self._error: Exception | None = error
        self.support_calls: list[tuple[Capability, str]] = []
        self.health_calls: int = 0
        self.outline_calls: list[str] = []

    async def supports(self, capability: Capability, language: str) -> bool:
        self.support_calls.append((capability, language))
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        self.health_calls += 1
        return self._health

    async def outline(self, path: str) -> str:
        self.outline_calls.append(path)
        if self._error is not None:
            raise self._error
        return self._result


T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def test_routes_by_capability_language_health_confidence() -> None:
    medium_healthy = RoutingProvider(
        name="medium-healthy",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.MEDIUM,
        health_score=1.0,
        result="medium",
    )
    high_degraded = RoutingProvider(
        name="high-degraded",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.HIGH,
        health_score=0.25,
        result="high-degraded",
    )
    high_healthy = RoutingProvider(
        name="high-healthy",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.HIGH,
        health_score=0.9,
        result="high-healthy",
    )
    wrong_capability = RoutingProvider(
        name="wrong-capability",
        capabilities={Capability.SYMBOL_SEARCH},
        languages={"python"},
        confidence=ConfidenceClass.HIGH,
        health_score=1.0,
        result="wrong-capability",
    )
    wrong_language = RoutingProvider(
        name="wrong-language",
        capabilities={Capability.OUTLINE},
        languages={"typescript"},
        confidence=ConfidenceClass.HIGH,
        health_score=1.0,
        result="wrong-language",
    )
    kernel = CodeIntelKernel([medium_healthy, high_degraded, wrong_capability, high_healthy, wrong_language])

    result = _run(kernel.call(Capability.OUTLINE, "python", path="src/service.py"))

    assert result.ok is True
    assert result.data == "high-healthy"
    assert result.meta.sources_used == ["high-healthy"]
    assert medium_healthy.outline_calls == []
    assert high_degraded.outline_calls == []
    assert high_healthy.outline_calls == ["src/service.py"]
    assert wrong_capability.health_calls == 0
    assert wrong_language.health_calls == 0
    assert kernel.last_trace is not None
    assert [attempt.provider for attempt in kernel.last_trace.attempts] == [
        "high-healthy",
        "high-degraded",
        "medium-healthy",
    ]
    assert [attempt.confidence for attempt in kernel.last_trace.attempts] == [
        ConfidenceClass.HIGH,
        ConfidenceClass.HIGH,
        ConfidenceClass.MEDIUM,
    ]
    assert kernel.last_trace.selected_provider == "high-healthy"


@pytest.mark.parametrize("fallback_error", [ProviderUnavailable(), LSPTimeout()])
def test_falls_back_when_provider_is_unavailable_or_times_out(fallback_error: CodeIntelError) -> None:
    primary = RoutingProvider(
        name="primary",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.HIGH,
        health_score=1.0,
        result="primary",
        error=fallback_error,
    )
    backup = RoutingProvider(
        name="backup",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.MEDIUM,
        health_score=1.0,
        result="backup",
    )
    kernel = CodeIntelKernel([primary, backup])

    result = _run(kernel.call(Capability.OUTLINE, "python", path="src/service.py"))

    assert result.ok is True
    assert result.data == "backup"
    assert result.meta.sources_used == ["backup"]
    assert kernel.last_trace is not None
    assert [attempt.provider for attempt in kernel.last_trace.attempts] == ["primary", "backup"]
    assert kernel.last_trace.attempts[0].fallback is True
    assert kernel.last_trace.attempts[0].error_code == fallback_error.code
    assert kernel.last_trace.attempts[1].selected is True
    assert kernel.last_trace.fallback_count == 1


def test_unavailable_health_is_not_attempted_and_next_provider_is_selected() -> None:
    unavailable = RoutingProvider(
        name="down",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.HIGH,
        health_score=0.0,
        status=ProviderStatus.UNAVAILABLE,
        result="down",
    )
    backup = RoutingProvider(
        name="backup",
        capabilities={Capability.OUTLINE},
        languages={"python"},
        confidence=ConfidenceClass.LOW,
        health_score=0.4,
        result="backup",
    )
    kernel = CodeIntelKernel([unavailable, backup])

    result = _run(kernel.call(Capability.OUTLINE, "python", path="src/service.py"))

    assert result.ok is True
    assert result.data == "backup"
    assert unavailable.outline_calls == []
    assert kernel.last_trace is not None
    assert [attempt.provider for attempt in kernel.last_trace.attempts] == ["backup", "down"]
    assert kernel.last_trace.attempts[1].attempted is False
    assert kernel.last_trace.attempts[1].error_code == "provider_unavailable"


def test_no_provider_returns_typed_tool_error_instead_of_raising() -> None:
    kernel = CodeIntelKernel()

    result = _run(kernel.call(Capability.OUTLINE, "python", path="src/service.py"))

    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.code == "unsupported_language"
    assert result.error.message == "当前语言不受支持。"
    assert result.error.hint == "请跳过本文件，或使用 read_file 直接查看源码。"
    assert kernel.last_trace is not None
    assert kernel.last_trace.attempts == []
    assert kernel.last_trace.selected_provider is None
