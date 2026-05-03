"""Code Intelligence Kernel facade and provider routing."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Iterable, Mapping
from typing import Protocol, cast

from pydantic import BaseModel, Field

from .core import (
    Capability,
    CodeIntelError,
    ConfidenceClass,
    LSPTimeout,
    LanguageNotSupported,
    ProviderHealth,
    ProviderStatus,
    ProviderUnavailable,
    ToolMeta,
    ToolResult,
)

_CONFIDENCE_RANK: dict[ConfidenceClass, int] = {
    ConfidenceClass.HIGH: 3,
    ConfidenceClass.MEDIUM: 2,
    ConfidenceClass.LOW: 1,
}

_CAPABILITY_METHODS: dict[Capability, str] = {
    Capability.OUTLINE: "outline",
    Capability.SYMBOL_SEARCH: "search_symbols",
    Capability.CONTEXT_EXTRACT: "extract_context",
    Capability.DEFINITION: "goto_definition",
    Capability.REFERENCES: "find_references",
    Capability.HOVER: "hover",
    Capability.DIAGNOSTICS: "diagnostics",
    Capability.DOCUMENT_SYMBOLS: "document_symbols",
    Capability.TEXT_SEARCH: "text_search",
}


class _DynamicMethod(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class ProviderAttemptTrace(BaseModel):
    """Safe routing trace for one provider candidate."""

    provider: str = Field(description="Provider name recorded for routing diagnostics.")
    confidence: ConfidenceClass = Field(description="Provider confidence class for the requested capability.")
    health: ProviderHealth = Field(description="Health snapshot used by this routing decision.")
    attempted: bool = Field(default=False, description="Whether the kernel invoked this provider.")
    selected: bool = Field(default=False, description="Whether this provider returned the selected result.")
    fallback: bool = Field(default=False, description="Whether the kernel fell back after this provider failed.")
    error_code: str | None = Field(default=None, description="Safe English error code, when this provider failed.")


class KernelTrace(BaseModel):
    """Last-call routing trace exposed by CodeIntelKernel."""

    capability: Capability = Field(description="Requested capability.")
    language: str = Field(description="Requested language.")
    attempts: list[ProviderAttemptTrace] = Field(
        default_factory=list,
        description="Provider candidates in routing order, including skipped unhealthy candidates.",
    )
    selected_provider: str | None = Field(default=None, description="Provider selected for the final result.")
    fallback_count: int = Field(default=0, ge=0, description="Number of transient provider fallbacks.")


class CodeIntelKernel:
    """Provider registry and async capability router for code intelligence."""

    def __init__(self, providers: Iterable[object] | None = None) -> None:
        self._providers: list[object] = []
        self.last_trace: KernelTrace | None = None
        for provider in providers or ():
            _ = self.register_provider(provider)

    @property
    def providers(self) -> tuple[object, ...]:
        """Return explicitly registered providers in registration order."""
        return tuple(self._providers)

    def register_provider(self, provider: object) -> "CodeIntelKernel":
        """Register a provider explicitly and return the kernel for chaining."""
        self._providers.append(provider)
        return self

    async def call(self, capability: Capability | str, language: str, **kwargs: object) -> ToolResult[object]:
        """Route an async capability call to the best available provider."""
        started_at = time.perf_counter()
        requested_capability = Capability(capability)
        trace = KernelTrace(capability=requested_capability, language=language)
        self.last_trace = trace

        candidate_records: list[tuple[object, ProviderAttemptTrace]] = []
        unavailable_records: list[ProviderAttemptTrace] = []

        for provider in self._providers:
            provider_name = self._provider_name(provider)
            try:
                supported = await self._supports(provider, requested_capability, language)
            except CodeIntelError as error:
                unavailable_records.append(self._failed_attempt(provider_name, error.code))
                continue
            except Exception:
                unavailable_records.append(self._failed_attempt(provider_name, CodeIntelError.code))
                continue

            if not supported:
                continue

            try:
                health = await self._health(provider)
            except CodeIntelError as error:
                unavailable_records.append(self._failed_attempt(provider_name, error.code))
                continue
            except Exception:
                unavailable_records.append(self._failed_attempt(provider_name, CodeIntelError.code))
                continue

            try:
                confidence = await self._confidence_for(provider, requested_capability, language)
            except CodeIntelError as error:
                unavailable_records.append(self._failed_attempt(provider_name, error.code, health))
                continue
            except Exception:
                unavailable_records.append(self._failed_attempt(provider_name, CodeIntelError.code, health))
                continue

            attempt = ProviderAttemptTrace(provider=provider_name, confidence=confidence, health=health)

            if health.status == ProviderStatus.UNAVAILABLE:
                attempt.error_code = ProviderUnavailable.code
                unavailable_records.append(attempt)
                continue

            candidate_records.append((provider, attempt))

        candidate_records.sort(
            key=lambda record: (
                _CONFIDENCE_RANK[record[1].confidence],
                record[1].health.health_score,
            ),
            reverse=True,
        )
        trace.attempts = [attempt for _provider, attempt in candidate_records] + unavailable_records

        if not candidate_records:
            error = ProviderUnavailable() if trace.attempts else LanguageNotSupported()
            return self._error_result(error, started_at)

        last_fallback_error: CodeIntelError | None = None
        for provider, attempt in candidate_records:
            attempt.attempted = True
            try:
                data = await self._invoke(provider, requested_capability, kwargs)
            except (ProviderUnavailable, LSPTimeout) as error:
                attempt.error_code = error.code
                attempt.fallback = True
                trace.fallback_count += 1
                last_fallback_error = error
                continue
            except CodeIntelError as error:
                attempt.error_code = error.code
                return self._error_result(error, started_at)
            except Exception:
                error = CodeIntelError()
                attempt.error_code = error.code
                return self._error_result(error, started_at)

            attempt.selected = True
            trace.selected_provider = attempt.provider
            return ToolResult(
                ok=True,
                data=data,
                meta=ToolMeta(
                    elapsed_ms=self._elapsed_ms(started_at),
                    sources_used=[attempt.provider],
                ),
            )

        return self._error_result(last_fallback_error or ProviderUnavailable(), started_at)

    @staticmethod
    def _failed_attempt(
        provider_name: str,
        error_code: str,
        health: ProviderHealth | None = None,
    ) -> ProviderAttemptTrace:
        return ProviderAttemptTrace(
            provider=provider_name,
            confidence=ConfidenceClass.LOW,
            health=health or ProviderHealth(status=ProviderStatus.UNAVAILABLE, health_score=0.0),
            error_code=error_code,
        )

    @staticmethod
    def _provider_name(provider: object) -> str:
        name = cast(object, getattr(provider, "name", provider.__class__.__name__))
        return str(name)

    @staticmethod
    def _attribute(provider: object, attribute_name: str) -> object | None:
        return cast(object | None, getattr(provider, attribute_name, None))

    @classmethod
    def _callable_attribute(cls, provider: object, attribute_name: str) -> _DynamicMethod | None:
        attribute = cls._attribute(provider, attribute_name)
        if not callable(attribute):
            return None
        return cast(_DynamicMethod, attribute)

    @staticmethod
    async def _maybe_await(value: object) -> object:
        if inspect.isawaitable(value):
            return await cast(Awaitable[object], value)
        return value

    async def _supports(self, provider: object, capability: Capability, language: str) -> bool:
        supports = self._callable_attribute(provider, "supports")
        if supports is None:
            return False

        return bool(await self._maybe_await(supports(capability, language)))

    async def _health(self, provider: object) -> ProviderHealth:
        health = self._callable_attribute(provider, "health")
        if health is None:
            return ProviderHealth(status=ProviderStatus.UNAVAILABLE, health_score=0.0)

        raw_health = await self._maybe_await(health())
        if isinstance(raw_health, ProviderHealth):
            return raw_health
        return ProviderHealth.model_validate(raw_health)

    async def _confidence_for(self, provider: object, capability: Capability, language: str) -> ConfidenceClass:
        confidence_for = self._callable_attribute(provider, "confidence_for")
        if confidence_for is not None:
            raw_confidence = await self._maybe_await(confidence_for(capability, language))
            return self._coerce_confidence(raw_confidence, capability)

        for attribute_name in ("confidence_class", "confidence"):
            raw_confidence = self._attribute(provider, attribute_name)
            if raw_confidence is None:
                continue
            callable_confidence = self._callable_attribute(provider, attribute_name)
            if callable_confidence is not None:
                raw_confidence = await self._maybe_await(callable_confidence(capability, language))
            return self._coerce_confidence(raw_confidence, capability)

        return ConfidenceClass.LOW

    @staticmethod
    def _coerce_confidence(raw_confidence: object, capability: Capability) -> ConfidenceClass:
        if isinstance(raw_confidence, Mapping):
            confidence_map = cast(Mapping[object, object], raw_confidence)
            raw_confidence = confidence_map.get(
                capability,
                confidence_map.get(capability.value, ConfidenceClass.LOW),
            )
        if isinstance(raw_confidence, ConfidenceClass):
            return raw_confidence
        return ConfidenceClass(str(raw_confidence))

    async def _invoke(self, provider: object, capability: Capability, kwargs: dict[str, object]) -> object:
        method_name = _CAPABILITY_METHODS.get(capability)
        if method_name is None:
            raise ProviderUnavailable(f"capability {capability.value} has no route")

        method = self._callable_attribute(provider, method_name)
        if method is None:
            raise ProviderUnavailable(f"provider missing {method_name}")

        return await self._maybe_await(method(**kwargs))

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((time.perf_counter() - started_at) * 1000))

    def _error_result(self, error: CodeIntelError, started_at: float) -> ToolResult[object]:
        return ToolResult(
            ok=False,
            error=error.to_tool_error(),
            meta=ToolMeta(elapsed_ms=self._elapsed_ms(started_at)),
        )


__all__ = ["CodeIntelKernel", "KernelTrace", "ProviderAttemptTrace"]
