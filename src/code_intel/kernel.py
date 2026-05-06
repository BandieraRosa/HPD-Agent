"""Code Intelligence Kernel facade and provider routing."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, Field

from .core import (
    Capability,
    CodeIntelError,
    CodeTarget,
    ConfidenceClass,
    ContextPart,
    LSPTimeout,
    LanguageNotSupported,
    ProviderHealth,
    ProviderStatus,
    ProviderUnavailable,
    SymbolKind,
    TargetResolver,
    ToolMeta,
    ToolResult,
)
from .index import IndexBackedCodeContext, IndexBackedTargetResolver, SymbolIndexStore
from .tracing import result_count, safe_error_metadata, trace_span

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
_INDEX_PROVIDER_NAME = "symbol_index"
_SEMANTIC_PLACEHOLDER_CAPABILITIES = {
    Capability.DEFINITION,
    Capability.REFERENCES,
    Capability.HOVER,
}


class _DynamicMethod(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class ProviderAttemptTrace(BaseModel):
    """Safe routing trace for one provider candidate."""

    provider: str = Field(description="Provider name recorded for routing diagnostics.")
    confidence: ConfidenceClass = Field(
        description="Provider confidence class for the requested capability."
    )
    health: ProviderHealth = Field(
        description="Health snapshot used by this routing decision."
    )
    attempted: bool = Field(
        default=False, description="Whether the kernel invoked this provider."
    )
    selected: bool = Field(
        default=False, description="Whether this provider returned the selected result."
    )
    fallback: bool = Field(
        default=False,
        description="Whether the kernel fell back after this provider failed.",
    )
    error_code: str | None = Field(
        default=None, description="Safe English error code, when this provider failed."
    )


class KernelTrace(BaseModel):
    """Last-call routing trace exposed by CodeIntelKernel."""

    capability: Capability = Field(description="Requested capability.")
    language: str = Field(description="Requested language.")
    attempts: list[ProviderAttemptTrace] = Field(
        default_factory=list,
        description="Provider candidates in routing order, including skipped unhealthy candidates.",
    )
    selected_provider: str | None = Field(
        default=None, description="Provider selected for the final result."
    )
    fallback_count: int = Field(
        default=0, ge=0, description="Number of transient provider fallbacks."
    )


class CodeIntelKernel:
    """Provider registry, optional symbol-index facade, and async capability router."""

    def __init__(
        self,
        providers: Iterable[object] | None = None,
        *,
        symbol_index: SymbolIndexStore | None = None,
        workspace_root: str | Path = ".",
    ) -> None:
        self._providers: list[object] = []
        self._symbol_index: SymbolIndexStore | None = symbol_index
        self._target_resolver: IndexBackedTargetResolver | None = (
            IndexBackedTargetResolver(symbol_index, workspace_root)
            if symbol_index is not None
            else None
        )
        self._context_extractor: IndexBackedCodeContext | None = (
            IndexBackedCodeContext(
                symbol_index, workspace_root, resolver=self._target_resolver
            )
            if symbol_index is not None and self._target_resolver is not None
            else None
        )
        self.last_trace: KernelTrace | None = None
        for provider in providers or ():
            _ = self.register_provider(provider)

    @property
    def providers(self) -> tuple[object, ...]:
        """Return explicitly registered providers in registration order."""
        return tuple(self._providers)

    @property
    def target_resolver(self) -> TargetResolver | None:
        """Return the configured target resolver, when a symbol index is available."""
        return self._target_resolver

    def register_provider(self, provider: object) -> "CodeIntelKernel":
        """Register a provider explicitly and return the kernel for chaining.

        Provider registration is idempotent by provider name so repeated runtime
        startup or explicit /lsp start calls cannot create duplicate routes.
        """
        provider_name = self._provider_name(provider)
        if any(
            self._provider_name(existing) == provider_name
            for existing in self._providers
        ):
            return self
        self._providers.append(provider)
        return self

    def attach_symbol_index(
        self, store: SymbolIndexStore | None, workspace_root: str | Path = "."
    ) -> "CodeIntelKernel":
        """Attach or detach the symbol index facade without replacing providers."""
        self._symbol_index = store
        self._target_resolver = (
            IndexBackedTargetResolver(store, workspace_root)
            if store is not None
            else None
        )
        self._context_extractor = (
            IndexBackedCodeContext(
                store, workspace_root, resolver=self._target_resolver
            )
            if store is not None and self._target_resolver is not None
            else None
        )
        return self

    async def resolve_target(self, target: CodeTarget) -> ToolResult[object]:
        """Resolve a CodeTarget through the index-backed resolver when configured."""
        with trace_span(
            "code_intel.provider.symbol_index.resolve_target",
            {
                "provider_name": _INDEX_PROVIDER_NAME,
                **self._target_trace_metadata(target),
            },
        ) as span:
            result = await self._resolve_target_untraced(target)
            span.add_metadata(self._result_trace_metadata(result))
            return result

    async def _resolve_target_untraced(self, target: CodeTarget) -> ToolResult[object]:
        started_at = time.perf_counter()
        if self._target_resolver is None:
            if target.location is not None:
                return ToolResult(
                    ok=True,
                    data=target.location,
                    meta=ToolMeta(elapsed_ms=self._elapsed_ms(started_at)),
                )
            return self._error_result(ProviderUnavailable(), started_at)

        try:
            resolved = await self._target_resolver.resolve_target(target)
        except CodeIntelError as error:
            return self._index_error_result(error, started_at)
        except Exception:
            return self._index_error_result(CodeIntelError(), started_at)

        return self._index_success_result(
            resolved.location,
            started_at,
            flags=list(resolved.flags),
        )

    async def call(
        self, capability: Capability | str, language: str, **kwargs: object
    ) -> ToolResult[object]:
        """Route an async capability call to the index or best available provider."""
        capability_label = (
            capability.value if isinstance(capability, Capability) else str(capability)
        )
        with trace_span(
            "code_intel.kernel.dispatch",
            {
                "capability": capability_label,
                "language": language,
                **self._call_path_trace_metadata(kwargs),
            },
        ) as span:
            result = await self._call_untraced(capability, language, **kwargs)
            span.add_metadata(self._dispatch_trace_metadata(result))
            return result

    async def _call_untraced(
        self, capability: Capability | str, language: str, **kwargs: object
    ) -> ToolResult[object]:
        started_at = time.perf_counter()
        requested_capability = Capability(capability)
        trace = KernelTrace(capability=requested_capability, language=language)
        self.last_trace = trace

        index_result = await self._call_index_capability(
            requested_capability, kwargs, started_at
        )
        if index_result is not None:
            return index_result

        candidate_records: list[tuple[object, ProviderAttemptTrace]] = []
        unavailable_records: list[ProviderAttemptTrace] = []

        for provider in self._providers:
            provider_name = self._provider_name(provider)
            try:
                supported = await self._supports(
                    provider, requested_capability, language
                )
            except CodeIntelError as error:
                unavailable_records.append(
                    self._failed_attempt(provider_name, error.code)
                )
                continue
            except Exception:
                unavailable_records.append(
                    self._failed_attempt(provider_name, CodeIntelError.code)
                )
                continue

            if not supported:
                continue

            try:
                health = await self._health(provider)
            except CodeIntelError as error:
                unavailable_records.append(
                    self._failed_attempt(provider_name, error.code)
                )
                continue
            except Exception:
                unavailable_records.append(
                    self._failed_attempt(provider_name, CodeIntelError.code)
                )
                continue

            try:
                confidence = await self._confidence_for(
                    provider, requested_capability, language
                )
            except CodeIntelError as error:
                unavailable_records.append(
                    self._failed_attempt(provider_name, error.code, health)
                )
                continue
            except Exception:
                unavailable_records.append(
                    self._failed_attempt(provider_name, CodeIntelError.code, health)
                )
                continue

            attempt = ProviderAttemptTrace(
                provider=provider_name, confidence=confidence, health=health
            )

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
        trace.attempts = [
            attempt for _provider, attempt in candidate_records
        ] + unavailable_records

        if not candidate_records:
            error = (
                ProviderUnavailable()
                if trace.attempts
                or requested_capability in _SEMANTIC_PLACEHOLDER_CAPABILITIES
                else LanguageNotSupported()
            )
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
            provider_meta = self._provider_meta(data)
            return ToolResult(
                ok=True,
                data=data,
                meta=provider_meta.model_copy(
                    update={
                        "elapsed_ms": self._elapsed_ms(started_at),
                        "sources_used": provider_meta.sources_used
                        or [attempt.provider],
                    }
                ),
            )

        return self._error_result(
            last_fallback_error or ProviderUnavailable(), started_at
        )

    async def _call_index_capability(
        self,
        capability: Capability,
        kwargs: dict[str, object],
        started_at: float,
    ) -> ToolResult[object] | None:
        if self._symbol_index is None:
            return None

        if capability == Capability.SYMBOL_SEARCH:
            query = kwargs.get("query")
            if not isinstance(query, str):
                return None
            limit = kwargs.get("limit", 20)
            if not isinstance(limit, int):
                return None
            try:
                kind = self._coerce_symbol_kind(kwargs.get("kind"))
            except ValueError:
                return None
            with trace_span(
                "code_intel.provider.symbol_index.symbol_search",
                {
                    "provider_name": _INDEX_PROVIDER_NAME,
                    "capability": capability.value,
                },
            ) as span:
                try:
                    await self._symbol_index.initialize()
                    symbols = await self._symbol_index.search_symbols(
                        query, kind=kind, limit=limit
                    )
                except CodeIntelError as error:
                    result = self._index_error_result(error, started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                except Exception:
                    result = self._index_error_result(CodeIntelError(), started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                result = self._index_success_result(symbols, started_at)
                span.add_metadata(self._result_trace_metadata(result))
                return result

        if capability == Capability.DOCUMENT_SYMBOLS:
            path = kwargs.get("path")
            if not isinstance(path, str):
                return None
            with trace_span(
                "code_intel.provider.symbol_index.document_symbols",
                {
                    "provider_name": _INDEX_PROVIDER_NAME,
                    "capability": capability.value,
                    "path": path,
                },
            ) as span:
                try:
                    await self._symbol_index.initialize()
                    symbols = await self._symbol_index.get_symbols(path)
                except CodeIntelError as error:
                    result = self._index_error_result(error, started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                except Exception:
                    result = self._index_error_result(CodeIntelError(), started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                result = self._index_success_result(symbols, started_at)
                span.add_metadata(self._result_trace_metadata(result))
                return result

        if (
            capability == Capability.CONTEXT_EXTRACT
            and self._context_extractor is not None
        ):
            target = kwargs.get("target")
            include = self._coerce_context_parts(kwargs.get("include"))
            max_tokens = kwargs.get("max_tokens")
            if (
                not isinstance(target, CodeTarget)
                or include is None
                or not isinstance(max_tokens, int)
            ):
                return None
            with trace_span(
                "code_intel.provider.symbol_index.context_extract",
                {
                    "provider_name": _INDEX_PROVIDER_NAME,
                    "capability": capability.value,
                    **self._target_trace_metadata(target),
                },
            ) as span:
                try:
                    context, flags = await self._context_extractor.extract_context(
                        target, include, max_tokens
                    )
                except CodeIntelError as error:
                    result = self._index_error_result(error, started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                except Exception:
                    result = self._index_error_result(CodeIntelError(), started_at)
                    span.add_metadata(self._result_trace_metadata(result))
                    return result
                result = self._index_success_result(
                    context,
                    started_at,
                    truncated=context.truncated,
                    flags=list(flags),
                )
                span.add_metadata(self._result_trace_metadata(result))
                return result

        return None

    @staticmethod
    def _coerce_symbol_kind(value: object) -> SymbolKind | None:
        if value is None:
            return None
        if isinstance(value, SymbolKind):
            return value
        if isinstance(value, str):
            return SymbolKind(value)
        raise ValueError("invalid symbol kind")

    @staticmethod
    def _coerce_context_parts(value: object) -> set[ContextPart] | None:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Iterable
        ):
            return None
        parts: set[ContextPart] = set()
        for item in value:
            try:
                parts.add(
                    item if isinstance(item, ContextPart) else ContextPart(str(item))
                )
            except ValueError:
                return None
        return parts

    def _index_success_result(
        self,
        data: object,
        started_at: float,
        *,
        truncated: bool = False,
        more_available: bool = False,
        flags: list[str] | None = None,
    ) -> ToolResult[object]:
        self._mark_index_trace(selected=True)
        return ToolResult(
            ok=True,
            data=data,
            meta=ToolMeta(
                elapsed_ms=self._elapsed_ms(started_at),
                truncated=truncated,
                more_available=more_available,
                sources_used=[_INDEX_PROVIDER_NAME],
                flags=flags or [],
            ),
        )

    def _index_error_result(
        self, error: CodeIntelError, started_at: float
    ) -> ToolResult[object]:
        self._mark_index_trace(selected=False, error_code=error.code)
        return self._error_result(error, started_at)

    def _mark_index_trace(
        self, *, selected: bool, error_code: str | None = None
    ) -> None:
        trace = self.last_trace
        if trace is None:
            return
        trace.attempts = [
            ProviderAttemptTrace(
                provider=_INDEX_PROVIDER_NAME,
                confidence=ConfidenceClass.HIGH,
                health=ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0),
                attempted=True,
                selected=selected,
                error_code=error_code,
            )
        ]
        trace.selected_provider = _INDEX_PROVIDER_NAME if selected else None

    @staticmethod
    def _failed_attempt(
        provider_name: str,
        error_code: str,
        health: ProviderHealth | None = None,
    ) -> ProviderAttemptTrace:
        return ProviderAttemptTrace(
            provider=provider_name,
            confidence=ConfidenceClass.LOW,
            health=health
            or ProviderHealth(status=ProviderStatus.UNAVAILABLE, health_score=0.0),
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
    def _callable_attribute(
        cls, provider: object, attribute_name: str
    ) -> _DynamicMethod | None:
        attribute = cls._attribute(provider, attribute_name)
        if not callable(attribute):
            return None
        return cast(_DynamicMethod, attribute)

    @staticmethod
    async def _maybe_await(value: object) -> object:
        if inspect.isawaitable(value):
            return await cast(Awaitable[object], value)
        return value

    async def _supports(
        self, provider: object, capability: Capability, language: str
    ) -> bool:
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

    async def _confidence_for(
        self, provider: object, capability: Capability, language: str
    ) -> ConfidenceClass:
        confidence_for = self._callable_attribute(provider, "confidence_for")
        if confidence_for is not None:
            raw_confidence = await self._maybe_await(
                confidence_for(capability, language)
            )
            return self._coerce_confidence(raw_confidence, capability)

        for attribute_name in ("confidence_class", "confidence"):
            raw_confidence = self._attribute(provider, attribute_name)
            if raw_confidence is None:
                continue
            callable_confidence = self._callable_attribute(provider, attribute_name)
            if callable_confidence is not None:
                raw_confidence = await self._maybe_await(
                    callable_confidence(capability, language)
                )
            return self._coerce_confidence(raw_confidence, capability)

        return ConfidenceClass.LOW

    @staticmethod
    def _coerce_confidence(
        raw_confidence: object, capability: Capability
    ) -> ConfidenceClass:
        if isinstance(raw_confidence, Mapping):
            confidence_map = cast(Mapping[object, object], raw_confidence)
            raw_confidence = confidence_map.get(
                capability,
                confidence_map.get(capability.value, ConfidenceClass.LOW),
            )
        if isinstance(raw_confidence, ConfidenceClass):
            return raw_confidence
        return ConfidenceClass(str(raw_confidence))

    async def _invoke(
        self, provider: object, capability: Capability, kwargs: dict[str, object]
    ) -> object:
        method_name = _CAPABILITY_METHODS.get(capability)
        if method_name is None:
            raise ProviderUnavailable(f"capability {capability.value} has no route")

        method = self._callable_attribute(provider, method_name)
        if method is None:
            raise ProviderUnavailable(f"provider missing {method_name}")

        provider_name = self._provider_name(provider)
        with trace_span(
            f"code_intel.provider.{provider_name}.{method_name}",
            {
                "provider_name": provider_name,
                "capability": capability.value,
                **self._call_path_trace_metadata(kwargs),
            },
        ) as span:
            result = await self._maybe_await(method(**kwargs))
            span.add_metadata(
                {
                    "result_count": result_count(result),
                    "truncated": getattr(result, "truncated", False),
                }
            )
            return result

    @staticmethod
    def _provider_meta(data: object) -> ToolMeta:
        raw_meta = getattr(data, "tool_meta", None)
        if raw_meta is None:
            raw_meta = getattr(data, "meta", None)
        if raw_meta is None:
            return ToolMeta()
        if isinstance(raw_meta, ToolMeta):
            return raw_meta
        try:
            return ToolMeta.model_validate(raw_meta)
        except Exception:
            return ToolMeta()

    def _dispatch_trace_metadata(self, result: ToolResult[object]) -> dict[str, object]:
        metadata = self._result_trace_metadata(result)
        trace = self.last_trace
        if trace is None:
            return metadata
        if trace.selected_provider is not None:
            metadata["provider_name"] = trace.selected_provider
        fallback_chain = [
            attempt.provider
            for attempt in trace.attempts
            if attempt.attempted
            or attempt.fallback
            or attempt.selected
            or attempt.error_code
        ]
        if fallback_chain:
            metadata["fallback_chain"] = fallback_chain
        return metadata

    @staticmethod
    def _result_trace_metadata(result: ToolResult[object]) -> dict[str, object]:
        metadata: dict[str, object] = {
            "result_count": result_count(result.data),
            "truncated": result.meta.truncated,
            "elapsed_ms": result.meta.elapsed_ms,
        }
        if result.error is not None:
            metadata.update(safe_error_metadata(result.error))
        return metadata

    @staticmethod
    def _call_path_trace_metadata(kwargs: Mapping[str, object]) -> dict[str, object]:
        path = kwargs.get("path")
        if isinstance(path, str):
            return {"path": path}
        target = kwargs.get("target")
        if isinstance(target, CodeTarget):
            return CodeIntelKernel._target_trace_metadata(target)
        return {}

    @staticmethod
    def _target_trace_metadata(target: CodeTarget) -> dict[str, object]:
        if target.location is not None:
            return {"path": target.location.path}
        if target.anchor is not None:
            return {"path": target.anchor.path}
        return {}

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((time.perf_counter() - started_at) * 1000))

    def _error_result(
        self, error: CodeIntelError, started_at: float
    ) -> ToolResult[object]:
        return ToolResult(
            ok=False,
            error=error.to_tool_error(),
            meta=ToolMeta(elapsed_ms=self._elapsed_ms(started_at)),
        )


__all__ = ["CodeIntelKernel", "KernelTrace", "ProviderAttemptTrace"]
