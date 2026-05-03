"""Tests for deterministic fake code-intelligence providers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from typing import TypeVar, cast

from pydantic import BaseModel

from src.code_intel import CodeIntelKernel
from src.code_intel.core import (
    Capability,
    CodeContext,
    CodeTarget,
    ConfidenceClass,
    ContextPart,
    Diagnostic,
    HoverInfo,
    Location,
    Symbol,
    SymbolKind,
)
from src.code_intel.providers.fake import (
    PYTHON_FAKE_PATH,
    FakeSemanticProvider,
    FakeSyntaxProvider,
    create_fake_providers,
)

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _json_dump_sequence(items: Sequence[BaseModel]) -> list[dict[str, object]]:
    return [cast(dict[str, object], item.model_dump(mode="json")) for item in items]


def test_fake_provider_is_explicit_not_kernel_default() -> None:
    empty_kernel = CodeIntelKernel()

    empty_result = _run(empty_kernel.call(Capability.OUTLINE, "python", path=PYTHON_FAKE_PATH))

    assert empty_result.ok is False
    assert empty_result.error is not None
    assert empty_result.error.code == "unsupported_language"

    explicit_kernel = CodeIntelKernel(create_fake_providers())
    outline_result = _run(explicit_kernel.call(Capability.OUTLINE, "python", path=PYTHON_FAKE_PATH))

    assert outline_result.ok is True
    assert explicit_kernel.last_trace is not None
    assert explicit_kernel.last_trace.selected_provider == "fake_syntax"


def test_fake_syntax_provider_returns_deterministic_outline_search_and_text_locations() -> None:
    provider = FakeSyntaxProvider()

    first_outline = _run(provider.outline(PYTHON_FAKE_PATH))
    second_outline = _run(provider.outline(PYTHON_FAKE_PATH))
    search_result = _run(provider.search_symbols("fake", limit=10))
    method_search = _run(provider.search_symbols("run", kind=SymbolKind.METHOD))
    document_symbols = _run(provider.document_symbols(PYTHON_FAKE_PATH))
    text_matches = _run(provider.text_search("fake", path=PYTHON_FAKE_PATH))

    assert _json_dump_sequence(first_outline) == _json_dump_sequence(second_outline)
    assert [symbol.name for symbol in first_outline] == ["FakeService", "run", "helper"]
    assert all(isinstance(symbol, Symbol) for symbol in first_outline)
    assert [symbol.name for symbol in search_result] == ["FakeService", "run", "fakeClient"]
    assert [symbol.qualified_name for symbol in method_search] == ["FakeService.run"]
    assert _json_dump_sequence(document_symbols) == _json_dump_sequence(first_outline)
    assert all(isinstance(location, Location) for location in text_matches)
    assert [location.path for location in text_matches] == [PYTHON_FAKE_PATH, PYTHON_FAKE_PATH]


def test_fake_semantic_provider_returns_context_definition_references_hover_and_verify_data() -> None:
    syntax = FakeSyntaxProvider()
    semantic = FakeSemanticProvider()
    run_symbol = _run(syntax.search_symbols("run", kind=SymbolKind.METHOD))[0]
    target = CodeTarget(symbol_id=run_symbol.id)

    context = _run(
        semantic.extract_context(
            target,
            {ContextPart.SIGNATURE, ContextPart.BODY, ContextPart.PARENTS, ContextPart.IMPORTS, ContextPart.NEARBY},
            max_tokens=256,
        )
    )
    definition = _run(semantic.goto_definition(target))
    references = _run(semantic.find_references(target))
    hover = _run(semantic.hover(target))
    diagnostics = _run(semantic.diagnostics(PYTHON_FAKE_PATH))
    verify_diagnostics = _run(semantic.verify(PYTHON_FAKE_PATH))

    assert isinstance(context, CodeContext)
    assert context.target_symbol.qualified_name == "FakeService.run"
    assert context.signature == "def run(self, value: str) -> str"
    assert context.body is not None and "helper(value)" in context.body
    assert [parent.name for parent in context.parents] == ["FakeService"]
    assert context.imports == ["from __future__ import annotations"]
    assert {symbol.name for symbol in context.nearby_symbols} == {"FakeService", "helper"}
    assert definition == [Location(path=PYTHON_FAKE_PATH, range=run_symbol.selection_range or run_symbol.range)]
    assert len(references) == 2
    assert all(isinstance(location, Location) for location in references)
    assert isinstance(hover, HoverInfo)
    assert hover.source == "fake_semantic"
    assert "def run" in hover.contents
    assert [diagnostic.model_dump(mode="json") for diagnostic in diagnostics] == [
        diagnostic.model_dump(mode="json") for diagnostic in verify_diagnostics
    ]
    assert all(isinstance(diagnostic, Diagnostic) for diagnostic in diagnostics)
    assert diagnostics[0].code == "fake-warning"


def test_kernel_routes_across_fake_vertical_slice() -> None:
    kernel = CodeIntelKernel(create_fake_providers())
    outline = _run(kernel.call(Capability.OUTLINE, "python", path=PYTHON_FAKE_PATH))
    assert outline.ok is True
    assert outline.data is not None
    outline_data = cast(list[Symbol], outline.data)
    run_symbol = next(symbol for symbol in outline_data if symbol.name == "run")
    target = CodeTarget(symbol_id=run_symbol.id)

    calls = {
        Capability.SYMBOL_SEARCH: {"query": "fake", "limit": 10},
        Capability.DOCUMENT_SYMBOLS: {"path": PYTHON_FAKE_PATH},
        Capability.TEXT_SEARCH: {"query": "fake", "path": PYTHON_FAKE_PATH},
        Capability.CONTEXT_EXTRACT: {
            "target": target,
            "include": {ContextPart.SIGNATURE, ContextPart.BODY},
            "max_tokens": 256,
        },
        Capability.DEFINITION: {"target": target},
        Capability.REFERENCES: {"target": target},
        Capability.HOVER: {"target": target},
        Capability.DIAGNOSTICS: {"path": PYTHON_FAKE_PATH},
    }

    results = {capability: _run(kernel.call(capability, "python", **kwargs)) for capability, kwargs in calls.items()}

    assert all(result.ok for result in results.values())
    assert all(result.data is not None for result in results.values())
    symbol_search_data = cast(list[Symbol], results[Capability.SYMBOL_SEARCH].data)
    document_symbol_data = cast(list[Symbol], results[Capability.DOCUMENT_SYMBOLS].data)
    text_search_data = cast(list[Location], results[Capability.TEXT_SEARCH].data)
    context_data = cast(CodeContext, results[Capability.CONTEXT_EXTRACT].data)
    definition_data = cast(list[Location], results[Capability.DEFINITION].data)
    references_data = cast(list[Location], results[Capability.REFERENCES].data)
    hover_data = cast(HoverInfo, results[Capability.HOVER].data)
    diagnostic_data = cast(list[Diagnostic], results[Capability.DIAGNOSTICS].data)

    assert [symbol.name for symbol in symbol_search_data] == ["FakeService", "run", "fakeClient"]
    assert [symbol.name for symbol in document_symbol_data] == ["FakeService", "run", "helper"]
    assert all(isinstance(location, Location) for location in text_search_data)
    assert context_data.target_symbol.id == run_symbol.id
    assert isinstance(definition_data[0], Location)
    assert len(references_data) == 2
    assert isinstance(hover_data, HoverInfo)
    assert isinstance(diagnostic_data[0], Diagnostic)
    assert kernel.last_trace is not None
    assert kernel.last_trace.selected_provider == "fake_semantic"


def test_fake_provider_supports_and_confidence_are_stable() -> None:
    syntax = FakeSyntaxProvider()
    semantic = FakeSemanticProvider()

    assert _run(syntax.supports(Capability.OUTLINE, "python")) is True
    assert _run(syntax.supports(Capability.DEFINITION, "python")) is False
    assert _run(semantic.supports(Capability.DEFINITION, "python")) is True
    assert _run(semantic.supports(Capability.OUTLINE, "python")) is False
    assert _run(syntax.confidence_for(Capability.OUTLINE, "python")) == ConfidenceClass.MEDIUM
    assert _run(syntax.confidence_for(Capability.TEXT_SEARCH, "python")) == ConfidenceClass.LOW
    assert _run(semantic.confidence_for(Capability.DEFINITION, "python")) == ConfidenceClass.HIGH
