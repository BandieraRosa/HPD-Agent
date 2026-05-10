"""Contract tests for code_intel capabilities and provider protocols."""

from __future__ import annotations

import asyncio
from typing import cast, get_type_hints

import pytest
from pydantic import BaseModel, ValidationError

from src.code_intel.core import (
    Capability,
    CodeContext,
    CodeTarget,
    ConfidenceClass,
    ContextExtractProvider,
    ContextPart,
    DefinitionProvider,
    Diagnostic,
    DiagnosticsProvider,
    HoverInfo,
    HoverProvider,
    LanguageDetection,
    Location,
    OutlineProvider,
    Provider,
    ProviderHealth,
    ProviderStatus,
    ReferenceProvider,
    Symbol,
    SymbolKind,
    SymbolSearchProvider,
    Workspace,
)


def test_capability_enum_has_exact_machine_readable_values() -> None:
    assert [(capability.name, capability.value) for capability in Capability] == [
        ("OUTLINE", "outline"),
        ("SYMBOL_SEARCH", "symbol_search"),
        ("CONTEXT_EXTRACT", "context_extract"),
        ("DEFINITION", "definition"),
        ("REFERENCES", "references"),
        ("HOVER", "hover"),
        ("DIAGNOSTICS", "diagnostics"),
        ("DOCUMENT_SYMBOLS", "document_symbols"),
        ("RENAME", "rename"),
        ("TEXT_SEARCH", "text_search"),
    ]


def test_provider_health_and_confidence_serialize_as_english_values() -> None:
    health = ProviderHealth.model_validate({"status": ProviderStatus.HEALTHY, "score": 0.75})

    assert health.score == 0.75
    assert health.model_dump(mode="json") == {
        "status": "healthy",
        "health_score": 0.75,
        "message": None,
    }
    assert [confidence.value for confidence in ConfidenceClass] == ["high", "medium", "low"]
    assert [status.value for status in ProviderStatus] == ["healthy", "degraded", "unavailable"]


def test_context_part_values_are_stable_context_extract_contract() -> None:
    assert [(part.name, part.value) for part in ContextPart] == [
        ("SIGNATURE", "signature"),
        ("BODY", "body"),
        ("PARENTS", "parents"),
        ("IMPORTS", "imports"),
        ("NEARBY", "nearby"),
    ]


class OutlineOnlyProvider:
    name: str = "outline-only"
    capabilities: set[Capability] = {Capability.OUTLINE}
    languages: set[str] = {"python"}

    async def supports(self, capability: Capability, language: str) -> bool:
        return capability in self.capabilities and language in self.languages

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=ProviderStatus.HEALTHY, health_score=1.0)

    async def outline(self, _path: str) -> list[Symbol]:
        return []


def test_provider_declares_capabilities_without_notimplemented() -> None:
    provider = OutlineOnlyProvider()

    assert isinstance(provider, Provider)
    assert isinstance(provider, OutlineProvider)
    assert not isinstance(provider, DefinitionProvider)
    assert asyncio.run(provider.supports(Capability.OUTLINE, "python")) is True
    assert asyncio.run(provider.supports(Capability.DEFINITION, "python")) is False
    assert asyncio.run(provider.health()).status is ProviderStatus.HEALTHY
    assert asyncio.run(provider.outline("src/service.py")) == []
    assert not hasattr(provider, "goto_definition")
    assert not hasattr(provider, "find_references")
    assert not hasattr(provider, "hover")
    assert not hasattr(provider, "diagnostics")


def test_capability_protocol_signatures_use_t3_models() -> None:
    assert get_type_hints(OutlineProvider.outline)["return"] == list[Symbol]

    search_hints = get_type_hints(SymbolSearchProvider.search_symbols)
    assert search_hints["kind"] == SymbolKind | None
    assert search_hints["return"] == list[Symbol]

    context_hints = get_type_hints(ContextExtractProvider.extract_context)
    assert context_hints["target"] is CodeTarget
    assert context_hints["include"] == set[ContextPart]
    assert context_hints["return"] is CodeContext

    assert get_type_hints(DefinitionProvider.goto_definition) == {
        "target": CodeTarget,
        "return": list[Location],
    }
    assert get_type_hints(ReferenceProvider.find_references) == {
        "target": CodeTarget,
        "return": list[Location],
    }
    assert get_type_hints(HoverProvider.hover) == {
        "target": CodeTarget,
        "return": HoverInfo | None,
    }
    assert get_type_hints(DiagnosticsProvider.diagnostics) == {
        "path": str,
        "return": list[Diagnostic],
    }


def test_workspace_models_are_minimal_pydantic_contracts() -> None:
    assert issubclass(Workspace, BaseModel)
    assert issubclass(LanguageDetection, BaseModel)

    detection = LanguageDetection(
        path="src/service.py",
        language="python",
        confidence=0.95,
        source="extension",
    )
    workspace = Workspace(root_path="/repo/project", languages={"python"}, detections=[detection])
    other_workspace = Workspace.model_validate({"root": "/repo/other"})

    workspace.languages.add("typescript")
    dumped = workspace.model_dump(mode="json")
    dumped_languages = cast(list[str], dumped["languages"])
    dumped_detections = cast(list[dict[str, object]], dumped["detections"])

    assert workspace.root_path == "/repo/project"
    assert other_workspace.root_path == "/repo/other"
    assert other_workspace.languages == set()
    assert set(dumped_languages) == {"python", "typescript"}
    assert dumped_detections[0] == {
        "path": "src/service.py",
        "language": "python",
        "confidence": 0.95,
        "source": "extension",
    }


def test_language_detection_reuses_workspace_relative_path_validation() -> None:
    with pytest.raises(ValidationError):
        _ = LanguageDetection(path="../service.py", language="python", confidence=0.9)

    with pytest.raises(ValidationError):
        _ = LanguageDetection(path="src/service.py", language="", confidence=0.9)

    with pytest.raises(ValidationError):
        _ = Workspace(root_path="")
