"""Contract tests for shared code_intel language helpers."""

from __future__ import annotations

import pytest

from src.code_intel.core.languages import (
    LANGUAGE_BY_EXTENSION,
    SUPPORTED_CODE_LANGUAGES,
    TOOL_LANGUAGE_BY_EXTENSION,
    language_for_path,
    language_for_path_or_default,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/module.py", "python"),
        ("pkg/types.PYI", "python"),
        ("web/component.tsx", "typescript"),
        ("web/entry.MTS", "typescript"),
        ("web/script.jsx", "javascript"),
        ("web/config.CJS", "javascript"),
    ],
)
def test_language_for_path_covers_supported_extensions_case_insensitively(
    path: str, expected: str
) -> None:
    assert language_for_path(path) == expected


def test_language_for_path_returns_none_for_unsupported_extension() -> None:
    assert language_for_path("docs/readme.md") is None


def test_supported_language_set_matches_extension_mapping_values() -> None:
    assert SUPPORTED_CODE_LANGUAGES == frozenset(LANGUAGE_BY_EXTENSION.values())
    assert SUPPORTED_CODE_LANGUAGES == frozenset({"python", "typescript", "javascript"})


def test_language_for_path_or_default_preserves_python_fallback() -> None:
    assert language_for_path_or_default(None) == "python"
    assert language_for_path_or_default("README.md") == "python"


def test_tool_language_mapping_covers_extra_tool_extensions_without_expanding_provider_languages() -> (
    None
):
    assert language_for_path("src/main.go", TOOL_LANGUAGE_BY_EXTENSION) == "go"
    assert language_for_path("src/lib.rs", TOOL_LANGUAGE_BY_EXTENSION) == "rust"
    assert "go" not in SUPPORTED_CODE_LANGUAGES
    assert "rust" not in SUPPORTED_CODE_LANGUAGES
