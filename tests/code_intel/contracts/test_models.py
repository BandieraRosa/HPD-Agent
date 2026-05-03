"""Contract tests for code_intel core Pydantic models."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import BaseModel, ValidationError

from src.code_intel.core.models import (
    CodeContext,
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    Range,
    Symbol,
    SymbolKind,
    ToolError,
    ToolMeta,
    ToolResult,
)


def _range() -> Range:
    return Range(start_line=1, start_col=2, end_line=3, end_col=4)


def _location() -> Location:
    return Location(path="src/service.py", range=_range())


def _symbol(
    name: str = "login",
    qualified_name: str | None = "UserService.login",
    kind: SymbolKind = SymbolKind.METHOD,
    language: str = "python",
    path: str = "src/service.py",
    symbol_range: Range | None = None,
    selection_range: Range | None = None,
    parent_id: str | None = None,
    signature: str | None = "def login(self, user: str) -> Token",
    doc: str | None = "Authenticate a user.",
    source: str = "tree_sitter",
    confidence: float = 0.9,
    file_hash: str = "deadbeefcafebabe",
    index_version: str = "v1",
    stale: bool = False,
) -> Symbol:
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        language=language,
        path=path,
        range=symbol_range or _range(),
        selection_range=selection_range or Range(start_line=1, start_col=6, end_line=1, end_col=11),
        parent_id=parent_id,
        signature=signature,
        doc=doc,
        source=source,
        confidence=confidence,
        file_hash=file_hash,
        index_version=index_version,
        stale=stale,
    )


def test_core_models_inherit_from_pydantic_base_model() -> None:
    models = [
        Range,
        Location,
        Symbol,
        Diagnostic,
        ToolResult,
        ToolError,
        ToolMeta,
        CodeContext,
        HoverInfo,
    ]

    for model in models:
        assert issubclass(model, BaseModel)


@pytest.mark.parametrize(
    "field",
    ["start_line", "start_col", "end_line", "end_col"],
)
def test_range_rejects_negative_coordinates(field: str) -> None:
    values = {"start_line": 0, "start_col": 0, "end_line": 0, "end_col": 1}
    values[field] = -1

    with pytest.raises(ValidationError):
        _ = Range(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"start_line": 2, "start_col": 0, "end_line": 1, "end_col": 0},
        {"start_line": 1, "start_col": 5, "end_line": 1, "end_col": 4},
    ],
)
def test_range_rejects_reversed_half_open_ranges(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        _ = Range(**values)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/src/service.py",
        "src\\service.py",
        "../service.py",
        "src/../service.py",
        "src//service.py",
        "./src/service.py",
        "C:service.py",
        "C:/service.py",
    ],
)
def test_location_rejects_invalid_workspace_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        _ = Location(path=bad_path, range=_range())


def test_symbol_id_is_deterministic_and_uses_qualified_name() -> None:
    symbol = _symbol()
    expected = hashlib.sha1(
        "python:src/service.py:UserService.login:deadbeefcafebabe".encode("utf-8")
    ).hexdigest()[:16]

    assert symbol.id == expected
    assert _symbol().id == expected


def test_symbol_id_falls_back_to_name_when_qualified_name_is_missing() -> None:
    symbol = _symbol(qualified_name=None)
    expected = hashlib.sha1(
        "python:src/service.py:login:deadbeefcafebabe".encode("utf-8")
    ).hexdigest()[:16]

    assert symbol.id == expected


def test_enum_values_serialize_as_lowercase_english_strings() -> None:
    symbol_dump = _symbol(kind=SymbolKind.CLASS).model_dump(mode="json")
    diagnostic = Diagnostic(
        path="src/service.py",
        range=_range(),
        severity=DiagnosticSeverity.ERROR,
        message="Name is not defined",
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint="fingerprint-1",
    )

    assert symbol_dump["kind"] == "class"
    assert diagnostic.model_dump(mode="json")["severity"] == "error"


def test_tool_result_serializes_stably() -> None:
    result = ToolResult[Location](
        ok=True,
        data=_location(),
        meta=ToolMeta(elapsed_ms=7, sources_used=["tree_sitter", "lsp"]),
    )

    assert result.model_dump(mode="json") == {
        "ok": True,
        "data": {
            "path": "src/service.py",
            "range": {
                "start_line": 1,
                "start_col": 2,
                "end_line": 3,
                "end_col": 4,
            },
        },
        "error": None,
        "meta": {
            "elapsed_ms": 7,
            "truncated": False,
            "more_available": False,
            "sources_used": ["tree_sitter", "lsp"],
            "flags": [],
        },
    }


def test_tool_meta_uses_independent_sources_used_defaults() -> None:
    first = ToolMeta()
    second = ToolMeta()

    first.sources_used.append("lsp")
    first.flags.append("recovered")

    assert first.sources_used == ["lsp"]
    assert second.sources_used == []
    assert first.flags == ["recovered"]
    assert second.flags == []


def test_tool_error_keeps_english_code_with_chinese_message_and_hint() -> None:
    result = ToolResult[Location](
        ok=False,
        error=ToolError(
            code="symbol_not_found",
            message="未找到指定符号。",
            hint="symbol 已被修改或删除，请用 code_search 重定位。",
        ),
    )

    dumped = result.model_dump(mode="json")

    assert dumped["error"] == {
        "code": "symbol_not_found",
        "message": "未找到指定符号。",
        "hint": "symbol 已被修改或删除，请用 code_search 重定位。",
    }
    assert dumped["meta"]["elapsed_ms"] == 0


def test_code_context_and_hover_info_are_typed_models() -> None:
    symbol = _symbol()
    context = CodeContext(
        target_symbol=symbol,
        signature=symbol.signature,
        body="return self.token_service.issue(user)",
        parents=[],
        imports=["from auth import Token"],
        nearby_symbols=[_symbol(name="logout", qualified_name="UserService.logout")],
        truncated=False,
    )
    hover = HoverInfo(contents="(method) UserService.login(user: str) -> Token", range=_range(), source="lsp")

    dumped = context.model_dump(mode="json")

    assert dumped["target_symbol"]["id"] == symbol.id
    assert dumped["nearby_symbols"][0]["name"] == "logout"
    assert hover.model_dump(mode="json")["range"]["start_line"] == 1
