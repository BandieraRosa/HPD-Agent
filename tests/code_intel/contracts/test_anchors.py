"""Contract tests for code_intel anchor and target models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from src.code_intel.core.anchors import CodeTarget, TextAnchor
from src.code_intel.core.models import Location, Range


def _range() -> Range:
    return Range(start_line=0, start_col=0, end_line=0, end_col=8)


def _location() -> Location:
    return Location(path="src/service.py", range=_range())


def _anchor() -> TextAnchor:
    return TextAnchor(
        path="src/service.py",
        symbol_name="login",
        needle="def login",
        surrounding_before="class UserService:",
        surrounding_after="return token",
        occurrence=0,
    )


def test_anchor_models_inherit_from_pydantic_base_model() -> None:
    assert issubclass(TextAnchor, BaseModel)
    assert issubclass(CodeTarget, BaseModel)


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
def test_text_anchor_rejects_invalid_workspace_paths(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        _ = TextAnchor(path=bad_path, symbol_name="login", needle="def login")


def test_text_anchor_rejects_negative_occurrence() -> None:
    with pytest.raises(ValidationError):
        _ = TextAnchor(path="src/service.py", needle="def login", occurrence=-1)


def test_code_target_requires_symbol_anchor_or_location() -> None:
    with pytest.raises(ValidationError):
        _ = CodeTarget()


@pytest.mark.parametrize(
    ("target", "expected_priority"),
    [
        (CodeTarget(symbol_id="symbol-1", anchor=_anchor(), location=_location()), "symbol_id"),
        (CodeTarget(anchor=_anchor(), location=_location()), "anchor"),
        (CodeTarget(location=_location()), "location"),
    ],
)
def test_code_target_priority_order(target: CodeTarget, expected_priority: str) -> None:
    assert target.priority == expected_priority


def test_code_target_rejects_empty_symbol_id_as_missing_target() -> None:
    with pytest.raises(ValidationError):
        _ = CodeTarget(symbol_id="")
