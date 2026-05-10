"""Tests for fixed-rule diagnostic message normalization."""

from __future__ import annotations

from src.code_intel.verifier import normalize_message


def test_replaces_quoted_strings_with_stable_placeholder() -> None:
    assert normalize_message("Cannot find name 'foo'") == "Cannot find name <quoted>"
    assert normalize_message('Type "Foo" is not assignable to `Bar`') == "Type <quoted> is not assignable to <quoted>"


def test_replaces_integers_floats_and_line_references() -> None:
    message = "Expected 3.14 on line 42, got 7 at src/app.py:12:5 and column 9"

    assert normalize_message(message) == (
        "Expected <float> on line <line>, got <int> at src/app.py:<line>:<column> and column <column>"
    )


def test_collapses_repeated_whitespace_after_replacements() -> None:
    assert normalize_message("Value   'foo'\n  appears on lines 10-12") == "Value <quoted> appears on lines <line>"
