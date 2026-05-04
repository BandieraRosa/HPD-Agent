"""Pyright-shaped samples for diagnostic normalization — 14 realistic templates."""

from __future__ import annotations

from src.code_intel.verifier import normalize_message


def test_pyright_undefined_name_sample() -> None:
    assert normalize_message("Cannot find name 'missing_symbol'") == "Cannot find name <quoted>"


def test_pyright_import_resolution_sample() -> None:
    assert (
        normalize_message('Import "missing_package.submodule" could not be resolved from source')
        == "Import <quoted> could not be resolved from source"
    )


def test_pyright_argument_type_sample() -> None:
    sample = 'Argument of type "int" cannot be assigned to parameter "value" of type "str" in function "run"'

    assert normalize_message(sample) == (
        "Argument of type <quoted> cannot be assigned to parameter <quoted> of type <quoted> in function <quoted>"
    )


def test_pyright_literal_and_line_number_sample() -> None:
    sample = 'Expression of type "Literal[3]" cannot be assigned to declared type "float" at line 128'

    assert normalize_message(sample) == "Expression of type <quoted> cannot be assigned to declared type <quoted> at line <line>"


def test_pyright_type_assignability_sample() -> None:
    sample = 'Type "int" is not assignable to type "str"'
    assert normalize_message(sample) == "Type <quoted> is not assignable to type <quoted>"


def test_pyright_missing_attribute_sample() -> None:
    sample = 'Cannot access attribute "missing_field" for class "ServiceClient"'
    assert normalize_message(sample) == "Cannot access attribute <quoted> for class <quoted>"


def test_pyright_missing_parameter_sample() -> None:
    sample = (
        'Function with parameter "callback" cannot be called without argument'
    )
    assert normalize_message(sample) == "Function with parameter <quoted> cannot be called without argument"


def test_pyright_missing_argument_sample() -> None:
    sample = 'Argument missing for parameter "timeout" in function "connect"'
    assert normalize_message(sample) == "Argument missing for parameter <quoted> in function <quoted>"


def test_pyright_positional_argument_count_sample() -> None:
    sample = "Expected 3 positional arguments, got 1"
    assert normalize_message(sample) == "Expected <int> positional arguments, got <int>"


def test_pyright_module_attribute_sample() -> None:
    sample = 'Module "config" has no attribute "DEFAULT_TIMEOUT"'
    assert normalize_message(sample) == "Module <quoted> has no attribute <quoted>"


def test_pyright_typealias_usage_sample() -> None:
    sample = '"ConnectionError" is declared as a TypeAlias but cannot be used as a regular type'
    assert normalize_message(sample) == "<quoted> is declared as a TypeAlias but cannot be used as a regular type"


def test_pyright_operator_support_sample() -> None:
    sample = 'Operator "+" is not supported between types "int" and "str"'
    assert normalize_message(sample) == "Operator <quoted> is not supported between types <quoted> and <quoted>"


def test_pyright_final_base_class_sample() -> None:
    sample = 'Base class "FinalBase" is marked as final and cannot be extended'
    assert normalize_message(sample) == "Base class <quoted> is marked as final and cannot be extended"


def test_pyright_not_iterable_sample() -> None:
    sample = (
        'Expression of type "None" is not iterable; type is determined to be "None"'
    )
    assert normalize_message(sample) == (
        "Expression of type <quoted> is not iterable; type is determined to be <quoted>"
    )
