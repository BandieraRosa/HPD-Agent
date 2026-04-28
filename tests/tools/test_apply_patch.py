# pyright: reportUnknownMemberType=false
from typing import cast

import pytest

from src.tools.apply_patch import (
    DUPLICATE_FILE_OPERATION,
    INVALID_ENVELOPE,
    MALFORMED_HUNK,
    MALFORMED_SECTION,
    NOT_IMPLEMENTED,
    PatchError,
    apply_patch,
    parse_patch_text,
)


def _patch(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _assert_parser_error(patch_text: str, code: str) -> None:
    with pytest.raises(PatchError) as exc_info:
        _ = parse_patch_text(patch_text)

    error = exc_info.value
    assert error.code == code
    assert error.to_error_result().startswith(f"[Error] {code}:")


def test_apply_patch_tool_name_is_stable():
    assert apply_patch.name == "apply_patch"


def test_apply_patch_tool_args_include_public_scaffold_parameters():
    tool_args = cast(dict[str, object], apply_patch.args)

    assert "patch_text" in tool_args
    assert "dry_run" in tool_args


def test_apply_patch_placeholder_output_is_structured_error():
    result = cast(
        str,
        apply_patch.invoke(
            {
                "patch_text": "*** Begin Patch\n*** End Patch\n",
                "dry_run": True,
            }
        ),
    )

    assert result.startswith(f"[Error] {NOT_IMPLEMENTED}:")
    assert "not implemented yet" in result
    assert "dry-run mode" in result


def test_parse_valid_add_update_delete_envelope():
    document = parse_patch_text(
        _patch(
            "*** Begin Patch",
            "*** Add File: docs/new.txt",
            "+hello",
            "+",
            "++literal",
            "",
            "*** Update File: src/example.py",
            "@@ -1,2 +1,3 @@",
            " keep",
            "-old",
            "+new",
            "+added",
            "\\ No newline at end of file",
            "",
            "@@ -10,1 +11,0 @@",
            "-remove",
            "*** Delete File: docs/old.txt",
            "*** End Patch",
        )
    )

    assert [operation.kind for operation in document.operations] == ["add", "update", "delete"]

    add_operation = document.operations[0]
    assert add_operation.path == "docs/new.txt"
    assert [line.content for line in add_operation.added_lines] == ["hello", "", "+literal"]

    update_operation = document.operations[1]
    assert update_operation.path == "src/example.py"
    assert len(update_operation.hunks) == 2
    first_hunk = update_operation.hunks[0]
    assert (first_hunk.old_start, first_hunk.old_count) == (1, 2)
    assert (first_hunk.new_start, first_hunk.new_count) == (1, 3)
    assert [line.kind for line in first_hunk.lines] == ["context", "delete", "add", "add"]
    assert [line.content for line in first_hunk.lines] == ["keep", "old", "new", "added"]
    assert first_hunk.lines[-1].no_newline_at_end is True

    second_hunk = update_operation.hunks[1]
    assert (second_hunk.old_start, second_hunk.old_count) == (10, 1)
    assert (second_hunk.new_start, second_hunk.new_count) == (11, 0)
    assert [line.kind for line in second_hunk.lines] == ["delete"]

    assert document.operations[2].path == "docs/old.txt"


@pytest.mark.parametrize(
    ("patch_text", "code"),
    [
        (
            _patch("*** Add File: a.txt", "+content", "*** End Patch"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** Add File: a.txt", "+content"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** Add File: a.txt", "+content", "*** End Patch", "trailing"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** End Patch"),
            INVALID_ENVELOPE,
        ),
        (
            _patch("*** Begin Patch", "*** Move File: a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "diff --git a/a.txt b/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "--- a/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch("*** Begin Patch", "+++ b/a.txt", "*** End Patch"),
            MALFORMED_SECTION,
        ),
        (
            _patch(
                "*** Begin Patch",
                "*** Add File: a.txt",
                "+content",
                "*** Delete File: a.txt",
                "*** End Patch",
            ),
            DUPLICATE_FILE_OPERATION,
        ),
    ],
)
def test_parser_rejects_invalid_envelope_and_section_errors(patch_text: str, code: str):
    _assert_parser_error(patch_text, code)


@pytest.mark.parametrize(
    "patch_text",
    [
        _patch("*** Begin Patch", "diff --git a/a.txt b/a.txt", "*** End Patch"),
        _patch("*** Begin Patch", "--- a/a.txt", "*** End Patch"),
        _patch("*** Begin Patch", "+++ b/a.txt", "*** End Patch"),
    ],
)
def test_parser_rejects_git_diff_snippets(patch_text: str):
    _assert_parser_error(patch_text, MALFORMED_SECTION)


@pytest.mark.parametrize(
    "patch_text",
    [
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1 +1 @@",
            "*** End Patch",
        ),
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1,1 +1,1 @@",
            "body without prefix",
            "*** End Patch",
        ),
        _patch(
            "*** Begin Patch",
            "*** Update File: a.txt",
            "@@ -1,1 +1,1 @@",
            " same",
            "+extra",
            "*** End Patch",
        ),
    ],
)
def test_parser_rejects_malformed_hunk_syntax(patch_text: str):
    _assert_parser_error(patch_text, MALFORMED_HUNK)


def test_parser_rejects_eof_marker_without_previous_hunk_body_line():
    patch_text = _patch(
        "*** Begin Patch",
        "*** Update File: a.txt",
        "@@ -0,0 +0,0 @@",
        "\\ No newline at end of file",
        "*** End Patch",
    )

    _assert_parser_error(patch_text, MALFORMED_HUNK)
