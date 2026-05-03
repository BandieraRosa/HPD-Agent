"""Contract shell for stale symbol_id recovery after unrelated edits."""

import pytest


@pytest.mark.xfail(reason="REVIEW_RESULT_BLOCKER", strict=True)
def test_symbol_id_recovers_after_unrelated_edit():
    """旧 symbol_id 失效后应通过 (path, qualified_name) 恢复。"""
    assert False, "T10 must recover stale symbol_id via symbol_id_history"
