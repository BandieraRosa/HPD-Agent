"""Contract shell for trace metadata redaction."""

import pytest


@pytest.mark.xfail(reason="REVIEW_RESULT_BLOCKER", strict=True)
def test_trace_metadata_redaction():
    """trace metadata 必须通过白名单过滤，不能泄漏源码或绝对路径。"""
    assert False, "T16 must implement trace metadata redaction"
