"""Contract shell for workflow/agent verification baseline isolation."""

import pytest


@pytest.mark.xfail(reason="REVIEW_RESULT_BLOCKER", strict=True)
def test_workflow_and_agent_baselines_are_isolated():
    """workflow baseline 与 agent baseline 必须互不污染。"""
    assert False, "T13 must implement isolated baseline buckets by call_source"
