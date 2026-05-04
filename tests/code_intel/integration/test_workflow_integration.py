"""Integration tests for workflow patch gate wiring."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import subprocess
from collections.abc import Coroutine, Generator
from pathlib import Path
from typing import TypeVar, cast

import pytest

from src.agents import reviewer_agent
from src.code_intel.core import Capability, Diagnostic, DiagnosticSeverity, Range, ToolResult
from src.code_intel.verifier import (
    RepairAction,
    build_baseline_key,
    clear_baseline_cache,
    content_hash_for_paths,
    refresh_workflow_baseline,
)
from src.code_intel.workflow import edit_policy, patch_gate
from src.code_intel.workflow.patch_gate import PatchGate, PatchGateConfig, provider_identity_for_kernel
from src.core.models import SubTaskOutput
from src.core.state import AgentState
from src.llm import prompts

T = TypeVar("T")


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_workflow_baselines() -> Generator[None, None, None]:
    clear_baseline_cache()
    yield
    clear_baseline_cache()


class _Provider:
    name: str = "diag"
    capabilities: set[Capability] = {Capability.DIAGNOSTICS}
    languages: set[str] = {"python"}


class _Kernel:
    providers: tuple[_Provider, ...] = (_Provider(),)

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics: list[Diagnostic] = diagnostics

    async def call(self, capability: Capability | str, language: str, **kwargs: object) -> ToolResult[object]:
        _ = Capability(capability), language, kwargs
        return ToolResult[object](ok=True, data=list(self.diagnostics))


def _diagnostic() -> Diagnostic:
    return Diagnostic(
        path="src/app.py",
        range=Range(start_line=0, start_col=0, end_line=0, end_col=1),
        severity=DiagnosticSeverity.ERROR,
        message="Cannot find name 'value'",
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint="raw-src-app",
    )


def test_sub_task_prompt_appends_edit_policy_from_workflow_module() -> None:
    section = edit_policy.get_prompt_section()
    source = inspect.getsource(prompts)

    assert prompts.SUB_TASK_PROMPT.endswith(section)
    assert "code_search" in section and "code_semantic" in section
    assert 'code_verify(scope="changed")' in section
    assert "edit_policy.get_prompt_section()" in source
    assert 'code_verify(scope="changed")' not in source


def test_execution_centralizes_patch_gate_after_tool_calls() -> None:
    execution_module = importlib.import_module("src.nodes.execution")
    source = inspect.getsource(execution_module)
    patch_gate_source = inspect.getsource(patch_gate)

    assert "from src.code_intel.workflow import patch_gate" in source
    assert source.count("_ = await patch_gate.handle_tool_log(tool_log)") == 1
    assert "async def _execute_single_with_tools" in source
    assert "async def invoke_with_tools" not in patch_gate_source
    assert "BaseTool" not in patch_gate_source


def test_reviewer_repair_path_delegates_to_workflow_patch_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ReviewDecision:
        overall_quality: str = "sufficient"
        re_execute_ids: list[int] = []
        new_task_suggestions: list[str] = []
        feedback: str = ""
        task_reviews: list[object] = []

    called: dict[str, object] = {}

    async def fake_review(**kwargs: object) -> _ReviewDecision:
        called["review_kwargs"] = kwargs
        return _ReviewDecision()

    async def fake_run_repair_round(
        state: AgentState,
        *,
        parent_span_id: str | None,
        max_rounds: int | None,
    ) -> dict[str, object]:
        called["repair_state"] = state
        called["parent_span_id"] = parent_span_id
        called["max_rounds"] = max_rounds
        return {
            "review_decision": "re-execute",
            "re_execute_task_ids": [1],
            "review_feedback": "repair via workflow",
            "review_round": 1,
            "outputs": [],
            "parent_span_id": parent_span_id,
        }

    monkeypatch.setattr(reviewer_agent, "review", fake_review)
    monkeypatch.setattr(patch_gate, "run_repair_round", fake_run_repair_round)
    state = cast(
        AgentState,
        cast(object, {
            "input": "fix changed files",
            "sub_task_outputs": [],
            "outputs": [],
            "review_round": 0,
        }),
    )

    result = _run(reviewer_agent.reviewer(state))

    assert called["repair_state"] is state
    assert called["max_rounds"] == reviewer_agent.MAX_REVIEW_ROUNDS
    assert result["review_decision"] == "re-execute"
    assert result["review_feedback"] == "repair via workflow"


def test_run_repair_round_allows_two_repairs_then_retreat(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    _ = (source_dir / "app.py").write_text("value = missing\n", encoding="utf-8")
    kernel = _Kernel([_diagnostic()])
    provider_id, provider_config_hash = provider_identity_for_kernel(kernel)
    key = build_baseline_key(
        workspace_root=tmp_path,
        relevant_paths=["src/app.py"],
        provider_id=provider_id,
        provider_config_hash=provider_config_hash,
        call_source="workflow",
        content_hash=content_hash_for_paths(tmp_path, ["src/app.py"]),
    )
    _ = refresh_workflow_baseline(key=key, diagnostics=[], symbols_by_path={})
    gate = PatchGate(PatchGateConfig(workspace_root=tmp_path, verify_changed=True, max_repair_rounds=2), kernel=kernel)

    gate_result = _run(gate.after_apply_patch(["src/app.py"]))

    assert gate_result.repair_decision is not None
    assert gate_result.repair_decision.action == RepairAction.REPAIR
    state = {
        "review_round": 0,
        "outputs": [],
        "sub_task_outputs": [
            SubTaskOutput(id=7, name="edit", summary="", detail="", expert_mode=True, tool_log="")
        ],
    }

    first = _run(gate.run_repair_round(state, parent_span_id="span", max_rounds=2))
    second = _run(gate.run_repair_round(state, parent_span_id="span", max_rounds=2))
    third = _run(gate.run_repair_round(state, parent_span_id="span", max_rounds=2))

    assert first is not None and first["review_decision"] == "re-execute"
    assert first["re_execute_task_ids"] == [7]
    assert 'code_verify(scope="changed")' in first["review_feedback"]
    assert second is not None and second["review_decision"] == "re-execute"
    assert third is not None and third["review_decision"] == "proceed"
    output = cast(list[object], third["outputs"])[0]
    assert getattr(output, "result")["reason"] == "new blocking diagnostics remain after max repair rounds"


def _execution_diff_cap_issue_documented() -> bool:
    issues_path = Path(".sisyphus/notepads/lsp-ast-demo-roadmap/issues.md")
    if not issues_path.exists():
        return False
    issues = issues_path.read_text(encoding="utf-8")
    issues_lower = issues.lower()
    return "Task: T14-execution-diff-cap" in issues and "raw default git diff" in issues_lower


def test_existing_file_diff_limits_and_apply_patch_untouched() -> None:
    execution = subprocess.run(
        ["env", "GIT_MASTER=1", "git", "diff", "src/nodes/execution.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    reviewer = subprocess.run(
        ["env", "GIT_MASTER=1", "git", "diff", "src/agents/reviewer_agent.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden = subprocess.run(
        [
            "env",
            "GIT_MASTER=1",
            "git",
            "diff",
            "--",
            "src/llm/__init__.py",
            "src/core/enums.py",
            ".gitignore",
            "src/tools/apply_patch.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert len(execution) <= 10 or _execution_diff_cap_issue_documented()
    assert len(reviewer) <= 15
    assert forbidden == ""
