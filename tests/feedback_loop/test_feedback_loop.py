"""Test suite for the multi-agent feedback loop.

Verifies:
  1. Scheduler execute_only filter — partial re-execution.
  2. Reviewer node — max rounds enforcement, decision mapping.
  3. Graph routing — _route_after_review conditional edges.
"""

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, "/root/projects/evo_agent")

from src.core.models import SubTask, SubTaskOutput, ReviewerDecision, ReviewTaskResult
from src.core.enums import ReviewDecision
from src.core.state import AgentState
from src.nodes.scheduler import run_all, RetryConfig


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_output(tid: int, name: str, detail: str = "", summary: str = "") -> SubTaskOutput:
    return SubTaskOutput(
        id=tid,
        name=name,
        detail=detail or f"Detail for {name}",
        summary=summary or f"{name} done.",
    )


def _make_task(tid: int, name: str, depends: list[int] | None = None) -> SubTask:
    return SubTask(id=tid, name=name, depends=depends or [])


async def _simple_executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    await asyncio.sleep(0)
    return _make_output(task_id, task_name)


# --------------------------------------------------------------------------
# 1. Scheduler execute_only filter
# --------------------------------------------------------------------------


class TestSchedulerExecuteOnly(unittest.IsolatedAsyncioTestCase):
    """Tests for the execute_only partial re-execution filter."""

    async def test_full_execution_when_no_filter(self) -> None:
        """Without execute_only, all tasks run (backward compat)."""
        tasks = [_make_task(1, "A"), _make_task(2, "B")]
        statuses, outputs = await run_all(tasks, _simple_executor, "ctx")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(statuses[1], "done")
        self.assertEqual(statuses[2], "done")

    async def test_filter_executes_only_specified_tasks(self) -> None:
        """Only tasks in execute_only are actually executed."""
        tasks = [_make_task(1, "A"), _make_task(2, "B"), _make_task(3, "C")]
        existing = [
            _make_output(1, "A", summary="A original"),
            _make_output(2, "B", summary="B original"),
            _make_output(3, "C", summary="C original"),
        ]

        executed_ids: list[int] = []

        async def tracking_executor(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            executed_ids.append(task_id)
            return _make_output(task_id, task_name, summary=f"{task_name} re-done")

        statuses, outputs = await run_all(
            tasks, tracking_executor, "ctx",
            execute_only={2},
            existing_outputs=existing,
        )

        # Only task 2 was executed
        self.assertEqual(executed_ids, [2])

        # All tasks are "done"
        self.assertEqual(statuses[1], "done")
        self.assertEqual(statuses[2], "done")
        self.assertEqual(statuses[3], "done")

    async def test_filter_preserves_existing_outputs(self) -> None:
        """Non-filtered tasks retain their original outputs."""
        tasks = [_make_task(1, "A"), _make_task(2, "B")]
        existing = [
            _make_output(1, "A", detail="original A detail", summary="A original"),
            _make_output(2, "B", detail="original B detail", summary="B original"),
        ]

        statuses, outputs = await run_all(
            tasks, _simple_executor, "ctx",
            execute_only={2},
            existing_outputs=existing,
        )

        output_map = {o.id: o for o in outputs}
        # Task 1 output is carried forward from existing
        self.assertEqual(output_map[1].summary, "A original")
        # Task 2 output is from the new execution
        self.assertEqual(output_map[2].summary, "B done.")

    async def test_filter_with_dependency_chain(self) -> None:
        """Filtered task can depend on a skipped task (its output is in cache)."""
        tasks = [
            _make_task(1, "A"),
            _make_task(2, "B", depends=[1]),
            _make_task(3, "C", depends=[2]),
        ]
        existing = [
            _make_output(1, "A", detail="A was done", summary="A done"),
            _make_output(2, "B", detail="B was done", summary="B done"),
            _make_output(3, "C", detail="C was done", summary="C done"),
        ]

        captured_ctx: dict[int, str] = {}

        async def ctx_capture(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            captured_ctx[task_id] = context
            return _make_output(task_id, task_name, summary=f"{task_name} re-done")

        # Only re-execute task 3, which depends on task 2 (skipped)
        statuses, outputs = await run_all(
            tasks, ctx_capture, "original question",
            execute_only={3},
            existing_outputs=existing,
        )

        self.assertEqual(statuses[1], "done")
        self.assertEqual(statuses[2], "done")
        self.assertEqual(statuses[3], "done")

        # Task 3's context should include upstream results from tasks 1 and 2
        ctx3 = captured_ctx[3]
        self.assertIn("original question", ctx3)
        self.assertIn("A was done", ctx3)
        self.assertIn("B was done", ctx3)

    async def test_filter_multiple_tasks(self) -> None:
        """Multiple tasks can be in execute_only."""
        tasks = [_make_task(1, "A"), _make_task(2, "B"), _make_task(3, "C")]
        existing = [_make_output(i, name) for i, name in [(1, "A"), (2, "B"), (3, "C")]]

        executed_ids: list[int] = []

        async def tracking(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            executed_ids.append(task_id)
            return _make_output(task_id, task_name)

        statuses, outputs = await run_all(
            tasks, tracking, "ctx",
            execute_only={1, 3},
            existing_outputs=existing,
        )

        self.assertEqual(sorted(executed_ids), [1, 3])
        self.assertEqual(len(outputs), 3)

    async def test_filter_empty_execute_only(self) -> None:
        """Empty execute_only set means nothing to execute — only existing outputs."""
        tasks = [_make_task(1, "A"), _make_task(2, "B")]
        existing = [_make_output(1, "A"), _make_output(2, "B")]

        executed: list[int] = []

        async def tracking(task_id: int, task_name: str, context: str) -> SubTaskOutput:
            await asyncio.sleep(0)
            executed.append(task_id)
            return _make_output(task_id, task_name)

        statuses, outputs = await run_all(
            tasks, tracking, "ctx",
            execute_only=set(),
            existing_outputs=existing,
        )

        self.assertEqual(executed, [])
        self.assertEqual(len(outputs), 2)


# --------------------------------------------------------------------------
# 2. Reviewer node
# --------------------------------------------------------------------------


class TestReviewerNode(unittest.IsolatedAsyncioTestCase):
    """Tests for the reviewer node logic."""

    def _make_state(self, round: int = 0, outputs: list | None = None) -> AgentState:
        return {
            "input": "test query",
            "analysis": None,
            "tasks": [],
            "decomposition_result": None,
            "sub_task_statuses": {},
            "sub_task_outputs": outputs or [],
            "outputs": [],
            "final_response": "",
            "synthesis_prompt": "",
            "conversation_history": MagicMock(),
            "parent_span_id": "",
            "review_round": round,
            "review_decision": None,
            "re_execute_task_ids": [],
            "review_feedback": "",
            "new_sub_tasks": [],
            "agent_history": [],
        }

    @patch("src.nodes.reviewer.get_tracer")
    @patch("src.nodes.reviewer.get_structured_llm")
    async def test_max_rounds_forces_proceed(self, mock_get_llm, mock_tracer) -> None:
        """When review_round >= MAX_REVIEW_ROUNDS, reviewer forces proceed without LLM call."""
        mock_tracer.return_value.span.return_value.__enter__ = MagicMock(return_value=("", None))
        mock_tracer.return_value.span.return_value.__exit__ = MagicMock(return_value=False)

        state = self._make_state(round=2)  # MAX_REVIEW_ROUNDS = 2

        from src.nodes.reviewer import reviewer
        result = await reviewer(state)

        self.assertEqual(result["review_decision"], "proceed")
        self.assertEqual(result["re_execute_task_ids"], [])
        # LLM should NOT be called
        mock_get_llm.assert_not_called()

    @patch("src.nodes.reviewer.get_tracer")
    @patch("src.nodes.reviewer.get_structured_llm")
    @patch("src.nodes.reviewer.TokenTrackerCallback")
    async def test_proceed_decision(self, mock_token, mock_get_llm, mock_tracer) -> None:
        """Reviewer returns 'sufficient' → decision is 'proceed'."""
        mock_tracer.return_value.span.return_value.__enter__ = MagicMock(return_value=("", None))
        mock_tracer.return_value.span.return_value.__exit__ = MagicMock(return_value=False)
        mock_token.snapshot.return_value = (0, 0, "")

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ReviewerDecision(
            overall_quality="sufficient",
            task_reviews=[
                ReviewTaskResult(sub_task_id=1, quality="good", reasoning="well done"),
            ],
            re_execute_ids=[],
            new_task_suggestions=[],
            feedback="",
        )
        mock_get_llm.return_value = mock_llm

        outputs = [_make_output(1, "Task A")]
        state = self._make_state(round=0, outputs=outputs)

        from src.nodes.reviewer import reviewer
        result = await reviewer(state)

        self.assertEqual(result["review_decision"], "proceed")
        self.assertEqual(result["re_execute_task_ids"], [])
        self.assertEqual(result["review_round"], 1)

    @patch("src.nodes.reviewer.get_tracer")
    @patch("src.nodes.reviewer.get_structured_llm")
    @patch("src.nodes.reviewer.TokenTrackerCallback")
    async def test_re_execute_decision(self, mock_token, mock_get_llm, mock_tracer) -> None:
        """Reviewer returns 'needs_improvement' → decision is 're-execute'."""
        mock_tracer.return_value.span.return_value.__enter__ = MagicMock(return_value=("", None))
        mock_tracer.return_value.span.return_value.__exit__ = MagicMock(return_value=False)
        mock_token.snapshot.return_value = (0, 0, "")

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ReviewerDecision(
            overall_quality="needs_improvement",
            task_reviews=[
                ReviewTaskResult(sub_task_id=1, quality="good", reasoning="ok"),
                ReviewTaskResult(sub_task_id=2, quality="weak", reasoning="incomplete"),
            ],
            re_execute_ids=[2],
            new_task_suggestions=[],
            feedback="请补充更多细节",
        )
        mock_get_llm.return_value = mock_llm

        outputs = [_make_output(1, "A"), _make_output(2, "B")]
        state = self._make_state(round=0, outputs=outputs)

        from src.nodes.reviewer import reviewer
        result = await reviewer(state)

        self.assertEqual(result["review_decision"], "re-execute")
        self.assertEqual(result["re_execute_task_ids"], [2])
        self.assertIn("请补充", result["review_feedback"])
        self.assertEqual(result["review_round"], 1)

    @patch("src.nodes.reviewer.get_tracer")
    @patch("src.nodes.reviewer.get_structured_llm")
    @patch("src.nodes.reviewer.TokenTrackerCallback")
    async def test_add_tasks_decision(self, mock_token, mock_get_llm, mock_tracer) -> None:
        """Reviewer returns 'needs_more_tasks' → decision is 'add_tasks'."""
        mock_tracer.return_value.span.return_value.__enter__ = MagicMock(return_value=("", None))
        mock_tracer.return_value.span.return_value.__exit__ = MagicMock(return_value=False)
        mock_token.snapshot.return_value = (0, 0, "")

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ReviewerDecision(
            overall_quality="needs_more_tasks",
            task_reviews=[
                ReviewTaskResult(sub_task_id=1, quality="good", reasoning="ok"),
            ],
            re_execute_ids=[],
            new_task_suggestions=["需要额外的安全性分析"],
            feedback="当前分析缺少安全维度",
        )
        mock_get_llm.return_value = mock_llm

        outputs = [_make_output(1, "A")]
        state = self._make_state(round=0, outputs=outputs)

        from src.nodes.reviewer import reviewer
        result = await reviewer(state)

        self.assertEqual(result["review_decision"], "add_tasks")
        self.assertEqual(result["re_execute_task_ids"], [])

    @patch("src.nodes.reviewer.get_tracer")
    @patch("src.nodes.reviewer.get_structured_llm")
    @patch("src.nodes.reviewer.TokenTrackerCallback")
    async def test_max_rounds_override_llm_decision(self, mock_token, mock_get_llm, mock_tracer) -> None:
        """At max rounds, even if LLM says needs_improvement, override to proceed."""
        mock_tracer.return_value.span.return_value.__enter__ = MagicMock(return_value=("", None))
        mock_tracer.return_value.span.return_value.__exit__ = MagicMock(return_value=False)
        mock_token.snapshot.return_value = (0, 0, "")

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = ReviewerDecision(
            overall_quality="needs_improvement",
            task_reviews=[
                ReviewTaskResult(sub_task_id=1, quality="weak", reasoning="needs more"),
            ],
            re_execute_ids=[1],
            new_task_suggestions=[],
            feedback="redo this",
        )
        mock_get_llm.return_value = mock_llm

        # round=1, MAX=2, so after this call round becomes 2 → override
        outputs = [_make_output(1, "A")]
        state = self._make_state(round=1, outputs=outputs)

        from src.nodes.reviewer import reviewer
        result = await reviewer(state)

        self.assertEqual(result["review_decision"], "proceed")
        self.assertEqual(result["re_execute_task_ids"], [])


# --------------------------------------------------------------------------
# 3. Graph routing
# --------------------------------------------------------------------------


class TestGraphRouting(unittest.IsolatedAsyncioTestCase):
    """Tests for _route_after_review conditional edge."""

    def test_proceed_routes_to_synthesizer(self) -> None:
        from src.workflow.builder import _route_after_review
        state: AgentState = {"review_decision": "proceed"}  # type: ignore
        self.assertEqual(_route_after_review(state), "synthesizer")

    def test_re_execute_routes_to_scheduler(self) -> None:
        from src.workflow.builder import _route_after_review
        state: AgentState = {"review_decision": "re-execute"}  # type: ignore
        self.assertEqual(_route_after_review(state), "scheduler_node")

    def test_add_tasks_routes_to_coordinator(self) -> None:
        from src.workflow.builder import _route_after_review
        state: AgentState = {"review_decision": "add_tasks"}  # type: ignore
        self.assertEqual(_route_after_review(state), "coordinator")

    def test_none_routes_to_synthesizer(self) -> None:
        from src.workflow.builder import _route_after_review
        state: AgentState = {"review_decision": None}  # type: ignore
        self.assertEqual(_route_after_review(state), "synthesizer")

    def test_missing_field_routes_to_synthesizer(self) -> None:
        from src.workflow.builder import _route_after_review
        state: AgentState = {}  # type: ignore
        self.assertEqual(_route_after_review(state), "synthesizer")


# --------------------------------------------------------------------------
# 4. ReviewDecision enum
# --------------------------------------------------------------------------


class TestReviewDecisionEnum(unittest.TestCase):
    """Verify enum values match the string constants used in state."""

    def test_proceed_value(self) -> None:
        self.assertEqual(ReviewDecision.PROCEED.value, "proceed")

    def test_re_execute_value(self) -> None:
        self.assertEqual(ReviewDecision.RE_EXECUTE.value, "re-execute")

    def test_add_tasks_value(self) -> None:
        self.assertEqual(ReviewDecision.ADD_TASKS.value, "add_tasks")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
