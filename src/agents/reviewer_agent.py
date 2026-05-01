"""Reviewer Agent: evaluates sub-task quality and decides next action.

This is a decision shell — all LLM logic lives in nodes/reviewer.py.
The agent handles:
  - Round control (max rounds enforcement)
  - Decision mapping (quality → proceed / re-execute / add_tasks)
  - User-facing output (print summaries)
"""

import uuid

from src.core.models import TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer
from src.nodes.reviewer import review

MAX_REVIEW_ROUNDS = 2


async def reviewer(state: AgentState) -> AgentState:
    """Evaluate sub-task quality and decide: proceed, re-execute, or add tasks.

    Reads from state:
        - ``input``: original user query
        - ``sub_task_outputs``: completed sub-task results
        - ``review_round``: current round counter

    Writes to state:
        - ``review_decision``: 'proceed' | 're-execute' | 'add_tasks'
        - ``re_execute_task_ids``: IDs to re-run (when re-execute)
        - ``review_feedback``: guidance for re-execution or re-planning
        - ``review_round``: incremented
    """
    tracer = get_tracer()
    with tracer.span("reviewer") as parent_span_id:
        reviewer_id = f"reviewer-{uuid.uuid4().hex[:8]}"
        current_round = state.get("review_round", 0)
        outputs = state.get("sub_task_outputs", [])

        # ── Round control: max rounds → force proceed ───────────────
        if current_round >= MAX_REVIEW_ROUNDS:
            print(f"\n[ReviewerAgent {reviewer_id}] 已达最大审查轮次 ({MAX_REVIEW_ROUNDS})，直接进入合成。")
            return {
                "review_decision": "proceed",
                "re_execute_task_ids": [],
                "review_feedback": "",
                "review_round": current_round + 1,
                "outputs": [
                    *state.get("outputs", []),
                    TaskOutput(
                        node="reviewer",
                        result={"decision": "proceed", "reason": "max_rounds_reached"},
                    ),
                ],
                "parent_span_id": parent_span_id,
            }

        # ── Delegate to node for LLM call ───────────────────────────
        print(f"\n[ReviewerAgent {reviewer_id}] 第 {current_round + 1} 轮质量审查中...")
        decision = await review(
            query=state["input"],
            outputs=outputs,
            current_round=current_round,
            max_rounds=MAX_REVIEW_ROUNDS,
        )

        # ── Decision mapping ────────────────────────────────────────
        quality = decision.overall_quality
        if quality == "needs_improvement":
            review_decision = "re-execute"
        elif quality == "needs_more_tasks":
            review_decision = "add_tasks"
        else:
            review_decision = "proceed"

        # Enforce max rounds at decision level too
        if review_decision != "proceed" and current_round + 1 >= MAX_REVIEW_ROUNDS:
            print(f"[ReviewerAgent {reviewer_id}] 已达最大轮次，覆盖为 proceed。")
            review_decision = "proceed"
            decision.re_execute_ids = []
            decision.new_task_suggestions = []

        re_ids = decision.re_execute_ids if review_decision == "re-execute" else []

        # ── Print summary ───────────────────────────────────────────
        for tr in decision.task_reviews:
            icon = {"good": "✓", "weak": "△", "failed": "✗"}.get(tr.quality, "?")
            print(f"  [{icon}] 子任务 {tr.sub_task_id}: {tr.quality} — {tr.reasoning}")

        if review_decision == "re-execute":
            print(f"[ReviewerAgent {reviewer_id}] 要求重做子任务: {re_ids}")
        elif review_decision == "add_tasks":
            print(f"[ReviewerAgent {reviewer_id}] 建议新增 {len(decision.new_task_suggestions)} 个子任务")
        else:
            print(f"[ReviewerAgent {reviewer_id}] 质量合格，进入合成。")

    return {
        "review_decision": review_decision,
        "re_execute_task_ids": re_ids,
        "review_feedback": decision.feedback,
        "review_round": current_round + 1,
        "outputs": [
            *state.get("outputs", []),
            TaskOutput(
                node="reviewer",
                result={
                    "decision": review_decision,
                    "overall_quality": quality,
                    "re_execute_ids": re_ids,
                    "new_task_suggestions": decision.new_task_suggestions,
                    "feedback": decision.feedback,
                    "task_reviews": [
                        {"id": tr.sub_task_id, "quality": tr.quality, "reasoning": tr.reasoning}
                        for tr in decision.task_reviews
                    ],
                },
            ),
        ],
        "parent_span_id": parent_span_id,
    }
