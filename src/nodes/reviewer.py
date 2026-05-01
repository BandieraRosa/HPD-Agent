"""Reviewer node: evaluates sub-task quality and decides next action.

Decisions:
  - proceed     → all tasks are good, go to synthesizer
  - re-execute  → some tasks need re-running
  - add_tasks   → new analysis directions discovered
"""

from src.core.models import ReviewerDecision, TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_structured_llm, REVIEW_PROMPT

MAX_REVIEW_ROUNDS = 2


def _format_sub_task_results(outputs) -> str:
    lines = []
    for o in outputs:
        status = "失败" if o.summary.startswith("[失败]") else "成功"
        lines.append(
            f"子任务 {o.id} ({o.name}) — {status}\n"
            f"  摘要: {o.summary}\n"
            f"  详情: {o.detail[:500]}"
        )
    return "\n\n".join(lines)


async def reviewer(state: AgentState) -> AgentState:
    """Evaluate sub-task quality and decide: proceed, re-execute, or add tasks."""
    tracer = get_tracer()
    parent_id = state.get("parent_span_id") or None

    with tracer.span("reviewer", parent_id=parent_id) as span_id:
        current_round = state.get("review_round", 0)
        outputs = state.get("sub_task_outputs", [])

        # Max rounds reached — force proceed
        if current_round >= MAX_REVIEW_ROUNDS:
            print(f"\n[Reviewer] 已达最大审查轮次 ({MAX_REVIEW_ROUNDS})，直接进入合成。")
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
                "parent_span_id": span_id,
            }

        # Call LLM for quality review
        print(f"\n[Reviewer] 第 {current_round + 1} 轮质量审查中...")

        results_text = _format_sub_task_results(outputs)
        prompt = REVIEW_PROMPT.format(
            query=state["input"],
            sub_task_results=results_text,
            round=current_round + 1,
            max_rounds=MAX_REVIEW_ROUNDS,
        )

        llm = get_structured_llm(ReviewerDecision)
        decision: ReviewerDecision = await llm.ainvoke(prompt)

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

        # Map to enum
        quality = decision.overall_quality
        if quality == "needs_improvement":
            review_decision = "re-execute"
        elif quality == "needs_more_tasks":
            review_decision = "add_tasks"
        else:
            review_decision = "proceed"

        # Enforce max rounds at LLM level too
        if review_decision != "proceed" and current_round + 1 >= MAX_REVIEW_ROUNDS:
            print(f"[Reviewer] 已达最大轮次，覆盖为 proceed。")
            review_decision = "proceed"
            decision.re_execute_ids = []
            decision.new_task_suggestions = []

        re_ids = decision.re_execute_ids if review_decision == "re-execute" else []

        # Print summary
        for tr in decision.task_reviews:
            icon = {"good": "✓", "weak": "△", "failed": "✗"}.get(tr.quality, "?")
            print(f"  [{icon}] 子任务 {tr.sub_task_id}: {tr.quality} — {tr.reasoning}")

        if review_decision == "re-execute":
            print(f"[Reviewer] 要求重做子任务: {re_ids}")
        elif review_decision == "add_tasks":
            print(f"[Reviewer] 建议新增 {len(decision.new_task_suggestions)} 个子任务")
        else:
            print("[Reviewer] 质量合格，进入合成。")

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
        "parent_span_id": span_id,
    }
