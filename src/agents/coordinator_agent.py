"""Coordinator Agent: decides whether to decompose a complex task.

This is a thin decision shell — all business logic lives in nodes/planning.py.
The agent only decides "yes/no" and delegates actual decomposition to the planning node.

The coordinator writes ``tasks`` and ``decomposition_result`` to state;
the graph edges then drive ``scheduler_node`` (execution) and ``synthesizer``
(streaming) as separate nodes.
"""

import uuid

from src.core.models import AgentMeta, TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.nodes.planning import decompose as decompose_node, replan as replan_node


async def coordinate(state: AgentState) -> AgentState:
    """Decide to decompose a complex task, delegating to the planning node.

    Supports two modes:
        - Initial planning: when review_decision is None or tasks is empty
        - Re-planning: when review_decision is 'add_tasks' (adds new sub-tasks)

    Reads from state:
        - ``input``: the original user query
        - ``review_decision``: if 'add_tasks', enters re-planning mode
        - ``tasks``: existing sub-tasks (for re-planning)
        - ``sub_task_outputs``: existing outputs (for re-planning)
        - ``review_feedback``: reviewer feedback (for re-planning)

    Writes to state:
        - ``tasks``: the decomposed DAG (merged with new tasks in re-planning)
        - ``decomposition_result``: raw LLM planner result
        - ``agent_history``: metadata about coordinator invocation
    """
    tracer = get_tracer()
    with tracer.span("coordinator") as parent_span_id:
        coordinator_id = f"coordinator-{uuid.uuid4().hex[:8]}"

        is_replan = (
            state.get("review_decision") == "add_tasks"
            and state.get("tasks")
        )

        if is_replan:
            # Re-planning mode: add new sub-tasks based on reviewer feedback
            existing_tasks = list(state.get("tasks", []))
            existing_outputs = list(state.get("sub_task_outputs", []))
            feedback = state.get("review_feedback", "")
            suggestions = []  # Extracted from reviewer's last output

            # Get suggestions from the reviewer's output
            for o in reversed(state.get("outputs", [])):
                if o.node == "reviewer":
                    suggestions = o.result.get("new_task_suggestions", [])
                    break

            next_id = max(t.id for t in existing_tasks) + 1

            print(f"\n[CoordinatorAgent {coordinator_id}] 根据审查反馈追加新子任务...")

            new_tasks, result = await replan_node(
                query=state["input"],
                existing_tasks=existing_tasks,
                existing_outputs=existing_outputs,
                feedback=feedback,
                suggestions=suggestions,
                next_id=next_id,
            )

            # Merge: existing tasks + new tasks
            all_tasks = existing_tasks + new_tasks

            tin, tout, model = TokenTrackerCallback.snapshot()
            tracer.record_tokens(parent_span_id, tokens_in=tin, tokens_out=tout, model=model)

            coordinator_meta = AgentMeta(
                role="coordinator",
                agent_id=coordinator_id,
                sub_task_id=None,
                result_summary=f"追加 {len(new_tasks)} 个子任务（共 {len(all_tasks)} 个）",
            )

            return {
                "tasks": all_tasks,
                "decomposition_result": result,
                "review_decision": None,  # Reset so scheduler runs normally
                "outputs": [
                    *state.get("outputs", []),
                    TaskOutput(
                        node="coordinator",
                        result={
                            "coordinator_id": coordinator_id,
                            "mode": "replan",
                            "new_tasks": [
                                {"id": t.id, "name": t.name, "depends": t.depends}
                                for t in new_tasks
                            ],
                            "total_sub_task_count": len(all_tasks),
                        },
                    ),
                ],
                "agent_history": [
                    *state.get("agent_history", []),
                    coordinator_meta,
                ],
                "parent_span_id": parent_span_id,
            }

        # Initial planning mode
        print(f"\n[CoordinatorAgent {coordinator_id}] 收到复杂任务，开始规划...")

        tasks, result = await decompose_node(state["input"])

        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(parent_span_id, tokens_in=tin, tokens_out=tout, model=model)

        coordinator_meta = AgentMeta(
            role="coordinator",
            agent_id=coordinator_id,
            sub_task_id=None,
            result_summary=f"分解为 {len(tasks)} 个子任务",
        )

        return {
            "tasks": tasks,
            "decomposition_result": result,
            "outputs": [
                *state.get("outputs", []),
                TaskOutput(
                    node="coordinator",
                    result={
                        "coordinator_id": coordinator_id,
                        "mode": "initial",
                        "total_sub_task_count": result.total_sub_task_count,
                        "sub_tasks": [
                            {"id": t.id, "name": t.name, "depends": t.depends}
                            for t in tasks
                        ],
                    },
                ),
            ],
            "agent_history": [
                *state.get("agent_history", []),
                coordinator_meta,
            ],
            "parent_span_id": parent_span_id,
        }
