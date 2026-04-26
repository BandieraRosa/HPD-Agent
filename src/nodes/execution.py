"""Execution node: difficulty assessment + LLM execution for a single sub-task.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - Sub-task difficulty assessment (easy / hard)
  - LLM execution with prompt templating
  - Summary extraction

The expert agent calls this to make the "execute" decision.
"""

import re

from src.core.enums import SubTaskDifficulty
from src.core.models import SubTaskAssessmentResult, SubTaskOutput
from src.llm import (
    invoke_with_tools,
    get_structured_llm,
    SUB_TASK_ASSESSMENT_PROMPT,
    SUB_TASK_PROMPT,
)
from src.tools import tool_list


def _extract_summary(detail: str) -> str:
    """Extract concise summary from LLM output, handling JSON-fragment edge cases."""
    import json

    try:
        parsed = json.loads(detail.strip())
        if isinstance(parsed, dict) and parsed.get("summary"):
            s = parsed["summary"].strip()
            if s:
                return s
    except (json.JSONDecodeError, TypeError):
        pass

    sentences = re.split(r"(?<=[。！？.!?\n])", detail)
    candidates = [s.strip() for s in sentences if 5 < len(s.strip()) < 120]
    return candidates[-1] if candidates else detail[:80].strip()


async def execute(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    """Assess difficulty and execute a single sub-task.

    Args:
        task_id:   Sub-task ID from the DAG.
        task_name: Human-readable sub-task name.
        context:   The original user query (shared background context).

    Returns:
        SubTaskOutput with detail, summary, and expert_mode flag.
    """
    # Step 1: Difficulty assessment
    classifier = get_structured_llm(SubTaskAssessmentResult)
    assessment: SubTaskAssessmentResult = await classifier.ainvoke(
        SUB_TASK_ASSESSMENT_PROMPT.format(task_id=task_id, task_name=task_name)
    )
    is_expert = assessment.difficulty == SubTaskDifficulty.HARD

    # Step 2: Execution via LLM with tool support
    prompt = SUB_TASK_PROMPT.format(
        context=context, task_id=task_id, task_name=task_name
    )
    full_content, tool_log = await invoke_with_tools(
        prompt,
        tools=tool_list,
    )
    content = full_content or ""

    # Include tool results in the detail so they feed into downstream task contexts
    if tool_log and tool_log.strip():
        detail = f"{content}\n\n[工具执行记录]\n{tool_log}"
    else:
        detail = content

    return SubTaskOutput(
        id=task_id,
        name=task_name,
        detail=detail,
        summary=_extract_summary(detail),
        expert_mode=is_expert,
    )
