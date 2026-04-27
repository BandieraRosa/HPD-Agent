"""Execution node: difficulty assessment + LLM execution for a single sub-task.

This is a pure business-logic node — no graph coupling, no state mutation.
It handles:
  - Sub-task difficulty assessment (easy / hard)
  - LLM execution with prompt templating
  - Summary extraction
  - Tool-usage tracking (which files/resources were read)
  - Key-finding extraction (structured facts for downstream context)

The expert agent calls this to make the "execute" decision.
"""

import re
from typing import Annotated

from pydantic import BaseModel, Field

from src.core.enums import SubTaskDifficulty
from src.core.models import SubTaskAssessmentResult, SubTaskOutput
from src.llm import (
    invoke_with_tools,
    get_structured_llm,
    SUB_TASK_ASSESSMENT_PROMPT,
    SUB_TASK_PROMPT,
    KEY_FINDINGS_PROMPT,
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


class KeyFindingsResult(BaseModel):
    """Structured key findings extracted from a sub-task's output."""

    findings: list[str] = Field(
        default_factory=list,
        description=(
            "Key facts discovered, one per entry. Format: concise, machine-readable "
            "facts (e.g. 'port=8080', 'python=3.11', 'error=connection refused'). "
            "Only include genuinely new facts — do NOT repeat the original question. "
            "Maximum 10 entries."
        ),
    )


def _parse_tools_used(tool_log: str) -> list[str]:
    """Extract file/resource identifiers from a tool call log.

    Parses both direct read_file calls and terminal commands (cat, ls, find)
    so downstream tasks know exactly which files were already read.
    """
    paths: list[str] = []
    seen: set[str] = set()

    # Pattern 1: read_file(...) — extract the first single- or double-quoted path arg
    # Handles: read_file('/path')  and  read_file(path='/path')
    for match in re.finditer(r"read_file\s*\([^'\"]*['\"]([^'\"]+)['\"]", tool_log):
        path = match.group(1).strip()
        if path and path not in seen:
            seen.add(path)
            paths.append(path)

    # Pattern 2: terminal(cmd='...') — single-quote variant
    for match in re.finditer(r"terminal\s*\(\s*cmd\s*=\s*'([^']+)'", tool_log):
        cmd = match.group(1).strip()
        for path in _extract_paths_from_terminal_cmd(cmd):
            if path and path not in seen:
                seen.add(path)
                paths.append(path)

    # Pattern 3: terminal(cmd="...") — double-quote variant
    for match in re.finditer(r'terminal\s*\(\s*cmd\s*=\s*"([^"]+)"', tool_log):
        cmd = match.group(1).strip()
        for path in _extract_paths_from_terminal_cmd(cmd):
            if path and path not in seen:
                seen.add(path)
                paths.append(path)

    return paths


def _extract_paths_from_terminal_cmd(cmd: str) -> list[str]:
    """Extract file/directory paths from a shell command string."""
    paths: list[str] = []
    if not cmd:
        return paths

    # cat file1 file2 ...  —  extract non-flag positional args after "cat"
    if cmd.lstrip().startswith("cat "):
        rest = cmd.lstrip()[4:]
        # Split on whitespace but keep quoted strings intact (shlex-like split)
        tokens = re.split(r"\s+", rest)
        for token in tokens:
            if token and not token.startswith("-") and "|" not in token:
                paths.append(token)

    # ls path  or  ls -la path  —  capture the first non-flag argument
    ls_match = re.search(r"\bls\s+(?:-\w+\s+)*([^\s|;&]+)", cmd)
    if ls_match:
        target = ls_match.group(1).strip().rstrip("/")
        if target and target != ".":
            paths.append(target)

    # find path  or  find . -name "..."  —  first non-flag positional arg after "find"
    find_match = re.search(r"\bfind\s+(?:[^\s|;&]+(?:\s+-(?!name|type|exec))?)*\s+([^\s|;&]+)", cmd)
    if find_match:
        target = find_match.group(1).strip().rstrip("/")
        if target and target not in ("-name", "-type", "-exec", "-ok", "-delete", "-print"):
            paths.append(target)

    return paths


def _extract_key_findings_llm(detail: str) -> list[str]:
    """Use an LLM to extract structured key findings from detail text."""
    findings: list[str] = []
    try:
        classifier = get_structured_llm(KeyFindingsResult)
        result: KeyFindingsResult = classifier.invoke(KEY_FINDINGS_PROMPT.format(detail=detail[:3000]))
        findings = result.findings
    except Exception:
        pass
    return findings


async def execute(task_id: int, task_name: str, context: str) -> SubTaskOutput:
    """Assess difficulty and execute a single sub-task.

    Args:
        task_id:   Sub-task ID from the DAG.
        task_name: Human-readable sub-task name.
        context:   The original user query (shared background context).

    Returns:
        SubTaskOutput with detail, summary, tools_used, key_findings, and expert_mode flag.
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

    # Step 3: Parse tool usage from tool_log
    tools_used = _parse_tools_used(tool_log) if tool_log else []

    # Step 4: Build detail (includes tool log for traceability)
    if tool_log and tool_log.strip():
        detail = f"{content}\n\n[工具执行记录]\n{tool_log}"
    else:
        detail = content

    # Step 5: Extract key findings via structured LLM call
    key_findings = _extract_key_findings_llm(detail)

    return SubTaskOutput(
        id=task_id,
        name=task_name,
        detail=detail,
        summary=_extract_summary(detail),
        expert_mode=is_expert,
        tools_used=tools_used,
        key_findings=key_findings,
    )
