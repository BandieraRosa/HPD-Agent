from src.core.models import TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer

_MAX_PROMPT_CHARS = 12000
_MAX_FILE_RE_READS = 2


async def _re_read_key_files(done, max_files: int = _MAX_FILE_RE_READS) -> str:
    """Re-read the most relevant files from sub-task tool usage.

    Returns a formatted section of file contents, or empty string if nothing to read.
    """
    from src.tools import tool_list

    read_file_tool = next((t for t in tool_list if t.name == "read_file"), None)
    if read_file_tool is None:
        return ""

    # Collect unique file paths from all sub-tasks, preserving order
    seen: set[str] = set()
    paths: list[str] = []
    for o in done:
        for p in o.tools_used:
            if p not in seen:
                seen.add(p)
                paths.append(p)

    if not paths:
        return ""

    # Read the first N files
    sections: list[str] = []
    for path in paths[:max_files]:
        try:
            result = read_file_tool.invoke({"path": path})
            if result and not str(result).startswith("[Error]"):
                # Truncate each file to 2000 chars to avoid bloat
                content = str(result)[:2000]
                sections.append(f"--- {path} ---\n{content}")
        except Exception:
            continue

    if not sections:
        return ""

    return "【关键文件内容（重新读取）】\n" + "\n\n".join(sections)


async def synthesizer(state: AgentState) -> AgentState:
    """Build the synthesis prompt that combines all sub-task results.

    Uses compressed detail (reasoning + tool chain, no file contents) plus
    key_findings for a compact prompt. If the prompt is still large, re-reads
    key files on demand so the synthesis LLM has actual file content.
    """
    tracer = get_tracer()
    with tracer.span("synthesizer") as span_id:
        done = state.get("sub_task_outputs", [])

        sub_task_section = "\n".join(
            f"### 子任务 {o.id}: {o.name}\n"
            f"{'[专家模式] ' if o.expert_mode else ''}"
            f"{o.detail}\n"
            f"关键发现: {', '.join(o.key_findings) if o.key_findings else '无'}\n"
            f"结论: {o.summary}"
            for o in sorted(done, key=lambda x: x.id)
        )

        history = state.get("conversation_history")
        history_section = ""
        if history:
            history_text = history.to_summary()
            if history_text:
                history_section = f"【对话历史】\n{history_text}\n\n"

        synthesis_prompt = (
            f"{history_section}"
            f"【用户原始问题】\n{state['input']}\n\n"
            f"【所有子任务执行结果】\n{sub_task_section}\n\n"
            "请基于以上信息，用流畅自然的语言给出完整、专业的回答。"
            "直接输出回答内容，无需额外说明。"
        )

        # If prompt is large, re-read key files for richer context
        if len(synthesis_prompt) > _MAX_PROMPT_CHARS:
            file_section = await _re_read_key_files(done)
            if file_section:
                synthesis_prompt += f"\n\n{file_section}\n"

        output = TaskOutput(
            node="synthesizer",
            result={
                "sub_task_count": len(done),
                "expert_mode_count": sum(1 for o in done if o.expert_mode),
                "failed_count": sum(1 for o in done if o.summary.startswith("[失败]")),
                "prompt_chars": len(synthesis_prompt),
                "files_re_read": min(len({p for o in done for p in o.tools_used}), _MAX_FILE_RE_READS),
            },
        )

        return {
            "synthesis_prompt": synthesis_prompt,
            "outputs": [*state.get("outputs", []), output],
            "parent_span_id": span_id,
        }
