import asyncio
import re

from src.core.models import TaskOutput
from src.core.state import AgentState
from src.core.observability import get_tracer, TokenTrackerCallback
from src.llm import get_llm
from src.tools import tool_list


def _build_direct_prompt(state: AgentState) -> str:
    """Build the LLM prompt for a direct-answer call."""
    history = state.get("conversation_history")
    history_text = history.to_summary() if history else ""

    if history_text:
        return (
            f"【对话历史】\n{history_text}\n\n"
            f"【当前问题】\n{state['input']}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
            "2. 请基于对话历史，直接、简洁地回答用户的问题。如果问题是对之前话题的追问，"
            "请结合历史上下文作答。如果不知道答案，请诚实说明。"
        )
    else:
        return (
            f"【用户问题】\n{state['input']}\n\n"
            "【重要规则】\n"
            "1. 如果问题需要获取实时信息或执行操作，你必须调用可用工具，不允许自己输出命令或猜测结果。\n"
            "2. 请直接、简洁地回答。如果不知道答案，请诚实说明。"
        )


async def direct_answer(state: AgentState) -> AgentState:
    """Handle simple tasks: ainvoke with tool-calling support.

    Uses ainvoke (not astream_events) so token usage is captured via the
    monkey-patch on ChatOpenAI._agenerate. The final text is streamed
    token-by-token to stdout for responsiveness.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    print()
    tracer = get_tracer()
    parent_id = state.get("parent_span_id") or None

    with tracer.span("direct_answer", parent_id=parent_id) as span_id:
        prompt = _build_direct_prompt(state)
        messages = [HumanMessage(content=prompt)]
        tool_results: list[str] = []

        for _ in range(20):
            llm = get_llm().bind_tools(tool_list)
            response = await llm.ainvoke(messages)

            content = getattr(response, "content", "") or ""
            tool_calls = getattr(response, "tool_calls", []) or []

            if tool_calls:
                for call in tool_calls:
                    call_id = call.get("id", "") or ""
                    name = call.get("name") or ""
                    args = call.get("args") or {}

                    tool = next((t for t in tool_list if t.name == name), None)

                    if tool is None:
                        result = f"[Error] Tool '{name}' not found"
                    elif name == "terminal":
                        cmd = args.get("cmd", "")
                        safe = (
                            "pwd", "ls", "cat", "echo", "date", "whoami",
                            "head", "tail", "less", "file ", "stat ",
                            "uname", "id", "hostname", "env", "printenv",
                        )
                        if cmd.strip().startswith(safe) or cmd.strip().startswith("cd "):
                            result = tool.invoke(args)
                            ok = not str(result).startswith("[Error]")
                            print(f"[DEBUG] Tool '{name}' {'succeeded' if ok else 'failed'}")
                        else:
                            confirm = await asyncio.to_thread(
                                lambda: input("    Confirm execution? (y/N): ").strip().lower()
                            )
                            if confirm != "y":
                                result = "[Cancelled] User declined to execute the terminal command"
                                print(f"[DEBUG] Tool '{name}' cancelled by user")
                            else:
                                result = tool.invoke(args)
                                ok = not str(result).startswith("[Error]") and not str(result).startswith("[Cancelled]")
                                print(f"[DEBUG] Tool '{name}' {'succeeded' if ok else 'failed'}")
                    else:
                        result = tool.invoke(args)
                        ok = not str(result).startswith("[Error]")
                        print(f"[DEBUG] Tool '{name}' {'succeeded' if ok else 'failed'}")

                    messages.append(AIMessage(content="", tool_calls=[call]))
                    messages.append(ToolMessage(name=name, content=str(result), tool_call_id=call_id))
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
                    tool_results.append(f"[Tool: {name}({args_str})]\n{result}")

                messages.append(HumanMessage(content=""))
            else:
                print(content, end="", flush=True)
                break

        # Record span tokens (all LLM calls were tracked by the monkey-patch)
        tin, tout, model = TokenTrackerCallback.snapshot()
        tracer.record_tokens(span_id, tokens_in=tin, tokens_out=tout, model=model)

    output = TaskOutput(
        node="direct_answer",
        result={"content": content, "tool_calls": "\n\n".join(tool_results)},
    )

    return {
        "outputs": [*state.get("outputs", []), output],
        "final_response": content,
        "parent_span_id": span_id,
    }
