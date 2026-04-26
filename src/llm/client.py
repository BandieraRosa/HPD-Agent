"""LLM factory — all ChatOpenAI instances are created here.

Active model is read from the profile store singleton so that /model
switches take effect for every LLM call without code changes.
"""

import os

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def _resolve_api_key(profile_key: str) -> str:
    """Prefer the per-profile key; fall back to the legacy env var."""
    if profile_key:
        return profile_key
    for env_key in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if env_key in os.environ:
            return os.environ[env_key]
    raise RuntimeError(
        "No API key configured. "
        "Set DEEPSEEK_API_KEY (or DASHSCOPE_API_KEY), "
        "or configure an api_key in your model profile via /model create."
    )


def _active_profile() -> object:
    """Import lazily to avoid circular imports at module startup."""
    from src.models import get_store
    return get_store().active_profile()


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI client from the active profile.

    Any explicitly-passed parameter overrides the profile (for backwards
    compatibility with callers that pass model=/temperature=/base_url=).
    """
    profile = _active_profile()

    actual_model = model if model is not None else (getattr(profile, "model", "deepseek-v4-flash") if profile else "deepseek-v4-flash")
    actual_temp  = temperature if temperature is not None else (getattr(profile, "temperature", 0.0) if profile else 0.0)
    actual_base  = base_url if base_url is not None else (getattr(profile, "base_url", "https://api.deepseek.com") if profile else "https://api.deepseek.com")
    actual_key   = _resolve_api_key(getattr(profile, "api_key", "") if profile else "")
    actual_eb    = getattr(profile, "extra_body", {}) if profile else {}

    return ChatOpenAI(
        model=actual_model,
        temperature=actual_temp,
        api_key=actual_key,
        base_url=actual_base,
        stream=stream,
        extra_body=actual_eb,
    )


def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Return an LLM bound to output a specific Pydantic schema."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    return llm.with_structured_output(schema, method="function_calling")


def get_llm_with_tools(
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    tools: list[BaseTool] | None = None,
    stream: bool = False,
) -> ChatOpenAI:
    """Build a ChatOpenAI client and bind tools to it."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    if tools:
        llm = llm.bind_tools(tools)
    return llm


async def invoke_with_tools(
    prompt: str,
    tools: list[BaseTool] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    stream: bool = False,
) -> tuple[str, str]:
    """Invoke the LLM with tool-calling support.

    Args:
        prompt: The user prompt to send.
        tools: List of tools available to the LLM.
        model / temperature / base_url: Override active profile settings.
        stream: If True, yields chunks via print; returns full content.

    Returns:
        (final_text, tool_calls_log) — the LLM's final text response and a
        concatenated string of all tool results for context.
    """
    llm = get_llm_with_tools(
        model=model, temperature=temperature, base_url=base_url,
        tools=tools, stream=stream,
    )

    messages = [HumanMessage(content=prompt)]
    tool_results: list[str] = []
    full_content = ""

    max_turns = 10
    for _ in range(max_turns):
        response = await llm.ainvoke(messages)
        content = getattr(response, "content", "") or ""
        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            full_content = content
            break

        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            tool = None
            if tools:
                for t in tools:
                    if t.name == name:
                        tool = t
                        break
            if tool is None:
                result = f"[Error] Tool '{name}' not found"
            else:
                result = tool.invoke(args)
            messages.append(AIMessage(content="", tool_calls=[call]))
            messages.append(ToolMessage(name=name, content=str(result), tool_call_id=call.get("id", "")))
            tool_results.append(f"[Tool: {name}]\n{result}")

    return full_content, "\n\n".join(tool_results)
