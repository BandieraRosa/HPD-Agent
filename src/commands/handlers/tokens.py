"""Handler for the /tokens command — shows context window token usage.

Token accounting follows the actual LLM context payload consumed in the synthesizer:
  1. conversation_history (messages)           ← always counted
  2. sub_task_outputs (accumulated DAG results) ← not stored in messages, but
     consumed by the synthesizer when building the final answer
  3. tool schemas (bind_tools overhead)         ← sent with every tool-bound call
  4. current synthesis prompt (if building)    ← the in-flight prompt

System prompts (assessment, planner, synthesis instructions) are NOT counted as
part of the rolling context window — they are injected per-call and not
persisted.  Only the accumulating conversation state is tracked here.
"""

import concurrent.futures
import json

from src.agents import QueryAgent
from src.memory.context import ConversationContext


# cl100k_base context window for the model (conservative: 128k for deepseek-v4)
MAX_TOKENS = 128_000


def _load_encoder():
    """Run in thread pool: download vocab + compile encoder."""
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


# tiktoken.get_encoding() blocks on first call (network + compilation).
# Warm it in a background thread pool so it never blocks the asyncio loop.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_encoder_future = _executor.submit(_load_encoder)


def _get_encoder():
    """Return the cached encoder, blocking only the background thread if not yet ready."""
    return _encoder_future.result()


def _count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    enc = _get_encoder()
    return len(enc.encode(text))


def _count_tool_schema_tokens() -> int:
    """Count tokens consumed by tool schemas from bind_tools().

    These are injected into every tool-bound API request but not tracked
    by the conversation context.
    """
    from src.tools import tool_list
    from langchain_core.utils.function_calling import convert_to_openai_function

    enc = _get_encoder()
    total = 0
    for t in tool_list:
        fn = convert_to_openai_function(t)
        total += len(enc.encode(json.dumps(fn, ensure_ascii=False)))
    return total


def _count_context_tokens(ctx: ConversationContext) -> int:
    """Count tokens for the rolling conversation_history (messages only).

    Each message is formatted as "{role}: {content}" per langchain convention.
    """
    enc = _get_encoder()
    total = 0
    for msg in ctx.messages:
        text = f"{msg.role.value}: {msg.content}"
        total += len(enc.encode(text))
    return total


def get_used_tokens(agent: QueryAgent) -> int:
    """Get the total tokens consumed by the current session's rolling context.

    This is the real token cost — it includes:
      - All messages in conversation_history (what's actually sent to the LLM)
      - All accumulated sub_task_outputs (consumed by the synthesizer but not
        stored in messages, so invisible to the message-only counter)
      - Tool schemas from bind_tools() (sent with every tool-bound call)
    """
    ctx = agent._get_context()
    total = _count_context_tokens(ctx)

    # Sub-task outputs are accumulated by QueryAgent and fed to the synthesizer,
    # but they are NOT stored in messages — they still consume the context window.
    for output in ctx.sub_task_outputs:
        total += _count_tokens(f"[子任务 {output['id']}: {output['name']}]\n{output['detail']}")

    # Tool schemas are injected into every tool-bound API request.
    total += _count_tool_schema_tokens()

    return total


def _format_bar(pct: float, width: int = 20) -> str:
    """Render a simple text progress bar."""
    filled = int(pct * width)
    return "[" + "=" * filled + " " * (width - filled) + "]"


def run(raw: str, agent: QueryAgent) -> bool:
    """Handle /tokens command — print context window token usage."""
    ctx = agent._get_context()

    # Count messages
    msg_tokens = _count_context_tokens(ctx)

    # Count sub-task outputs
    sub_task_tokens = 0
    for output in ctx.sub_task_outputs:
        sub_task_tokens += _count_tokens(
            f"[子任务 {output['id']}: {output['name']}]\n{output['detail']}"
        )

    # Count tool schemas (bind_tools overhead, sent with every tool-bound call)
    tool_schema_tokens = _count_tool_schema_tokens()

    # Count tool summaries (informational — already included in messages)
    tool_tokens = 0
    for msg in ctx.messages:
        if msg.tool_summary:
            tool_tokens += _count_tokens(msg.tool_summary)

    total = msg_tokens + sub_task_tokens + tool_schema_tokens

    pct = min(total / MAX_TOKENS, 1.0)
    bar = _format_bar(pct)

    print(f"=== Context Token Usage ===")
    print(f"  Conversation history:  {msg_tokens:>6} tokens  (messages)")
    print(f"  Sub-task results:     {sub_task_tokens:>6} tokens  (DAG outputs, not in messages)")
    print(f"  Tool schemas:          {tool_schema_tokens:>6} tokens  (bind_tools overhead)")
    print(f"  Total:                {total:>6} tokens")
    print(f"  Model window:        {MAX_TOKENS:>6} tokens")
    print(f"  Usage:              {bar} {pct * 100:.1f}%")
    if tool_tokens > 0:
        print(f"  (tool summaries:      {tool_tokens:>6} tokens — included above)")
    print()

    if total >= MAX_TOKENS:
        print("  [WARNING] 上下文窗口已满，请开启新对话或使用 /summary 总结上下文")

    return False
