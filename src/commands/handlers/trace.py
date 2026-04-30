"""Handler for the /trace command — enable or disable tracing."""

from src.agents import QueryAgent


# Module-level tracing toggle — read by main.py
_trace_enabled: bool = True


def is_trace_enabled() -> bool:
    return _trace_enabled


def run(raw: str, agent: QueryAgent) -> bool:
    global _trace_enabled

    parts = raw.strip().split()
    if len(parts) == 1:
        status = "启用" if _trace_enabled else "禁用"
        print(f"链路追踪当前: {status}")
        return False

    sub = parts[1].lower()
    if sub in ("on", "1", "enable", "true"):
        _trace_enabled = True
        print("链路追踪: 已启用")
    elif sub in ("off", "0", "disable", "false"):
        _trace_enabled = False
        print("链路追踪: 已禁用")
    else:
        print(f"用法: /trace [on|off]")
        return False

    return False
