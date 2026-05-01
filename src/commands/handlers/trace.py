"""Handler for the /trace command — enable or disable tracing.

Sub-commands:
    /trace          → show current status
    /trace on       → full tracing (console + file)
    /trace half     → console only, no file saved
    /trace off      → disable tracing
"""

from src.agents import QueryAgent


# Module-level tracing toggle — read by main.py
# "on" = full (console + file), "half" = console only, "off" = disabled
_trace_mode: str = "on"


def is_trace_enabled() -> bool:
    return _trace_mode != "off"


def is_trace_save_enabled() -> bool:
    return _trace_mode == "on"


def run(raw: str, agent: QueryAgent) -> bool:
    global _trace_mode

    parts = raw.strip().split()
    if len(parts) == 1:
        labels = {"on": "启用（完整）", "half": "仅控制台", "off": "禁用"}
        print(f"链路追踪当前: {labels.get(_trace_mode, _trace_mode)}")
        return False

    sub = parts[1].lower()
    if sub in ("on", "1", "enable", "true"):
        _trace_mode = "on"
        print("链路追踪: 已启用（控制台 + 文件）")
    elif sub in ("half", "console"):
        _trace_mode = "half"
        print("链路追踪: 仅控制台输出，不保存文件")
    elif sub in ("off", "0", "disable", "false"):
        _trace_mode = "off"
        print("链路追踪: 已禁用")
    else:
        print("用法: /trace [on|half|off]")
        return False

    return False
