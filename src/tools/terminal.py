"""Terminal tool: execute shell commands when no other tool is suitable."""

import subprocess
from langchain_core.tools import tool


@tool
def terminal(cmd: str) -> str:
    """Execute a shell command and return its output.

    Use this only when no other built-in tool is appropriate for the task.

    IMPORTANT — safety: read-only commands (pwd, ls, cat, echo, date, whoami, etc.)
    execute without user confirmation. Commands that modify the filesystem
    or system state require explicit user approval before execution.

    Args:
        cmd: The shell command to execute.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode != 0:
            return (
                f"[Command failed with exit code {result.returncode}]\n"
                f"[STDOUT]\n{output}\n"
                f"[STDERR]\n{stderr}"
            )
        return output if output else "[Command completed with no output]"
    except subprocess.TimeoutExpired:
        return "[Error] Command timed out after 60 seconds"
    except Exception as exc:
        return f"[Error] Failed to execute command: {exc}"
