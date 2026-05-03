"""Runtime kernel injection for agent-facing code-intelligence tools."""

from __future__ import annotations

from src.code_intel import CodeIntelKernel

_code_intel_kernel: CodeIntelKernel = CodeIntelKernel()


def get_code_intel_kernel() -> CodeIntelKernel:
    """Return the explicitly configured CodeIntelKernel.

    默认内核不注册任何 provider；测试或上层运行时必须显式注入 provider。
    """
    return _code_intel_kernel


def set_code_intel_kernel(kernel: CodeIntelKernel | None) -> None:
    """Set the kernel used by code_* tools, or reset to an empty provider-free kernel."""
    global _code_intel_kernel
    _code_intel_kernel = kernel if kernel is not None else CodeIntelKernel()


__all__ = ["get_code_intel_kernel", "set_code_intel_kernel"]
