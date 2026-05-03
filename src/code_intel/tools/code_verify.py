"""LangChain tool for agent-triggered code verification through CodeIntelKernel."""

from __future__ import annotations

from typing import Literal


from src.code_intel.core import Capability, Diagnostic, DiagnosticSeverity, ToolMeta, ToolResult

from ._helpers import (
    diagnostic_sequence,
    error_json,
    first_sources,
    kernel_error_json,
    language_for_path,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import ChecksSkipped, CodeVerifyData
from .runtime import get_code_intel_kernel

_CHECKS = {"lsp_diagnostics", "tests", "lint"}
_SEVERITIES = ("error", "warning", "info", "hint")


def _severity_delta(diagnostics: list[Diagnostic], baseline: bool) -> dict[str, int]:
    delta = {severity: 0 for severity in _SEVERITIES}
    if baseline:
        return delta
    for diagnostic in diagnostics:
        delta[diagnostic.severity.value] += 1
    return delta


def _skipped_checks(
    requested: list[str],
    paths: list[str],
    baseline: bool,
    ran_lsp: bool,
) -> list[ChecksSkipped]:
    skipped: list[ChecksSkipped] = []
    if "lsp_diagnostics" in requested and not ran_lsp:
        skipped.append(
            ChecksSkipped(
                check="lsp_diagnostics",
                reason="T6 只对显式 paths 运行诊断；changed/workspace 自动收集将在后续任务实现。",
            )
        )
    for check in requested:
        if check in {"tests", "lint"}:
            skipped.append(
                ChecksSkipped(
                    check=check,
                    reason="T6 暂不直接运行 tests/lint；请使用 terminal 或后续 verifier 工作流执行。",
                )
            )
    if baseline:
        skipped.append(
            ChecksSkipped(
                check="baseline",
                reason="T6 的 agent 验证不实现 workflow baseline bucket；当前诊断归入 unchanged_diagnostics。",
            )
        )
    if not paths:
        skipped.append(
            ChecksSkipped(
                check="path_collection",
                reason="T6 不扫描 changed/workspace 范围；请显式传入 paths。",
            )
        )
    return skipped


@code_intel_tool
async def code_verify(
    scope: Literal["changed", "file", "workspace"] = "changed",
    paths: list[str] | None = None,
    checks: list[Literal["lsp_diagnostics", "tests", "lint"]] | None = None,
    baseline: bool = True,
) -> str:
    """代码验证：agent 侧入口，当前通过 Kernel 对显式 paths 运行 lsp_diagnostics。

    Args:
        scope: changed、file 或 workspace；T6 只执行显式 paths。
        paths: 要诊断的工作区相对路径列表。
        checks: 请求的检查列表，默认 lsp_diagnostics。
        baseline: 是否按基线语义解释诊断；T6 不实现 workflow baseline bucket。
    """
    requested_checks = list(checks or ["lsp_diagnostics"])
    invalid_checks = [check for check in requested_checks if check not in _CHECKS]
    if invalid_checks:
        return error_json("invalid_input", "不支持的 checks。", f"请移除这些检查项：{', '.join(invalid_checks)}。")
    if scope not in {"changed", "file", "workspace"}:
        return error_json("invalid_input", "不支持的 scope。", "请使用 changed、file 或 workspace。")

    explicit_paths = list(paths or [])
    if any(not path.strip() for path in explicit_paths):
        return error_json("invalid_input", "paths 不能包含空路径。", "请提供工作区相对路径。")

    diagnostics: list[Diagnostic] = []
    results: list[ToolResult[object]] = []
    ran_lsp = False

    if "lsp_diagnostics" in requested_checks and explicit_paths:
        for path in explicit_paths:
            result = safe_cast_tool_result(
                await get_code_intel_kernel().call(
                    Capability.DIAGNOSTICS,
                    language_for_path(path),
                    path=path,
                )
            )
            results.append(result)
            if not result.ok:
                return kernel_error_json(result)
            try:
                diagnostics.extend(diagnostic_sequence(result.data))
            except (TypeError, ValueError) as error:
                return validation_error_json(error)
        ran_lsp = True

    has_error = any(diagnostic.severity == DiagnosticSeverity.ERROR for diagnostic in diagnostics)
    data = CodeVerifyData(
        ok=not has_error,
        new_diagnostics=[] if baseline else diagnostics,
        resolved_diagnostics=[],
        unchanged_diagnostics=diagnostics if baseline else [],
        severity_delta=_severity_delta(diagnostics, baseline),
        checks_run=["lsp_diagnostics"] if ran_lsp else [],
        checks_skipped=_skipped_checks(requested_checks, explicit_paths, baseline, ran_lsp),
        recommended_next_action="repair" if has_error else "proceed",
        call_source="agent",
    )
    meta = merge_meta(results[0].meta, sources_used=first_sources(*results)) if results else ToolMeta()
    return serialize_result(ToolResult[object](ok=True, data=data, meta=meta))
