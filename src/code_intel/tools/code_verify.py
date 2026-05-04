"""LangChain tool for agent-triggered code verification through CodeIntelKernel."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from src.code_intel.core import Capability, Diagnostic, DiagnosticSeverity, Symbol, ToolMeta, ToolResult
from src.code_intel.verifier import (
    PatchSummary,
    build_baseline_key,
    compute_delta,
    content_hash_for_paths,
    get_cached_baseline,
    refresh_agent_baseline,
    stable_json_hash,
)
from ._helpers import (
    diagnostic_sequence,
    error_json,
    first_sources,
    kernel_error_json,
    language_for_path,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    symbol_sequence,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import ChecksSkipped, CodeVerifyData
from .runtime import get_code_intel_kernel

_CHECKS = {"lsp_diagnostics", "tests", "lint"}
_SEVERITIES = ("error", "warning", "info", "hint")
_PROVIDER_UNAVAILABLE = "provider_unavailable"


def _severity_counts(diagnostics: Iterable[Diagnostic]) -> dict[str, int]:
    delta = {severity: 0 for severity in _SEVERITIES}
    for diagnostic in diagnostics:
        delta[diagnostic.severity.value] += 1
    return delta


def _skipped_checks(
    requested: list[str],
    paths: list[str],
    ran_lsp: bool,
) -> list[ChecksSkipped]:
    skipped: list[ChecksSkipped] = []
    if "lsp_diagnostics" in requested and not ran_lsp:
        skipped.append(
            ChecksSkipped(
                check="lsp_diagnostics",
                reason="当前仅对显式 paths 运行诊断；changed/workspace 自动收集将在后续任务实现。",
            )
        )
    for check in requested:
        if check in {"tests", "lint"}:
            skipped.append(
                ChecksSkipped(
                    check=check,
                    reason="当前 verifier 不直接运行 tests/lint；请使用 terminal 或后续 workflow 集成执行。",
                )
            )
    if not paths:
        skipped.append(
            ChecksSkipped(
                check="path_collection",
                reason="当前不扫描 changed/workspace 范围；请显式传入 paths。",
            )
        )
    return skipped


def _provider_identity() -> tuple[str, str]:
    records: list[dict[str, object]] = []
    for provider in get_code_intel_kernel().providers:
        provider_name = str(getattr(provider, "name", provider.__class__.__name__))
        records.append(
            {
                "name": provider_name,
                "class": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
                "capabilities": _stable_strings(getattr(provider, "capabilities", ())),
                "languages": _stable_strings(getattr(provider, "languages", ())),
            }
        )
    provider_id = "+".join(str(record["name"]) for record in records) if records else "none"
    return provider_id, stable_json_hash(records)


def _stable_strings(values: object) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)):
        return [str(values)]
    if not isinstance(values, Iterable):
        return []
    return sorted(str(getattr(item, "value", item)) for item in values)


async def _symbols_for_path(path: str) -> list[Symbol]:
    language = language_for_path(path)
    kernel = get_code_intel_kernel()
    outline = safe_cast_tool_result(await kernel.call(Capability.OUTLINE, language, path=path))
    if outline.ok:
        try:
            return symbol_sequence(outline.data)
        except (TypeError, ValueError):
            return []
    document_symbols = safe_cast_tool_result(await kernel.call(Capability.DOCUMENT_SYMBOLS, language, path=path))
    if not document_symbols.ok:
        return []
    try:
        return symbol_sequence(document_symbols.data)
    except (TypeError, ValueError):
        return []


async def _symbols_by_path(paths: list[str]) -> dict[str, list[Symbol]]:
    symbols: dict[str, list[Symbol]] = {}
    for path in paths:
        symbols[path] = await _symbols_for_path(path)
    return symbols


def _status_for_diagnostics(diagnostics: list[Diagnostic]) -> Literal["success", "partial", "blocked"]:
    if any(diagnostic.severity == DiagnosticSeverity.ERROR for diagnostic in diagnostics):
        return "blocked"
    if diagnostics:
        return "partial"
    return "success"


def _action_for_status(status: str) -> Literal["proceed", "repair", "abort"]:
    return "repair" if status == "blocked" else "proceed"


def _provider_partial_data(error_result: ToolResult[object], requested_checks: list[str], paths: list[str]) -> CodeVerifyData:
    skipped = _skipped_checks(requested_checks, paths, ran_lsp=False)
    skipped.append(
        ChecksSkipped(
            check="lsp_diagnostics",
            reason="诊断提供方暂时不可用，verification_status=partial；这不是成功验证。",
        )
    )
    return CodeVerifyData(
        ok=False,
        new_diagnostics=[],
        resolved_diagnostics=[],
        unchanged_diagnostics=[],
        severity_delta={severity: 0 for severity in _SEVERITIES},
        checks_run=[],
        checks_skipped=skipped,
        recommended_next_action="abort",
        call_source="agent",
        verification_status="partial",
        provider_error=error_result.error,
    )


@code_intel_tool
async def code_verify(
    scope: Literal["changed", "file", "workspace"] = "changed",
    paths: list[str] | None = None,
    checks: list[Literal["lsp_diagnostics", "tests", "lint"]] | None = None,
    baseline: bool = True,
) -> str:
    """代码验证：agent 侧入口，通过 Kernel 对显式 paths 运行 lsp_diagnostics。

    Args:
        scope: changed、file 或 workspace；当前只执行显式 paths。
        paths: 要诊断的工作区相对路径列表。
        checks: 请求的检查列表，默认 lsp_diagnostics。
        baseline: True 时使用 T13 agent baseline bucket；首次调用会捕获当前诊断作为基线。
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
                if result.error is not None and result.error.code == _PROVIDER_UNAVAILABLE:
                    meta = merge_meta(result.meta, sources_used=first_sources(*results))
                    data = _provider_partial_data(result, requested_checks, explicit_paths)
                    return serialize_result(ToolResult[object](ok=True, data=data, meta=meta))
                return kernel_error_json(result)
            try:
                diagnostics.extend(diagnostic_sequence(result.data))
            except (TypeError, ValueError) as error:
                return validation_error_json(error)
        ran_lsp = True

    checks_skipped = _skipped_checks(requested_checks, explicit_paths, ran_lsp)
    meta = merge_meta(results[0].meta, sources_used=first_sources(*results)) if results else ToolMeta()

    if baseline:
        symbols_by_path = await _symbols_by_path(explicit_paths)
        provider_id, provider_config_hash = _provider_identity()
        # Compute content hash off the event loop to avoid blocking file IO
        content_hash = await asyncio.to_thread(content_hash_for_paths, Path.cwd(), explicit_paths)
        key = build_baseline_key(
            workspace_root=Path.cwd(),
            relevant_paths=explicit_paths,
            provider_id=provider_id,
            provider_config_hash=provider_config_hash,
            call_source="agent",
            content_hash=content_hash,
        )
        cached = get_cached_baseline(key)
        baseline_refreshed = False
        if cached is None:
            cached = refresh_agent_baseline(key=key, diagnostics=diagnostics, symbols_by_path=symbols_by_path)
            baseline_refreshed = True
        delta = compute_delta(cached, diagnostics, PatchSummary(), after_symbols_by_path=symbols_by_path)
        status: Literal["success", "partial", "blocked"] = "blocked" if delta.has_new_errors else "partial" if delta.partial else "success"
        data = CodeVerifyData(
            ok=not delta.has_new_errors,
            new_diagnostics=delta.new_diagnostics,
            resolved_diagnostics=delta.resolved_diagnostics,
            unchanged_diagnostics=delta.unchanged_diagnostics,
            severity_delta=delta.severity_delta,
            checks_run=["lsp_diagnostics"] if ran_lsp else [],
            checks_skipped=checks_skipped,
            recommended_next_action=_action_for_status(status),
            call_source="agent",
            verification_status=status,
            baseline_key=key.cache_key(),
            baseline_refreshed=baseline_refreshed,
        )
        return serialize_result(ToolResult[object](ok=True, data=data, meta=meta))

    status = _status_for_diagnostics(diagnostics)
    data = CodeVerifyData(
        ok=status != "blocked",
        new_diagnostics=diagnostics,
        resolved_diagnostics=[],
        unchanged_diagnostics=[],
        severity_delta=_severity_counts(diagnostics),
        checks_run=["lsp_diagnostics"] if ran_lsp else [],
        checks_skipped=checks_skipped,
        recommended_next_action=_action_for_status(status),
        call_source="agent",
        verification_status=status,
    )
    return serialize_result(ToolResult[object](ok=True, data=data, meta=meta))
