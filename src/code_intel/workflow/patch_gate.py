"""Workflow patch gate for apply_patch side effects and changed-file verification."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, Field

from src.code_intel.core import Capability, Diagnostic, ToolError, ToolResult
from src.code_intel.core.languages import language_for_path_or_default
from src.code_intel.core.models import validate_workspace_relative_path
from src.code_intel.tools._helpers import stable_strings
from src.code_intel.tools.models import ChecksSkipped, CodeVerifyData
from src.code_intel.verifier import (
    BaselineSnapshot,
    DiagnosticsDelta,
    PatchSummary,
    RepairAction,
    RepairDecision,
    RepairPolicy,
    VerificationStatus,
    build_baseline_key,
    compute_delta,
    content_hash_for_paths,
    get_cached_baseline,
    stable_json_hash,
)
from src.core.models import TaskOutput
from src.core.state import AgentState

_CHANGED_SCOPE: Literal["changed"] = "changed"
_PATCH_SECTION_RE = re.compile(
    r"^\*\*\* (?:Add|Update|Delete|Replace) File: (?P<path>.+)$", re.MULTILINE
)
_TOOL_ENTRY_RE = re.compile(
    r"\[Tool:\s*apply_patch\((?P<args>.*?)\)\]\n(?P<result>.*?)(?=\n\n\[Tool:|\Z)",
    re.DOTALL,
)
_APPLY_RESULT_PATH_RE = re.compile(
    r"^- (?:Add|Update|Replace|Delete): (?P<path>.+?) \(\+\d+/-\d+, final \d+ bytes\)$",
    re.MULTILINE,
)


class KernelLike(Protocol):
    """Subset of CodeIntelKernel used by workflow verification."""

    @property
    def providers(self) -> tuple[object, ...]: ...

    async def call(
        self, capability: Capability | str, language: str, **kwargs: object
    ) -> ToolResult[object]: ...


class PatchGateConfig(BaseModel):
    """Runtime switches for workflow-side patch handling."""

    workspace_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Workspace root for changed files.",
    )
    verify_changed: bool = Field(
        default=False,
        description="Run changed-file diagnostics through the workflow verifier.",
    )
    invalidate_syntax: bool = Field(
        default=False,
        description="Call a configured syntax invalidator for changed paths.",
    )
    invalidate_index: bool = Field(
        default=False,
        description="Call a configured index invalidator for changed paths.",
    )
    notify_lsp_did_change: bool = Field(
        default=False, description="Notify a configured, already-managed LSP adapter."
    )
    max_repair_rounds: int = Field(
        default=2, ge=0, description="Maximum workflow repair rounds before retreat."
    )
    block_on_warnings: bool = Field(
        default=False, description="Treat new warnings as blocking diagnostics."
    )


class PatchGateResult(BaseModel):
    """Structured workflow result after observing one apply_patch tool log."""

    changed_files: list[str] = Field(
        default_factory=list,
        description="Workspace-relative files changed by apply_patch.",
    )
    invalidated_paths: list[str] = Field(
        default_factory=list, description="Paths passed to configured invalidators."
    )
    notified_paths: list[str] = Field(
        default_factory=list, description="Paths sent to configured didChange notifier."
    )
    verification: CodeVerifyData | None = Field(
        default=None, description="Changed-file verifier result, when enabled."
    )
    repair_decision: RepairDecision | None = Field(
        default=None, description="Repair policy decision for the verifier delta."
    )
    verify_scope: Literal["changed"] = Field(
        default=_CHANGED_SCOPE,
        description="PatchGate never auto-runs workspace verify.",
    )
    workspace_verify_started: bool = Field(
        default=False,
        description="Always false; guards against workspace-wide verification.",
    )


class PatchGate:
    """Workflow-owned patch gate for changed files, invalidation, LSP sync, and repair prompts."""

    def __init__(
        self,
        config: PatchGateConfig | None = None,
        *,
        kernel: KernelLike | None = None,
        syntax_invalidator: object | None = None,
        index_invalidator: object | None = None,
        lsp_notifier: object | None = None,
    ) -> None:
        self.config: PatchGateConfig = config or PatchGateConfig()
        self._kernel: KernelLike | None = kernel
        self._syntax_invalidator: object | None = syntax_invalidator
        self._index_invalidator: object | None = index_invalidator
        self._lsp_notifier: object | None = lsp_notifier
        self._last_delta: DiagnosticsDelta | None = None
        self._last_changed_files: list[str] = []
        self._last_verification: CodeVerifyData | None = None
        self._repair_round: int = 0

    async def handle_tool_log(self, tool_log: str) -> PatchGateResult | None:
        """Observe an invoke_with_tools log and gate successful non-dry-run apply_patch calls."""
        changed_files = changed_files_from_tool_log(tool_log)
        if not changed_files:
            return None
        return await self.after_apply_patch(changed_files)

    async def after_apply_patch(
        self,
        changed_files: Iterable[str],
        patch_summary: PatchSummary | None = None,
    ) -> PatchGateResult:
        """Run configured changed-file side effects after apply_patch has written files."""
        paths = _unique_valid_paths(changed_files)
        invalidated: list[str] = []
        if self.config.invalidate_syntax:
            invalidated.extend(await _invalidate_paths(self._syntax_invalidator, paths))
        if self.config.invalidate_index:
            invalidated.extend(await _invalidate_paths(self._index_invalidator, paths))

        notified: list[str] = []
        if self.config.notify_lsp_did_change:
            notified = await self._notify_lsp(paths)

        verification: CodeVerifyData | None = None
        decision: RepairDecision | None = None
        if self.config.verify_changed:
            verification, decision = await self._verify_changed(
                paths, patch_summary or PatchSummary()
            )

        return PatchGateResult(
            changed_files=paths,
            invalidated_paths=_unique_valid_paths(invalidated),
            notified_paths=notified,
            verification=verification,
            repair_decision=decision,
            verify_scope=_CHANGED_SCOPE,
            workspace_verify_started=False,
        )

    async def run_repair_round(
        self,
        state: Mapping[str, object] | None = None,
        *,
        parent_span_id: str | None = None,
        max_rounds: int | None = None,
    ) -> AgentState | None:
        """Return a minimal reviewer-state update when the latest changed-file delta needs repair."""
        if (
            self._last_delta is None
            or not self._last_changed_files
            or self._last_verification is None
        ):
            return None

        policy = RepairPolicy(
            max_rounds=(
                self.config.max_repair_rounds if max_rounds is None else max_rounds
            ),
            block_on_warnings=self.config.block_on_warnings,
        )
        decision = policy.decide(self._last_delta, round=self._repair_round)
        if decision.action != RepairAction.REPAIR:
            if decision.action == RepairAction.RETREAT:
                return self._state_update(state, decision, "proceed", parent_span_id)
            return None

        update = self._state_update(state, decision, "re-execute", parent_span_id)
        self._repair_round += 1
        return update

    async def _verify_changed(
        self,
        changed_files: list[str],
        patch_summary: PatchSummary,
    ) -> tuple[CodeVerifyData, RepairDecision]:
        if not changed_files:
            delta = compute_delta([], [], patch_summary)
            decision = RepairPolicy(max_rounds=self.config.max_repair_rounds).decide(
                delta, round=self._repair_round
            )
            data = _verify_data(
                ok=True,
                diagnostics=[],
                delta=delta,
                decision=decision,
                checks_run=[],
                checks_skipped=_workflow_skips(changed_files),
            )
            self._remember(delta, changed_files, data)
            return data, decision

        kernel = self._kernel_or_runtime()
        diagnostics: list[Diagnostic] = []
        for path in changed_files:
            result = await kernel.call(
                Capability.DIAGNOSTICS, _language_for_path(path), path=path
            )
            if not result.ok:
                error = result.error or ToolError(
                    code="provider_unavailable",
                    message="诊断提供方暂时不可用。",
                    hint="PatchGate 已返回 partial；请稍后重试 changed 范围验证。",
                )
                return self._provider_partial(changed_files, error)
            try:
                diagnostics.extend(_diagnostic_sequence(result.data))
            except (TypeError, ValueError):
                return self._provider_partial(
                    changed_files,
                    ToolError(
                        code="invalid_diagnostics",
                        message="诊断提供方返回了无法解析的数据。",
                        hint="请检查诊断 provider 输出格式。",
                    ),
                )

        provider_id, provider_config_hash = provider_identity_for_kernel(kernel)
        content_hash = await asyncio.to_thread(
            content_hash_for_paths, self.config.workspace_root, changed_files
        )
        key = build_baseline_key(
            workspace_root=self.config.workspace_root,
            relevant_paths=changed_files,
            provider_id=provider_id,
            provider_config_hash=provider_config_hash,
            call_source="workflow",
            content_hash=content_hash,
        )
        cached = get_cached_baseline(key)
        baseline_refreshed = False
        baseline: BaselineSnapshot | list[Diagnostic] = (
            cached if cached is not None else []
        )

        delta = compute_delta(baseline, diagnostics, patch_summary)
        decision = RepairPolicy(
            max_rounds=self.config.max_repair_rounds,
            block_on_warnings=self.config.block_on_warnings,
        ).decide(delta, round=self._repair_round)
        data = _verify_data(
            ok=decision.action == RepairAction.PROCEED
            and decision.status != VerificationStatus.BLOCKED,
            diagnostics=diagnostics,
            delta=delta,
            decision=decision,
            checks_run=["lsp_diagnostics"],
            checks_skipped=_workflow_skips(changed_files),
            baseline_key=key.cache_key(),
            baseline_refreshed=baseline_refreshed,
        )
        self._remember(delta, changed_files, data)
        return data, decision

    def _provider_partial(
        self, changed_files: list[str], error: ToolError
    ) -> tuple[CodeVerifyData, RepairDecision]:
        delta = DiagnosticsDelta.provider_partial(error)
        decision = RepairPolicy(max_rounds=self.config.max_repair_rounds).decide(
            delta, round=self._repair_round
        )
        data = _verify_data(
            ok=False,
            diagnostics=[],
            delta=delta,
            decision=decision,
            checks_run=[],
            checks_skipped=_workflow_skips(changed_files, provider_error=error),
            provider_error=error,
        )
        self._remember(delta, changed_files, data)
        return data, decision

    async def _notify_lsp(self, paths: list[str]) -> list[str]:
        if self._lsp_notifier is None:
            return []
        notified: list[str] = []
        for path in paths:
            content = await asyncio.to_thread(
                _read_workspace_text, self.config.workspace_root, path
            )
            if content is None:
                continue
            if await _call_first(
                self._lsp_notifier, ("notify_did_change", "did_change"), path, content
            ):
                notified.append(path)
        return notified

    def _kernel_or_runtime(self) -> KernelLike:
        if self._kernel is not None:
            return self._kernel
        from src.code_intel.tools.runtime import get_code_intel_kernel

        return get_code_intel_kernel()

    def _remember(
        self,
        delta: DiagnosticsDelta,
        changed_files: list[str],
        verification: CodeVerifyData,
    ) -> None:
        self._last_delta = delta
        self._last_changed_files = list(changed_files)
        self._last_verification = verification

    def _state_update(
        self,
        state: Mapping[str, object] | None,
        decision: RepairDecision,
        routed_decision: Literal["proceed", "re-execute"],
        parent_span_id: str | None,
    ) -> AgentState:
        current_round = _int_state_value(state, "review_round")
        outputs = _sequence_state_value(state, "outputs")
        re_execute_ids = _repair_task_ids(state)
        feedback = build_repair_prompt(
            self._last_changed_files, self._last_verification, decision
        )
        return cast(
            AgentState,
            cast(
                object,
                {
                    "review_decision": routed_decision,
                    "re_execute_task_ids": (
                        re_execute_ids if routed_decision == "re-execute" else []
                    ),
                    "review_feedback": feedback,
                    "review_round": current_round + 1,
                    "outputs": [
                        *outputs,
                        TaskOutput(
                            node="reviewer",
                            result={
                                "decision": (
                                    "repair"
                                    if routed_decision == "re-execute"
                                    else "proceed"
                                ),
                                "routed_as": routed_decision,
                                "changed_files": list(self._last_changed_files),
                                "repair_round": decision.repair_round,
                                "max_rounds": decision.max_rounds,
                                "verification_status": decision.status.value,
                                "reason": decision.reason,
                            },
                        ),
                    ],
                    "parent_span_id": parent_span_id,
                },
            ),
        )


def changed_files_from_patch_text(patch_text: str) -> list[str]:
    """Extract workspace-relative paths from a v2 apply_patch document."""
    return _unique_valid_paths(
        match.group("path").strip() for match in _PATCH_SECTION_RE.finditer(patch_text)
    )


def changed_files_from_tool_log(tool_log: str) -> list[str]:
    """Extract files changed by successful non-dry-run apply_patch tool entries."""
    changed: list[str] = []
    for match in _TOOL_ENTRY_RE.finditer(tool_log):
        args = match.group("args")
        result = match.group("result")
        if "dry_run=True" in args or "[OK] Applied patch" not in result:
            continue
        changed.extend(
            path_match.group("path").strip()
            for path_match in _APPLY_RESULT_PATH_RE.finditer(result)
        )
    return _unique_valid_paths(changed)


def build_repair_prompt(
    changed_files: Sequence[str],
    verification: CodeVerifyData | None,
    decision: RepairDecision,
) -> str:
    """Build the reviewer feedback text for a workflow repair round."""
    diagnostics = verification.new_diagnostics if verification is not None else []
    diagnostic_lines = _diagnostic_lines(diagnostics)
    files = "、".join(changed_files) if changed_files else "（未捕获 changed files）"
    return (
        "【Workflow PatchGate 修复指令】\n"
        f"changed files：{files}\n"
        f"状态：{decision.status.value}；原因：{decision.reason}\n"
        f"新错误：{decision.new_errors}；新警告：{decision.new_warnings}\n"
        f"诊断摘要：\n{diagnostic_lines}\n\n"
        "请只围绕上述 changed files 修复本轮 apply_patch 引入的问题。编辑前先使用 "
        "code_search / code_outline / code_context / code_semantic 获取定位上下文；"
        '修改后必须调用 code_verify(scope="changed") 验证 changed 范围。'
    )


def provider_identity_for_kernel(kernel: object) -> tuple[str, str]:
    """Return the stable workflow verifier provider identity used by baseline keys."""
    providers_obj = getattr(kernel, "providers", ())
    if isinstance(providers_obj, (str, bytes, bytearray)) or not isinstance(
        providers_obj, Iterable
    ):
        providers: Iterable[object] = ()
    else:
        providers = cast(Iterable[object], providers_obj)
    records: list[dict[str, object]] = []
    for provider in providers:
        provider_name = str(getattr(provider, "name", provider.__class__.__name__))
        records.append(
            {
                "name": provider_name,
                "class": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
                "capabilities": stable_strings(getattr(provider, "capabilities", ())),
                "languages": stable_strings(getattr(provider, "languages", ())),
            }
        )
    provider_id = (
        "+".join(str(record["name"]) for record in records) if records else "none"
    )
    return provider_id, stable_json_hash(records)


_default_gate = PatchGate()


def configure_default_gate(gate: PatchGate | None = None) -> PatchGate:
    """Replace the process-local default gate and return the active instance."""
    global _default_gate
    _default_gate = gate or PatchGate()
    return _default_gate


async def handle_tool_log(tool_log: str) -> PatchGateResult | None:
    """Module-level entry used by existing execution loops."""
    return await _default_gate.handle_tool_log(tool_log)


async def run_repair_round(
    state: Mapping[str, object] | None = None,
    *,
    parent_span_id: str | None = None,
    max_rounds: int | None = None,
) -> AgentState | None:
    """Module-level reviewer repair hook."""
    return await _default_gate.run_repair_round(
        state, parent_span_id=parent_span_id, max_rounds=max_rounds
    )


async def _invalidate_paths(target: object | None, paths: list[str]) -> list[str]:
    if target is None or not paths:
        return []
    if await _call_first(
        target, ("invalidate_paths", "mark_stale", "invalidate"), list(paths)
    ):
        return list(paths)
    method = getattr(target, "delete_file", None)
    if callable(method):
        invalidated: list[str] = []
        for path in paths:
            try:
                result = method(path)
                if inspect.isawaitable(result):
                    _ = await cast(Awaitable[object], result)
                invalidated.append(path)
            except Exception:
                continue
        return invalidated
    return []


async def _call_first(
    target: object, method_names: Sequence[str], *args: object
) -> bool:
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(*args)
            if inspect.isawaitable(result):
                _ = await cast(Awaitable[object], result)
            return True
        except Exception:
            return False
    return False


def _read_workspace_text(workspace_root: Path, path: str) -> str | None:
    root = workspace_root.expanduser().resolve(strict=False)
    absolute_path = (root / path).resolve(strict=False)
    try:
        _ = absolute_path.relative_to(root)
    except ValueError:
        return None
    if not absolute_path.is_file():
        return None
    try:
        return absolute_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _verify_data(
    *,
    ok: bool,
    diagnostics: list[Diagnostic],
    delta: DiagnosticsDelta,
    decision: RepairDecision,
    checks_run: list[str],
    checks_skipped: list[ChecksSkipped],
    baseline_key: str | None = None,
    baseline_refreshed: bool = False,
    provider_error: ToolError | None = None,
) -> CodeVerifyData:
    recommended_next_action: Literal["proceed", "repair", "abort"]
    if provider_error is not None or decision.action == RepairAction.RETREAT:
        recommended_next_action = "abort"
    elif decision.action == RepairAction.REPAIR:
        recommended_next_action = "repair"
    else:
        recommended_next_action = "proceed"
    return CodeVerifyData(
        ok=ok,
        new_diagnostics=(
            delta.new_diagnostics
            if delta.new
            else diagnostics if decision.status == VerificationStatus.BLOCKED else []
        ),
        resolved_diagnostics=delta.resolved_diagnostics,
        unchanged_diagnostics=delta.unchanged_diagnostics,
        severity_delta=delta.severity_delta,
        checks_run=checks_run,
        checks_skipped=checks_skipped,
        recommended_next_action=recommended_next_action,
        call_source="workflow",
        verification_status=decision.status.value,
        baseline_key=baseline_key,
        baseline_refreshed=baseline_refreshed,
        provider_error=provider_error,
    )


def _workflow_skips(
    changed_files: Sequence[str], provider_error: ToolError | None = None
) -> list[ChecksSkipped]:
    skipped = [
        ChecksSkipped(
            check="workspace_verify",
            reason="PatchGate 只处理 changed files，不会自动启动 workspace-wide verify。",
        ),
        ChecksSkipped(
            check="tests",
            reason="PatchGate 不自动运行测试；需要由执行循环或用户命令显式触发。",
        ),
        ChecksSkipped(
            check="lint",
            reason="PatchGate 不自动运行 lint；需要由执行循环或用户命令显式触发。",
        ),
    ]
    if not changed_files:
        skipped.append(
            ChecksSkipped(
                check="path_collection",
                reason="未从 apply_patch 日志中捕获 changed files。",
            )
        )
    if provider_error is not None:
        skipped.append(
            ChecksSkipped(
                check="lsp_diagnostics",
                reason="诊断提供方暂时不可用，verification_status=partial；这不是成功验证。",
            )
        )
    return skipped


def _diagnostic_sequence(data: object) -> list[Diagnostic]:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        raise TypeError("kernel diagnostics data is not a sequence")
    return [
        item if isinstance(item, Diagnostic) else Diagnostic.model_validate(item)
        for item in data
    ]


def _diagnostic_lines(diagnostics: Sequence[Diagnostic]) -> str:
    if not diagnostics:
        return "- （无新诊断详情）"
    lines: list[str] = []
    for diagnostic in diagnostics[:8]:
        message = f"{diagnostic.code or diagnostic.source}: {diagnostic.message}"
        lines.append(
            f"- {diagnostic.path}:{diagnostic.range.start_line + 1} [{diagnostic.severity.value}] {message}"
        )
    if len(diagnostics) > 8:
        lines.append(f"- 另有 {len(diagnostics) - 8} 条诊断未展开。")
    return "\n".join(lines)


def _repair_task_ids(state: Mapping[str, object] | None) -> list[int]:
    outputs = _sequence_state_value(state, "sub_task_outputs")
    ids: list[int] = []
    for item in outputs:
        value = getattr(item, "id", None)
        if isinstance(value, int):
            ids.append(value)
        elif isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            raw = mapping.get("id")
            if isinstance(raw, int):
                ids.append(raw)
    return ids[-1:] if ids else []


def _sequence_state_value(state: Mapping[str, object] | None, key: str) -> list[object]:
    if state is None:
        return []
    value = state.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _int_state_value(state: Mapping[str, object] | None, key: str) -> int:
    if state is None:
        return 0
    value = state.get(key)
    return value if isinstance(value, int) else 0


def _unique_valid_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    valid: list[str] = []
    for path in paths:
        try:
            normalized = validate_workspace_relative_path(path.strip())
        except ValueError:
            continue
        if normalized not in seen:
            seen.add(normalized)
            valid.append(normalized)
    return valid


def _language_for_path(path: str) -> str:
    return language_for_path_or_default(path)


__all__ = [
    "PatchGate",
    "PatchGateConfig",
    "PatchGateResult",
    "build_repair_prompt",
    "changed_files_from_patch_text",
    "changed_files_from_tool_log",
    "configure_default_gate",
    "handle_tool_log",
    "provider_identity_for_kernel",
    "run_repair_round",
]
