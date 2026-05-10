"""LangChain tool for focused code context through CodeIntelKernel."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from src.code_intel.core import Capability, CodeTarget, ToolResult

from ._helpers import (
    coerce_context_parts,
    coerce_target,
    context_model,
    error_json,
    kernel_error_json,
    language_for_target,
    merge_meta,
    safe_cast_tool_result,
    serialize_result,
    validation_error_json,
)
from ._langchain import code_intel_tool
from .models import CodeContextData
from .runtime import get_code_intel_kernel


@code_intel_tool
async def code_context(
    target: CodeTarget,
    include: (
        list[Literal["signature", "body", "parents", "imports", "nearby_symbols"]]
        | None
    ) = None,
    max_tokens: int = 4000,
) -> str:
    """代码上下文：按目标 symbol 提取最小必要上下文，优先用于修改任务。

    Args:
        target: CodeTarget JSON，可使用 symbol_id、anchor 或 location。
        include: 需要的上下文部分，默认 signature、body、imports。
        max_tokens: 上下文预算上限。
    """
    if max_tokens < 1:
        return error_json(
            "invalid_input", "max_tokens 必须大于 0。", "请提供正整数 token 预算。"
        )

    try:
        target_model = coerce_target(target)
        include_parts = coerce_context_parts(include)
    except (ValidationError, ValueError) as error:
        return validation_error_json(error)

    result = safe_cast_tool_result(
        await get_code_intel_kernel().call(
            Capability.CONTEXT_EXTRACT,
            language_for_target(target_model),
            target=target_model,
            include=include_parts,
            max_tokens=max_tokens,
        )
    )
    if not result.ok:
        return kernel_error_json(result)

    try:
        context = context_model(result.data)
    except (TypeError, ValueError) as error:
        return validation_error_json(error)

    data = CodeContextData(
        target_symbol=context.target_symbol,
        signature=context.signature,
        body=context.body,
        parents=context.parents,
        imports=context.imports,
        nearby_symbols=context.nearby_symbols,
        truncated=context.truncated,
    )
    return serialize_result(
        ToolResult[object](
            ok=True,
            data=data,
            meta=merge_meta(result.meta, truncated=context.truncated),
        )
    )
