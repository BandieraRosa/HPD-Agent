"""Chinese-first edit policy injected into execution prompts."""

from __future__ import annotations


def get_prompt_section() -> str:
    """Return the workflow edit-policy section appended to sub-task prompts."""
    return (
        "\n\n【代码编辑策略（强制）】\n"
        "- 编辑前：先用 code_search / code_outline 定位相关文件与符号；需要理解调用关系、定义、引用或悬停信息时，再用 code_context / code_semantic 获取上下文。\n"
        "- 编辑中：写文件只能使用 apply_patch，并遵循 dry_run=True 校验后 dry_run=False 写入的流程。\n"
        '- 编辑后：必须调用 code_verify(scope="changed") 验证 changed 范围；不要把 changed 验证扩大成 workspace-wide verify。\n'
        "- 若 code_verify 返回 partial 或 provider_error，这不是成功验证；请把结构化结果写入总结，并说明仍需显式复验。"
    )


__all__ = ["get_prompt_section"]
