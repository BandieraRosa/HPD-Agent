"""Typed LangChain tool decorator for code-intelligence tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

import langchain_core.tools as langchain_tools
from langchain_core.tools import BaseTool

_AsyncToolFunction = Callable[..., Awaitable[str]]
_AsyncToolDecorator = Callable[[_AsyncToolFunction], BaseTool]

code_intel_tool = cast(_AsyncToolDecorator, langchain_tools.tool)

__all__ = ["code_intel_tool"]
