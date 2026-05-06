"""Tests for the pathspec-backed text search provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Coroutine, Generator
from pathlib import Path
from typing import Protocol, TypeVar, cast

import pytest
from langchain_core.tools import BaseTool

from src.code_intel import CodeIntelKernel
from src.code_intel.core import Capability, Location, ToolResult
from src.code_intel.providers.text_search import InvalidSearchPath, TextSearchProvider
from src.code_intel.tools import code_search, set_code_intel_kernel
from src.code_intel.tools._langchain import strip_legacy_error_prefix

T = TypeVar("T")


class _AsyncInvokableTool(Protocol):
    def ainvoke(self, input: dict[str, object]) -> Awaitable[object]: ...


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


async def _ainvoke_text(item: BaseTool, args: dict[str, object]) -> str:
    invokable = cast(_AsyncInvokableTool, cast(object, item))
    return cast(str, await invokable.ainvoke(args))


def _payload(raw: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(strip_legacy_error_prefix(raw)))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _provider_result(result: ToolResult[object]) -> list[Location]:
    assert result.ok is True
    assert result.data is not None
    assert isinstance(result.data, list)
    items = cast(list[object], result.data)
    return [
        item if isinstance(item, Location) else Location.model_validate(item)
        for item in items
    ]


@pytest.fixture(autouse=True)
def reset_code_intel_kernel() -> Generator[None, None, None]:
    yield
    set_code_intel_kernel(None)


def test_text_search_respects_gitignore_and_limit(tmp_path: Path) -> None:
    _write(tmp_path / ".gitignore", "ignored/\nsecret.txt\n")
    _write(tmp_path / "src/a.py", "needle first\nneedle second\n")
    _write(tmp_path / "src/b.py", "needle third\n")
    _write(tmp_path / "ignored/hidden.py", "needle ignored\n")
    _write(tmp_path / "secret.txt", "needle secret\n")
    provider = TextSearchProvider(tmp_path)
    kernel = CodeIntelKernel([provider])

    result = _run(
        kernel.call(Capability.TEXT_SEARCH, "python", query="needle", limit=2)
    )
    locations = _provider_result(result)

    assert [location.path for location in locations] == ["src/a.py", "src/a.py"]
    assert result.meta.sources_used == ["text_search"]
    assert result.meta.more_available is True
    assert result.meta.truncated is False
    assert all(
        "ignored" not in location.path and location.path != "secret.txt"
        for location in locations
    )
    assert locations[0].range.start_line == 0
    assert locations[0].range.start_col == 0
    assert locations[0].range.end_col == len("needle")


def test_invalid_regex_returns_tool_error(tmp_path: Path) -> None:
    _write(tmp_path / "src/app.py", "needle\n")
    provider = TextSearchProvider(tmp_path)
    kernel = CodeIntelKernel([provider])

    kernel_result = _run(
        kernel.call(Capability.TEXT_SEARCH, "python", query="(", regex=True, limit=5)
    )

    assert kernel_result.ok is False
    assert kernel_result.error is not None
    assert kernel_result.error.code == "invalid_regex"
    assert "正则" in kernel_result.error.message

    set_code_intel_kernel(kernel)
    raw = _run(
        _ainvoke_text(
            code_search, {"query": "(", "mode": "text", "regex": True, "limit": 5}
        )
    )
    payload = _payload(raw)
    error = cast(dict[str, object], payload["error"])

    assert payload["ok"] is False
    assert error["code"] == "invalid_regex"
    assert "正则" in cast(str, error["message"])


def test_binary_and_oversized_files_are_skipped_with_truncated_meta(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "small.py", "needle visible\n")
    _ = (tmp_path / "binary.bin").write_bytes(b"needle\0hidden")
    _write(tmp_path / "large.py", "needle " + ("x" * 80))
    provider = TextSearchProvider(tmp_path, max_file_size_bytes=32)
    kernel = CodeIntelKernel([provider])

    result = _run(
        kernel.call(Capability.TEXT_SEARCH, "python", query="needle", limit=10)
    )
    locations = _provider_result(result)

    assert [location.path for location in locations] == ["small.py"]
    assert result.meta.truncated is True
    assert result.meta.more_available is False


def test_case_sensitivity_and_regex_ranges_are_correct(tmp_path: Path) -> None:
    _write(tmp_path / "src/app.py", "Alpha\nalpha token\n")
    provider = TextSearchProvider(tmp_path)

    insensitive = _run(provider.text_search("alpha", limit=10))
    sensitive = _run(provider.text_search("alpha", limit=10, case_sensitive=True))
    regex = _run(
        provider.text_search(r"a\w+ token", limit=10, regex=True, case_sensitive=True)
    )

    assert [location.range.start_line for location in insensitive] == [0, 1]
    assert [location.range.start_line for location in sensitive] == [1]
    assert len(regex) == 1
    assert regex[0].path == "src/app.py"
    assert regex[0].range.start_line == 1
    assert regex[0].range.start_col == 0
    assert regex[0].range.end_col == len("alpha token")


def test_path_filter_and_symlink_out_of_root_are_safe(tmp_path: Path) -> None:
    _write(tmp_path / "src/app.py", "needle app\n")
    _write(tmp_path / "docs/readme.md", "needle docs\n")
    outside = tmp_path.parent / "outside_text_search_target.txt"
    _ = outside.write_text("needle outside\n", encoding="utf-8")
    (tmp_path / "src/outside_link.py").symlink_to(outside)
    provider = TextSearchProvider(tmp_path)

    filtered = _run(provider.text_search("needle", path="src", limit=10))
    with pytest.raises(InvalidSearchPath):
        _ = _run(provider.text_search("needle", path="../", limit=10))

    assert [location.path for location in filtered] == ["src/app.py"]


def test_code_search_mixed_mode_falls_back_to_text_provider(tmp_path: Path) -> None:
    _write(tmp_path / "src/app.py", "needle from text provider\n")
    set_code_intel_kernel(CodeIntelKernel([TextSearchProvider(tmp_path)]))

    raw = _run(
        _ainvoke_text(code_search, {"query": "needle", "mode": "mixed", "limit": 5})
    )
    payload = _payload(raw)
    data = cast(dict[str, object], payload["data"])
    matches = cast(list[dict[str, object]], data["matches"])
    meta = cast(dict[str, object], payload["meta"])

    assert payload["ok"] is True
    assert [match["kind"] for match in matches] == ["text"]
    assert matches[0]["path"] == "src/app.py"
    assert matches[0]["source"] == "text_search"
    assert meta["sources_used"] == ["text_search"]
