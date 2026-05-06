"""Tests for the async LSP stdio transport."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar, cast

import pytest

from src.code_intel.core import LSPTimeout, ProviderUnavailable
from src.code_intel.providers.lsp.transport import LSPTransport, encode_lsp_message

T = TypeVar("T")

_MOCK_TRANSPORT_SERVER = r'''
from __future__ import annotations

import json
import sys

SEPARATOR = b"\r\n\r\n"


def read_message():
    header = b""
    while not header.endswith(SEPARATOR):
        chunk = sys.stdin.buffer.read(1)
        if not chunk:
            return None
        header += chunk
    length = None
    for line in header[:-len(SEPARATOR)].split(b"\r\n"):
        name, separator, value = line.partition(b":")
        if separator and name.lower() == b"content-length":
            length = int(value.strip())
    if length is None:
        return None
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def write_message(payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


def write_malformed_header():
    sys.stdout.buffer.write(b"Content-Length: nope\r\n\r\n{}")
    sys.stdout.buffer.flush()


def write_malformed_json():
    body = b"{not-json"
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


def response_for(message, result):
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main():
    mode = sys.argv[1]
    if mode == "crash":
        print("mock crash line", file=sys.stderr, flush=True)
        raise SystemExit(7)
    pending = []
    while True:
        message = read_message()
        if message is None:
            return
        if mode == "malformed_header":
            write_malformed_header()
            return
        if mode == "malformed_json":
            write_malformed_json()
            return
        method = message.get("method")
        if mode == "timeout" and method == "mock/never":
            continue
        if mode == "error_response" and method == "mock/echo":
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            continue
        if mode == "malformed_error_response" and method == "mock/echo":
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": "not an object",
                }
            )
            continue
        if mode == "concurrent" and method == "mock/echo":
            pending.append(message)
            if len(pending) == 2:
                write_message({"jsonrpc": "2.0", "method": "mock/notice", "params": {"tag": "between"}})
                second = pending[1]
                first = pending[0]
                write_message(response_for(second, {"name": second.get("params", {}).get("name"), "order": 2}))
                write_message(response_for(first, {"name": first.get("params", {}).get("name"), "order": 1}))
            continue
        if method == "mock/echo":
            write_message(response_for(message, message.get("params")))


if __name__ == "__main__":
    main()
'''


def _run(coro: Coroutine[object, object, T]) -> T:
    return asyncio.run(coro)


def _server_command(tmp_path: Path, mode: str) -> list[str]:
    server_path = tmp_path / f"mock_transport_{mode}.py"
    _ = server_path.write_text(_MOCK_TRANSPORT_SERVER, encoding="utf-8")
    return [sys.executable, str(server_path), mode]


def test_content_length_uses_utf8_byte_length_and_crlf() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "mock/echo", "params": {"text": "é🙂"}}

    frame = encode_lsp_message(payload)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    assert frame == f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    assert b"\r\n\r\n" in frame


def test_concurrent_requests_map_responses_and_notifications(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "concurrent"), default_timeout=1.0)
        notifications: list[str] = []

        async def on_notice(params: object | None) -> None:
            payload = cast(dict[str, object], params)
            notifications.append(cast(str, payload["tag"]))

        await transport.add_notification_handler("mock/notice", on_notice)
        try:
            first = asyncio.create_task(transport.request("mock/echo", {"name": "first"}))
            second = asyncio.create_task(transport.request("mock/echo", {"name": "second"}))
            results = await asyncio.gather(first, second)
        finally:
            await transport.close()

        assert results == [{"name": "first", "order": 1}, {"name": "second", "order": 2}]
        assert notifications == ["between"]

    _run(scenario())


def test_request_timeout_maps_to_lsp_timeout(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "timeout"), default_timeout=0.05)
        try:
            with pytest.raises(LSPTimeout):
                _ = await transport.request("mock/never", {})
            assert await transport.is_running() is True
        finally:
            await transport.close()

    _run(scenario())


def test_jsonrpc_error_response_preserves_code_and_message(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(
            _server_command(tmp_path, "error_response"), default_timeout=1.0
        )
        try:
            with pytest.raises(ProviderUnavailable) as raised:
                _ = await transport.request("mock/echo", {"name": "bad"})
        finally:
            await transport.close()

        assert "code=-32601" in str(raised.value)
        assert "Method not found" in str(raised.value)
        assert raised.value.to_tool_error().code == "provider_unavailable"

    _run(scenario())


def test_jsonrpc_malformed_error_response_uses_generic_detail(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(
            _server_command(tmp_path, "malformed_error_response"), default_timeout=1.0
        )
        try:
            with pytest.raises(ProviderUnavailable) as raised:
                _ = await transport.request("mock/echo", {"name": "bad"})
        finally:
            await transport.close()

        assert str(raised.value) == "language server returned an error"
        assert raised.value.to_tool_error().code == "provider_unavailable"

    _run(scenario())


def test_malformed_header_marks_transport_unavailable(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "malformed_header"), default_timeout=1.0)
        try:
            with pytest.raises(ProviderUnavailable):
                _ = await transport.request("mock/echo", {"name": "bad"})
        finally:
            await transport.close()

    _run(scenario())


def test_malformed_json_marks_transport_unavailable(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "malformed_json"), default_timeout=1.0)
        try:
            with pytest.raises(ProviderUnavailable):
                _ = await transport.request("mock/echo", {"name": "bad"})
        finally:
            await transport.close()

    _run(scenario())


def test_server_crash_does_not_crash_kernel(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "crash"), default_timeout=0.5)
        try:
            await transport.start()
            with pytest.raises(ProviderUnavailable):
                _ = await transport.request("mock/echo", {"name": "after-crash"}, timeout=0.5)
            summary = await transport.stderr_summary()
        finally:
            await transport.close()

        assert "mock crash line" in summary
        assert transport.returncode is not None

    _run(scenario())


def test_shutdown_waits_for_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        transport = LSPTransport(_server_command(tmp_path, "echo"), default_timeout=1.0)
        await transport.start()
        assert await transport.is_running() is True
        await transport.close()
        assert await transport.is_running() is False
        assert transport.returncode is not None

    _run(scenario())
