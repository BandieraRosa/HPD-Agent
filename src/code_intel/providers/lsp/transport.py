"""Async JSON-RPC/LSP stdio transport."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from src.code_intel.core import CodeIntelError, LSPTimeout, ProviderUnavailable

_HEADER_SEPARATOR = b"\r\n\r\n"
_JSONRPC_VERSION = "2.0"
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 5.0
_DEFAULT_STDERR_LINES = 200
_MAX_STDERR_LINE_CHARS = 500
_CLOSE_TIMEOUT_SECONDS = 1.0

RequestId = int | str
Payload = dict[str, object]


class NotificationHandler(Protocol):
    """Async callback for LSP notifications."""

    def __call__(self, params_payload: object | None, /) -> Awaitable[None]: ...


def encode_lsp_message(payload: Mapping[str, object]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _parse_content_length(header_block: bytes) -> int:
    content_length: int | None = None
    for line in header_block.split(b"\r\n"):
        if not line:
            continue
        name, separator, value = line.partition(b":")
        if not separator:
            raise ProviderUnavailable("malformed LSP header")
        if name.lower() != b"content-length":
            continue
        if content_length is not None:
            raise ProviderUnavailable("duplicate LSP content length")
        try:
            content_length = int(value.strip())
        except ValueError as error:
            raise ProviderUnavailable("invalid LSP content length") from error
    if content_length is None:
        raise ProviderUnavailable("missing LSP content length")
    if content_length < 0:
        raise ProviderUnavailable("invalid LSP content length")
    return content_length


def _jsonrpc_error_detail(error_payload: object) -> str:
    detail = "language server returned an error"
    if not isinstance(error_payload, Mapping):
        return detail
    mapping = cast(Mapping[object, object], error_payload)

    parts: list[str] = []
    code = mapping.get("code")
    if isinstance(code, str):
        parts.append(f"code={code[:80]}")
    elif isinstance(code, int) and not isinstance(code, bool):
        parts.append(f"code={code}")

    message = mapping.get("message")
    if isinstance(message, str) and message.strip():
        parts.append(f"message={' '.join(message.split())[:300]}")

    if not parts:
        return detail
    return f"{detail}: {', '.join(parts)}"


class LSPTransport:
    """Async stdio transport for JSON-RPC based language servers."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        default_timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stderr_lines: int = _DEFAULT_STDERR_LINES,
    ) -> None:
        self._command: tuple[str, ...] = tuple(command)
        self._cwd: Path | None = (
            Path(cwd).expanduser().resolve(strict=False) if cwd is not None else None
        )
        self._env: dict[str, str] | None = dict(env) if env is not None else None
        self._default_timeout: float = default_timeout
        self._stderr_lines: deque[str] = deque(maxlen=max(1, stderr_lines))
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None
        self._pending: dict[RequestId, asyncio.Future[object | None]] = {}
        self._notification_handlers: dict[str, list[NotificationHandler]] = {}
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._start_lock: asyncio.Lock = asyncio.Lock()
        self._next_request_id: int = 1
        self._closing: bool = False
        self._closed: bool = False
        self._unavailable_detail: str | None = None

    @property
    def returncode(self) -> int | None:
        process = self._process
        return None if process is None else process.returncode

    async def start(self) -> None:
        """Launch the language server process if needed."""
        async with self._start_lock:
            if self._closed:
                raise ProviderUnavailable("LSP transport is closed")
            if not self._command:
                raise ProviderUnavailable("LSP launch command is empty")
            if self._process is not None:
                self._ensure_available()
                return

            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self._cwd) if self._cwd is not None else None,
                    env=self._env,
                )
            except OSError as error:
                raise ProviderUnavailable("failed to launch language server") from error

            if (
                process.stdin is None
                or process.stdout is None
                or process.stderr is None
            ):
                raise ProviderUnavailable("language server stdio pipes are unavailable")

            self._process = process
            self._stdout_task = asyncio.create_task(
                self._read_stdout_loop(process.stdout)
            )
            self._stderr_task = asyncio.create_task(
                self._read_stderr_loop(process.stderr)
            )
            self._wait_task = asyncio.create_task(self._wait_process_loop(process))

    async def add_notification_handler(
        self, method: str, handler: NotificationHandler
    ) -> None:
        """Register an async notification handler for a method."""
        self._notification_handlers.setdefault(method, []).append(handler)

    async def request(
        self, method: str, params: object | None = None, *, timeout: float | None = None
    ) -> object | None:
        """Send a JSON-RPC request and await its response."""
        await self.start()
        self._ensure_available()
        loop = asyncio.get_running_loop()
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[object | None] = loop.create_future()
        self._pending[request_id] = future

        payload: Payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        try:
            await self._write_payload(payload)
            request_timeout = self._default_timeout if timeout is None else timeout
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=request_timeout
            )
        except asyncio.TimeoutError:
            pending = self._pending.pop(request_id, None)
            if pending is not None and not pending.done():
                _ = pending.cancel()
            raise LSPTimeout(f"LSP request timed out: {method}") from None
        except CodeIntelError:
            self._cancel_pending_request(request_id)
            raise
        except Exception:
            self._cancel_pending_request(request_id)
            raise ProviderUnavailable("LSP request failed") from None

    async def notify(self, method: str, params: object | None = None) -> None:
        """Send a JSON-RPC notification."""
        await self.start()
        self._ensure_available()
        payload: Payload = {"jsonrpc": _JSONRPC_VERSION, "method": method}
        if params is not None:
            payload["params"] = params
        await self._write_payload(payload)

    async def is_running(self) -> bool:
        process = self._process
        return process is not None and process.returncode is None and not self._closed

    async def stderr_summary(self, max_lines: int = 20) -> str:
        line_count = max(0, min(max_lines, len(self._stderr_lines)))
        if line_count == 0:
            return ""
        return "\n".join(list(self._stderr_lines)[-line_count:])

    async def close(self) -> None:
        """Close pipes, stop the child process, and resolve pending requests safely."""
        if self._closed:
            return
        self._closing = True
        self._fail_pending(ProviderUnavailable("LSP transport is closed"))
        process = self._process
        if process is not None:
            await self._close_stdin(process)
            await self._wait_or_stop_process(process)
        await self._cancel_background_tasks()
        self._closed = True

    async def _write_payload(self, payload: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            self._mark_unavailable("language server is not running")
            raise ProviderUnavailable("language server is not running")
        data = encode_lsp_message(payload)
        async with self._write_lock:
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._mark_unavailable("language server pipe is closed")
                raise ProviderUnavailable("language server pipe is closed") from None

    async def _read_stdout_loop(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                message = await self._read_message(reader)
                if message is None:
                    break
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except CodeIntelError as error:
            self._mark_unavailable(error.detail or "invalid language server output")
        except Exception:
            self._mark_unavailable("invalid language server output")
        finally:
            if (
                not self._closing
                and self._process is not None
                and self._process.returncode is not None
            ):
                self._mark_unavailable("language server exited unexpectedly")

    async def _read_stderr_loop(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                self._stderr_lines.append(self._safe_stderr_line(line))
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _wait_process_loop(self, process: asyncio.subprocess.Process) -> None:
        try:
            _ = await process.wait()
        except asyncio.CancelledError:
            raise
        if not self._closing:
            self._mark_unavailable("language server exited unexpectedly")

    async def _read_message(self, reader: asyncio.StreamReader) -> Payload | None:
        try:
            header_bytes = await reader.readuntil(_HEADER_SEPARATOR)
        except asyncio.IncompleteReadError as error:
            if not error.partial:
                return None
            raise ProviderUnavailable("incomplete LSP header") from error
        except asyncio.LimitOverrunError as error:
            raise ProviderUnavailable("oversized LSP header") from error

        header_block = header_bytes[: -len(_HEADER_SEPARATOR)]
        content_length = _parse_content_length(header_block)
        try:
            body = await reader.readexactly(content_length)
        except asyncio.IncompleteReadError as error:
            raise ProviderUnavailable("incomplete LSP body") from error
        try:
            decoded = body.decode("utf-8")
            payload = cast(object, json.loads(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderUnavailable("malformed LSP JSON") from error
        if not isinstance(payload, dict):
            raise ProviderUnavailable("LSP payload must be an object")
        return cast(Payload, payload)

    async def _handle_message(self, message: Mapping[str, object]) -> None:
        method = message.get("method")
        if isinstance(method, str):
            await self._dispatch_notification(method, message.get("params"))
            return

        message_id = message.get("id")
        if not isinstance(message_id, (int, str)):
            return
        future = self._pending.pop(message_id, None)
        if future is None or future.done():
            return
        if "error" in message:
            future.set_exception(
                ProviderUnavailable(_jsonrpc_error_detail(message.get("error")))
            )
            return
        future.set_result(message.get("result"))

    async def _dispatch_notification(self, method: str, params: object | None) -> None:
        handlers = tuple(self._notification_handlers.get(method, ()))
        for handler in handlers:
            try:
                await handler(params)
            except Exception:
                continue

    def _ensure_available(self) -> None:
        if self._unavailable_detail is not None:
            raise ProviderUnavailable(self._unavailable_detail)
        process = self._process
        if process is not None and process.returncode is not None:
            self._mark_unavailable("language server exited unexpectedly")
            raise ProviderUnavailable("language server exited unexpectedly")

    def _mark_unavailable(self, detail: str) -> None:
        if self._unavailable_detail is None:
            self._unavailable_detail = detail
        self._fail_pending(ProviderUnavailable(detail))

    def _fail_pending(self, error: CodeIntelError) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                _ = future.set_exception(error)

    def _cancel_pending_request(self, request_id: RequestId) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            _ = future.cancel()

    async def _close_stdin(self, process: asyncio.subprocess.Process) -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.close()
            await process.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            return

    async def _wait_or_stop_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            _ = await asyncio.wait_for(process.wait(), timeout=_CLOSE_TIMEOUT_SECONDS)
            return
        except asyncio.TimeoutError:
            pass
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            _ = await asyncio.wait_for(process.wait(), timeout=_CLOSE_TIMEOUT_SECONDS)
            return
        except asyncio.TimeoutError:
            pass
        try:
            process.kill()
        except ProcessLookupError:
            return
        _ = await process.wait()

    async def _cancel_background_tasks(self) -> None:
        tasks = [
            task
            for task in (self._stdout_task, self._stderr_task, self._wait_task)
            if task is not None and not task.done()
        ]
        for task in tasks:
            _ = task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            _ = results

    @staticmethod
    def _safe_stderr_line(raw_line: bytes) -> str:
        line = raw_line.decode("utf-8", errors="replace").replace("\x00", "�").strip()
        if len(line) > _MAX_STDERR_LINE_CHARS:
            return f"{line[:_MAX_STDERR_LINE_CHARS]}…"
        return line


__all__ = ["LSPTransport", "NotificationHandler", "encode_lsp_message"]
