"""Safe observability helpers for code_intel tracing."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping, Sequence
from types import TracebackType
from typing import Final, Literal, Protocol, final, cast

from src.code_intel.core import CodeIntelError, ToolError
from src.code_intel.core.models import validate_workspace_relative_path
from src.core.observability import get_tracer

ALLOWED_TRACE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "elapsed_ms",
        "provider_name",
        "capability",
        "language",
        "cache_hit",
        "fallback_chain",
        "result_count",
        "truncated",
        "error",
        "error.code",
        "error.message",
        "path",
        "paths",
        "workspace_relative_path",
        "workspace_relative_paths",
        "diagnostic_templates",
    }
)

_LABEL_FIELDS = {"provider_name", "capability", "language"}
_BOOL_FIELDS = {"cache_hit", "truncated"}
_PATH_FIELDS = {"path", "workspace_relative_path"}
_PATH_LIST_FIELDS = {"paths", "workspace_relative_paths"}
_DIAGNOSTIC_MESSAGE_FIELDS = {"diagnostic_messages", "diagnostic_templates", "diagnostics"}
_DEFAULT_ERROR_CODE = "code_intel_error"
_DEFAULT_ERROR_MESSAGE = "代码智能服务发生错误。"
_MAX_LABEL_CHARS = 80
_MAX_ERROR_CODE_CHARS = 80
_MAX_DIAGNOSTIC_TEMPLATES = 8
_MAX_DIAGNOSTIC_TEMPLATE_CHARS = 180
_QUOTED_RE = re.compile(r"([\"'`])(?:\\.|(?!\1).)*\1")
_LINE_COL_TUPLE_RE = re.compile(r"\(\s*\d+\s*,\s*\d+\s*\)")
_LINE_COLON_RE = re.compile(r":\d+:\d+\b")
_LINE_REF_RE = re.compile(r"\b([Ll]ines?)\s+\d+(?:\s*(?:-|to|through)\s*\d+)?")
_COLUMN_REF_RE = re.compile(r"\b([Cc]ol(?:umn)?s?)\s+\d+")
_FLOAT_RE = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
_INTEGER_RE = re.compile(r"(?<![\w.])-?\d+(?![\w.])")

class _TracerLike(Protocol):
    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        model: str = "",
        metadata: dict[str, object] | None = None,
    ) -> str: ...

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        tokens_in: int = 0,
        tokens_out: int = 0,
        error_msg: str = "",
    ) -> None: ...


@final
class CodeIntelTraceSpan:
    """Exception-safe wrapper around the shared observability tracer."""

    __slots__: Final = ("_name", "_input_metadata", "_metadata", "_span_id", "_started_at", "_tracer")

    def __init__(self, name: str, metadata: Mapping[str, object] | None = None) -> None:
        self._name: str = name
        self._input_metadata: Mapping[str, object] | None = metadata
        self._metadata: dict[str, object] = {}
        self._span_id: str = ""
        self._started_at: float = 0.0
        self._tracer: _TracerLike | None = None

    def __enter__(self) -> "CodeIntelTraceSpan":
        self._started_at = time.perf_counter()
        self._metadata = redact(self._input_metadata, schema=ALLOWED_TRACE_FIELDS)
        try:
            tracer = cast(_TracerLike, get_tracer())
            self._tracer = tracer
            self._span_id = tracer.start_span(self._name, metadata=self._metadata)
        except Exception:
            self._tracer = None
            self._span_id = ""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        _ = (exc_type, tb)
        status = "ok"
        error_msg = ""
        if exc is not None:
            status = "error"
            error_metadata = safe_error_metadata(exc)
            self.add_metadata(error_metadata)
            error = self._metadata.get("error")
            if isinstance(error, dict):
                error_map = cast(dict[str, object], error)
                message = error_map.get("message")
                if isinstance(message, str):
                    error_msg = message
        self.add_metadata({"elapsed_ms": _elapsed_ms(self._started_at)})
        if self._span_id and self._tracer is not None:
            try:
                self._tracer.end_span(self._span_id, status=status, error_msg=error_msg)
            except Exception:
                pass
        return False

    def add_metadata(self, payload: Mapping[str, object] | None) -> None:
        """Merge sanitized metadata into the active span payload."""
        if payload is None:
            return
        try:
            self._metadata.update(redact(payload, schema=ALLOWED_TRACE_FIELDS))
        except Exception:
            return


def trace_span(name: str, metadata: Mapping[str, object] | None = None) -> CodeIntelTraceSpan:
    """Return a safe span context manager using the shared observability tracer."""
    return CodeIntelTraceSpan(name, metadata)


def redact(
    payload: Mapping[str, object] | None,
    *,
    schema: Iterable[str] = ALLOWED_TRACE_FIELDS,
) -> dict[str, object]:
    """Whitelist and normalize trace metadata before it reaches observability.

    The function accepts deliberately broad input so call sites can pass small
    local dictionaries defensively. Only whitelisted fields survive, source-like
    fields are ignored, paths must be workspace-relative, errors are converted to
    Chinese generic messages, and diagnostic messages become normalized templates.
    """
    allowed = set(schema)
    if not payload:
        return {}

    output: dict[str, object] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key in _LABEL_FIELDS and key in allowed:
            label = _safe_label(value)
            if label is not None:
                output[key] = label
            continue
        if key in _BOOL_FIELDS and key in allowed:
            output[key] = bool(value)
            continue
        if key == "elapsed_ms" and key in allowed:
            elapsed = _safe_non_negative_int(value)
            if elapsed is not None:
                output[key] = elapsed
            continue
        if key == "result_count" and key in allowed:
            count = _safe_non_negative_int(value)
            if count is not None:
                output[key] = count
            continue
        if key == "fallback_chain" and key in allowed:
            fallback_chain = _safe_label_list(value)
            if fallback_chain:
                output[key] = fallback_chain
            continue
        if key in _PATH_FIELDS and key in allowed:
            path = _safe_workspace_path(value)
            if path is not None:
                output[key] = path
            continue
        if key in _PATH_LIST_FIELDS and key in allowed:
            paths = _safe_workspace_path_list(value)
            if paths:
                output[key] = paths
            continue
        if key == "error" and ("error" in allowed or "error.code" in allowed or "error.message" in allowed):
            error = _safe_error_mapping(value)
            if error:
                output["error"] = error
            continue
        if key == "error.code" and "error.code" in allowed:
            code = _safe_error_code(value)
            if code is not None:
                _set_error_field(output, "code", code)
            continue
        if key == "error.message" and "error.message" in allowed:
            message = _safe_chinese_message(value)
            _set_error_field(output, "message", message)
            continue
        if key in _DIAGNOSTIC_MESSAGE_FIELDS and "diagnostic_templates" in allowed:
            templates = _diagnostic_templates(value)
            if templates:
                output["diagnostic_templates"] = templates
            continue
    return output


def safe_error_metadata(error: object) -> dict[str, object]:
    """Return safe Chinese error metadata for trace spans."""
    tool_error: ToolError
    if isinstance(error, ToolError):
        tool_error = error
    elif isinstance(error, CodeIntelError):
        tool_error = error.to_tool_error()
    else:
        tool_error = ToolError(code=_DEFAULT_ERROR_CODE, message=_DEFAULT_ERROR_MESSAGE)
    return {"error": {"code": tool_error.code, "message": tool_error.message}}


def result_count(value: object) -> int:
    """Return a non-sensitive count for common result containers."""
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return len(cast(Mapping[object, object], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return len(list(value))
        except Exception:
            return 1
    return 1


def _elapsed_ms(started_at: float) -> int:
    if started_at <= 0.0:
        return 0
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _safe_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return max(0, number)


def _safe_label(value: object) -> str | None:
    raw_value = getattr(value, "value", value)
    text = str(raw_value).strip()
    if not text:
        return None
    safe = "".join(char for char in text if char.isalnum() or char in {"_", "-", "."})
    if not safe:
        return None
    return safe[:_MAX_LABEL_CHARS]


def _safe_label_list(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        candidate_values: Iterable[object] = (value,)
    elif isinstance(value, Iterable):
        candidate_values = value
    else:
        candidate_values = (value,)
    labels: list[str] = []
    for item in candidate_values:
        label = _safe_label(item)
        if label is not None:
            labels.append(label)
    return labels


def _safe_workspace_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return validate_workspace_relative_path(value)
    except ValueError:
        return None


def _safe_workspace_path_list(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        candidate_values: Iterable[object] = (value,)
    elif isinstance(value, Iterable):
        candidate_values = value
    else:
        return []
    paths: list[str] = []
    for item in candidate_values:
        path = _safe_workspace_path(item)
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def _safe_error_mapping(value: object) -> dict[str, str]:
    if isinstance(value, ToolError):
        return {"code": _safe_error_code(value.code) or _DEFAULT_ERROR_CODE, "message": _safe_chinese_message(value.message)}
    if isinstance(value, CodeIntelError):
        tool_error = value.to_tool_error()
        return {"code": _safe_error_code(tool_error.code) or _DEFAULT_ERROR_CODE, "message": _safe_chinese_message(tool_error.message)}
    if not isinstance(value, Mapping):
        return {"code": _DEFAULT_ERROR_CODE, "message": _DEFAULT_ERROR_MESSAGE}
    mapping = cast(Mapping[object, object], value)
    code = _safe_error_code(mapping.get("code")) or _DEFAULT_ERROR_CODE
    message = _safe_chinese_message(mapping.get("message"))
    return {"code": code, "message": message}


def _safe_error_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    safe = "".join(char for char in value if char.isalnum() or char in {"_", "-", "."})
    if not safe:
        return None
    return safe[:_MAX_ERROR_CODE_CHARS]


def _safe_chinese_message(value: object) -> str:
    if isinstance(value, str) and _has_cjk(value):
        return " ".join(value.split())[:120]
    return _DEFAULT_ERROR_MESSAGE


def _has_cjk(value: str) -> bool:
    return any("一" <= char <= "鿿" for char in value)


def _diagnostic_templates(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)):
        candidate_values: Iterable[object] = (value,)
    elif isinstance(value, Iterable):
        candidate_values = value
    else:
        candidate_values = (value,)

    templates: list[str] = []
    for item in candidate_values:
        message = _diagnostic_message(item)
        if message is None:
            continue
        template = _normalize_diagnostic_message(message)
        if template and template not in templates:
            templates.append(template[:_MAX_DIAGNOSTIC_TEMPLATE_CHARS])
        if len(templates) >= _MAX_DIAGNOSTIC_TEMPLATES:
            break
    return templates


def _diagnostic_message(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        message = mapping.get("message")
        return message if isinstance(message, str) else None
    message = getattr(value, "message", None)
    return message if isinstance(message, str) else None


def _normalize_diagnostic_message(message: str) -> str:
    normalized = _QUOTED_RE.sub("<quoted>", message)
    normalized = _LINE_COL_TUPLE_RE.sub("(<line>, <column>)", normalized)
    normalized = _LINE_COLON_RE.sub(":<line>:<column>", normalized)
    normalized = _LINE_REF_RE.sub(lambda match: f"{match.group(1)} <line>", normalized)
    normalized = _COLUMN_REF_RE.sub(lambda match: f"{match.group(1)} <column>", normalized)
    normalized = _FLOAT_RE.sub("<float>", normalized)
    normalized = _INTEGER_RE.sub("<int>", normalized)
    return " ".join(normalized.split())


def _set_error_field(output: dict[str, object], field: str, value: str) -> None:
    error = output.get("error")
    if isinstance(error, dict):
        error_map = cast(dict[str, object], error)
    else:
        error_map = {}
        output["error"] = error_map
    error_map[field] = value


__all__ = [
    "ALLOWED_TRACE_FIELDS",
    "CodeIntelTraceSpan",
    "redact",
    "result_count",
    "safe_error_metadata",
    "trace_span",
]
