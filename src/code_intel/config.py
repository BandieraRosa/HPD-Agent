"""Configuration loading for the Code Intelligence Kernel."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_DEFAULT_LSP_LANGUAGES = ("python", "typescript", "javascript")
_DEFAULT_CONFIG_ERROR_HINT = (
    "请检查 ~/.hpagent/config.json 中的 code_intel 段；"
    "如果暂时不需要自定义配置，可以删除该段并使用默认值。"
)


class CodeIntelIndexConfig(BaseModel):
    """Index-related user configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    auto_build_on_startup: bool = Field(default=True)
    respect_gitignore: bool = Field(default=True)
    max_file_size_bytes: int = Field(default=1_000_000, ge=1)


class CodeIntelLSPConfig(BaseModel):
    """LSP lifecycle and language-server configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True)
    languages: list[str] = Field(default_factory=lambda: list(_DEFAULT_LSP_LANGUAGES))
    prewarm_on_startup: bool = Field(default=False)
    request_timeout_ms: int = Field(default=5_000, ge=1)
    idle_shutdown_minutes: int = Field(default=10, ge=0)
    max_restart_count: int = Field(default=3, ge=0)

    @field_validator("languages")
    @classmethod
    def _validate_languages(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for language in value:
            clean = language.strip().lower()
            if not clean:
                raise ValueError("language must be a non-empty string")
            if clean not in normalized:
                normalized.append(clean)
        if not normalized:
            raise ValueError("languages must contain at least one language")
        return normalized


class CodeIntelProvidersConfig(BaseModel):
    """Provider toggles for code intelligence routing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    tree_sitter: bool = Field(default=True)
    text_search: bool = Field(default=True)
    lsp: bool = Field(default=True)


class CodeIntelVerifyConfig(BaseModel):
    """Post-edit verification configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    auto_verify_on_patch: bool = Field(default=True)
    max_repair_rounds: int = Field(default=2, ge=0)
    retreat_on_max_rounds: bool = Field(default=True)


class CodeIntelToolsConfig(BaseModel):
    """Default limits used by agent-facing code_intel tools."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    code_search_default_limit: int = Field(default=20, ge=1)
    code_context_default_max_tokens: int = Field(default=4_000, ge=1)
    code_semantic_default_max_results: int = Field(default=50, ge=1)


class CodeIntelConfig(BaseModel):
    """Typed code_intel config with zero-config defaults."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True)
    locale: str = Field(default="zh-CN")
    cache_dir: str = Field(default_factory=lambda: str(default_code_intel_cache_dir()))
    index: CodeIntelIndexConfig = Field(default_factory=CodeIntelIndexConfig)
    lsp: CodeIntelLSPConfig = Field(default_factory=CodeIntelLSPConfig)
    providers: CodeIntelProvidersConfig = Field(
        default_factory=CodeIntelProvidersConfig
    )
    verify: CodeIntelVerifyConfig = Field(default_factory=CodeIntelVerifyConfig)
    tools: CodeIntelToolsConfig = Field(default_factory=CodeIntelToolsConfig)

    @field_validator("locale", "cache_dir")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("value must be a non-empty string")
        return clean


class CodeIntelConfigError(BaseModel):
    """Chinese-first, predictable config load error."""

    code: str = Field(default="malformed_code_intel_config")
    message: str = Field(default="配置错误: code_intel 配置无效。")
    hint: str = Field(default=_DEFAULT_CONFIG_ERROR_HINT)
    path: str | None = Field(default=None)
    detail: str | None = Field(default=None)

    def format(self) -> str:
        lines = [self.message, f"提示: {self.hint}"]
        if self.path:
            lines.append(f"文件: {self.path}")
        if self.detail:
            lines.append(f"详情: {self.detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CodeIntelConfigLoadResult:
    """Result wrapper so callers can avoid exceptions in REPL paths."""

    config: CodeIntelConfig
    error: CodeIntelConfigError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def default_config_path() -> Path:
    """Return the HPD-Agent user config path."""

    return Path.home() / ".hpagent" / "config.json"


def default_code_intel_cache_dir() -> Path:
    """Return the default code_intel cache directory without creating it."""

    return Path.home() / ".hpagent" / "index"


def code_intel_cache_dir(config: CodeIntelConfig) -> Path:
    """Resolve the configured cache directory without creating it."""

    return Path(config.cache_dir).expanduser()


def code_intel_index_db_path(
    workspace_root: str | Path, config: CodeIntelConfig
) -> Path:
    """Return the per-workspace symbol DB path for a config."""

    resolved_workspace = Path(workspace_root).expanduser().resolve(strict=False)
    workspace_key = sha256(str(resolved_workspace).encode("utf-8")).hexdigest()
    return code_intel_cache_dir(config) / workspace_key / "symbols.db"


def load_code_intel_config(
    config_path: str | Path | None = None,
) -> CodeIntelConfigLoadResult:
    """Load code_intel config, returning defaults on missing config and errors on malformed config."""

    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else default_config_path()
    )
    defaults = CodeIntelConfig()
    if not path.exists():
        return CodeIntelConfigLoadResult(config=defaults)

    try:
        raw_config = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        return CodeIntelConfigLoadResult(
            config=defaults,
            error=_config_error(path, f"JSON 格式无效: {error.msg}"),
        )
    except OSError as error:
        return CodeIntelConfigLoadResult(
            config=defaults,
            error=_config_error(path, f"无法读取配置文件: {error.__class__.__name__}"),
        )

    if not isinstance(raw_config, Mapping):
        return CodeIntelConfigLoadResult(
            config=defaults,
            error=_config_error(path, "顶层配置必须是 JSON object"),
        )

    root_config = cast(Mapping[str, object], raw_config)
    raw_code_intel = root_config.get("code_intel")
    if raw_code_intel is None:
        return CodeIntelConfigLoadResult(config=defaults)
    if not isinstance(raw_code_intel, Mapping):
        return CodeIntelConfigLoadResult(
            config=defaults,
            error=_config_error(path, "code_intel 必须是 JSON object"),
        )

    try:
        config = CodeIntelConfig.model_validate(raw_code_intel)
    except ValidationError as error:
        return CodeIntelConfigLoadResult(
            config=defaults,
            error=_config_error(path, _validation_summary(error)),
        )
    return CodeIntelConfigLoadResult(config=config)


def _config_error(path: Path, detail: str) -> CodeIntelConfigError:
    return CodeIntelConfigError(
        message="配置错误: code_intel 配置无效。",
        path=str(path),
        detail=detail,
    )


def _validation_summary(error: ValidationError) -> str:
    first_error = error.errors(
        include_url=False, include_context=False, include_input=False
    )[0]
    location = (
        ".".join(str(part) for part in first_error.get("loc", ())) or "code_intel"
    )
    message = str(first_error.get("msg", "字段无效"))
    return f"{location}: {message}"


__all__ = [
    "CodeIntelConfig",
    "CodeIntelConfigError",
    "CodeIntelConfigLoadResult",
    "CodeIntelIndexConfig",
    "CodeIntelLSPConfig",
    "CodeIntelProvidersConfig",
    "CodeIntelToolsConfig",
    "CodeIntelVerifyConfig",
    "code_intel_cache_dir",
    "code_intel_index_db_path",
    "default_code_intel_cache_dir",
    "default_config_path",
    "load_code_intel_config",
]
