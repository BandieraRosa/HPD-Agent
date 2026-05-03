"""Language-server registry for the LSP provider."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field


class LanguageServerSpec(BaseModel):
    """Static configuration needed to detect and launch one language server."""

    language: str = Field(description="Code-intel language identifier handled by this server.")
    name: str = Field(description="Human-readable language server name.")
    detect_command: list[str] = Field(description="Command shape used to detect whether the server executable exists.")
    launch_command: list[str] = Field(description="Command used to launch the stdio language server.")
    install_hint: str = Field(description="Exact install command shown when the executable is missing.")
    root_markers: list[str] = Field(description="Workspace-root marker filenames for this server.")
    init_options: dict[str, object] = Field(
        default_factory=dict,
        description="Initialization options passed to the server during initialize.",
    )


DEFAULT_LANGUAGE_SERVER_SPECS: tuple[LanguageServerSpec, ...] = (
    LanguageServerSpec(
        language="python",
        name="pyright",
        detect_command=["pyright", "--version"],
        launch_command=["pyright", "--stdio"],
        install_hint="npm i -g pyright",
        root_markers=["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", ".git"],
    ),
    LanguageServerSpec(
        language="typescript",
        name="typescript-language-server",
        detect_command=["typescript-language-server", "--version"],
        launch_command=["typescript-language-server", "--stdio"],
        install_hint="npm i -g typescript-language-server typescript",
        root_markers=["tsconfig.json", "package.json", ".git"],
    ),
    LanguageServerSpec(
        language="javascript",
        name="typescript-language-server",
        detect_command=["typescript-language-server", "--version"],
        launch_command=["typescript-language-server", "--stdio"],
        install_hint="npm i -g typescript-language-server typescript",
        root_markers=["jsconfig.json", "package.json", ".git"],
    ),
)


def default_language_server_specs() -> dict[str, LanguageServerSpec]:
    """Return fresh default specs keyed by language."""
    return {spec.language: spec.model_copy(deep=True) for spec in DEFAULT_LANGUAGE_SERVER_SPECS}


def language_server_specs(specs: Iterable[LanguageServerSpec] | None = None) -> dict[str, LanguageServerSpec]:
    """Normalize a custom spec iterable or return default specs."""
    if specs is None:
        return default_language_server_specs()
    return {spec.language: spec.model_copy(deep=True) for spec in specs}


__all__ = [
    "DEFAULT_LANGUAGE_SERVER_SPECS",
    "LanguageServerSpec",
    "default_language_server_specs",
    "language_server_specs",
]
