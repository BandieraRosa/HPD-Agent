"""Pure workspace and language detection contracts."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, field_validator

from .models import validate_workspace_relative_path


class LanguageDetection(BaseModel):
    """Language detection result supplied by callers or later provider layers."""

    path: str = Field(description="Workspace-relative path using forward slashes.")
    language: str = Field(description="Detected lowercase language identifier, such as python.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence from 0.0 to 1.0.")
    source: str = Field(default="unknown", description="Machine-readable detection source.")

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)

    @field_validator("language", "source")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value:
            raise ValueError("value must not be empty")
        return value


class Workspace(BaseModel):
    """Workspace identity without scanning, indexing, or filesystem IO."""

    root_path: str = Field(
        validation_alias=AliasChoices("root_path", "workspace_root", "root"),
        description="Workspace root path supplied by the caller; not checked on disk.",
    )
    name: str | None = Field(default=None, description="Optional workspace display name.")
    languages: set[str] = Field(
        default_factory=set,
        description="Declared language identifiers known for this workspace.",
    )
    detections: list[LanguageDetection] = Field(
        default_factory=list,
        description="Optional precomputed language detections.",
    )

    @field_validator("root_path")
    @classmethod
    def validate_root_path(cls, root_path: str) -> str:
        if not root_path:
            raise ValueError("root_path must not be empty")
        return root_path


__all__ = ["LanguageDetection", "Workspace"]
