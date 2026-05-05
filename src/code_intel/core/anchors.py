"""Anchor and target models for stable code location."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import Location, validate_workspace_relative_path


class TextAnchor(BaseModel):
    """Fuzzy text anchor based on symbol names and surrounding source text."""

    path: str = Field(description="Workspace-relative path using forward slashes.")
    symbol_name: str | None = Field(
        default=None, description="Nearby symbol name for disambiguation."
    )
    needle: str | None = Field(default=None, description="Text fragment to locate.")
    surrounding_before: str | None = Field(
        default=None,
        description="Text before the needle for disambiguation.",
    )
    surrounding_after: str | None = Field(
        default=None,
        description="Text after the needle for disambiguation.",
    )
    occurrence: int | None = Field(
        default=None,
        ge=0,
        description="0-based occurrence index when multiple matches exist.",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        return validate_workspace_relative_path(path)


class CodeTarget(BaseModel):
    """Unified code target with symbol_id, anchor, then location priority."""

    symbol_id: str | None = Field(
        default=None, description="Stable symbol ID target, highest priority."
    )
    anchor: TextAnchor | None = Field(
        default=None, description="Fuzzy text anchor target, second priority."
    )
    location: Location | None = Field(
        default=None, description="Exact source location target, third priority."
    )

    @field_validator("symbol_id")
    @classmethod
    def validate_symbol_id(cls, symbol_id: str | None) -> str | None:
        if symbol_id is not None and not symbol_id:
            raise ValueError("symbol_id must not be empty")
        return symbol_id

    @model_validator(mode="after")
    def validate_has_target(self) -> "CodeTarget":
        if self.symbol_id is None and self.anchor is None and self.location is None:
            raise ValueError("CodeTarget requires symbol_id, anchor, or location")
        return self

    @property
    def priority(self) -> str:
        """Return the first usable target kind in symbol_id -> anchor -> location order."""
        if self.symbol_id is not None:
            return "symbol_id"
        if self.anchor is not None:
            return "anchor"
        if self.location is not None:
            return "location"
        raise ValueError("CodeTarget has no usable target")


class TargetResolver(Protocol):
    """Resolve a CodeTarget into an LSP-style workspace location."""

    async def resolve(self, target: CodeTarget) -> Location | None:
        """Return a resolved location, or None when the target cannot be resolved."""
        ...


__all__ = ["CodeTarget", "TargetResolver", "TextAnchor"]
