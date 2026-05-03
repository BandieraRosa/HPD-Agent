"""Explicit fake providers for contract-first code intelligence tests."""

from __future__ import annotations

from .semantic import FakeSemanticProvider
from .syntax import FakeSyntaxProvider, PYTHON_FAKE_PATH, TYPESCRIPT_FAKE_PATH, fake_symbols


def create_fake_providers() -> tuple[FakeSyntaxProvider, FakeSemanticProvider]:
    """Create deterministic fake providers for tests and demos that opt in explicitly."""
    return (FakeSyntaxProvider(), FakeSemanticProvider())


__all__ = [
    "FakeSemanticProvider",
    "FakeSyntaxProvider",
    "PYTHON_FAKE_PATH",
    "TYPESCRIPT_FAKE_PATH",
    "create_fake_providers",
    "fake_symbols",
]
