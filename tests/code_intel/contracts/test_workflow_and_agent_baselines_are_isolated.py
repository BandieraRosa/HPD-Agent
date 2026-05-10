"""Contract for workflow/agent verification baseline isolation."""

from __future__ import annotations

from pathlib import Path

from src.code_intel.core import Diagnostic, DiagnosticSeverity, Range, Symbol, SymbolKind
from src.code_intel.verifier import (
    BaselineCache,
    build_baseline_key,
    get_cached_baseline,
    refresh_agent_baseline,
    refresh_workflow_baseline,
)


def _diagnostic(message: str) -> Diagnostic:
    return Diagnostic(
        path="src/app.py",
        range=Range(start_line=1, start_col=0, end_line=1, end_col=1),
        severity=DiagnosticSeverity.ERROR,
        message=message,
        code="reportUndefinedVariable",
        source="pyright",
        fingerprint=f"raw-{message}",
    )


def _symbol() -> Symbol:
    return Symbol(
        name="run",
        qualified_name="Service.run",
        kind=SymbolKind.FUNCTION,
        language="python",
        path="src/app.py",
        range=Range(start_line=0, start_col=0, end_line=3, end_col=0),
        selection_range=Range(start_line=0, start_col=4, end_line=0, end_col=7),
        signature="def run() -> None",
        doc=None,
        source="test",
        confidence=1.0,
        file_hash="file-hash",
        index_version="test-v1",
    )


def test_workflow_and_agent_baselines_are_isolated(tmp_path: Path) -> None:
    """workflow baseline 与 agent baseline 必须互不污染。"""
    source_path = tmp_path / "src" / "app.py"
    _ = source_path.parent.mkdir(parents=True)
    _ = source_path.write_text("def run() -> None:\n    pass\n", encoding="utf-8")
    cache = BaselineCache()

    workflow_key = build_baseline_key(
        workspace_root=tmp_path,
        relevant_paths=["src/app.py"],
        provider_id="pyright",
        provider_config_hash="config-v1",
        call_source="workflow",
    )
    agent_key = workflow_key.model_copy(update={"call_source": "agent"})

    assert workflow_key.workspace_hash
    assert workflow_key.relevant_paths == ("src/app.py",)
    assert workflow_key.content_hash
    assert workflow_key.provider_id == "pyright"
    assert workflow_key.provider_config_hash == "config-v1"
    assert workflow_key.cache_key() != agent_key.cache_key()
    assert workflow_key.bucket_key() != agent_key.bucket_key()

    workflow_snapshot = refresh_workflow_baseline(
        key=workflow_key,
        diagnostics=[_diagnostic("workflow baseline")],
        symbols_by_path={"src/app.py": [_symbol()]},
        cache=cache,
    )
    agent_snapshot = refresh_agent_baseline(
        key=agent_key,
        diagnostics=[_diagnostic("agent baseline")],
        symbols_by_path={"src/app.py": []},
        cache=cache,
    )

    assert workflow_snapshot.symbols_by_path["src/app.py"][0].qualified_name == "Service.run"
    assert get_cached_baseline(workflow_key, cache=cache) == workflow_snapshot
    assert get_cached_baseline(agent_key, cache=cache) == agent_snapshot

    _ = refresh_agent_baseline(
        key=agent_key,
        diagnostics=[_diagnostic("agent refreshed")],
        symbols_by_path={"src/app.py": []},
        cache=cache,
    )

    workflow_after_agent_refresh = get_cached_baseline(workflow_key, cache=cache)
    agent_after_refresh = get_cached_baseline(agent_key, cache=cache)

    assert workflow_after_agent_refresh is not None
    assert agent_after_refresh is not None
    assert workflow_after_agent_refresh.diagnostics[0].message == "workflow baseline"
    assert agent_after_refresh.diagnostics[0].message == "agent refreshed"
