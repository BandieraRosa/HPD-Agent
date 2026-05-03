"""Contract tests for code_intel package import boundaries."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CODE_INTEL_ROOT = SRC_ROOT / "code_intel"
FORBIDDEN_LEGACY_PACKAGE = SRC_ROOT / "codeintel"

HIGHER_LAYER_NAMES = {"providers", "index", "verifier", "workflow", "tools"}
EXISTING_AGENT_MODULES = {"src.nodes", "src.agents", "nodes", "agents"}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _imported_modules(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            modules.append((module, node.level))

    return modules


def _first_segment(module: str) -> str:
    return module.split(".", 1)[0] if module else ""


def _is_code_intel_provider_import(module: str) -> bool:
    return (
        module == "src.code_intel.providers"
        or module.startswith("src.code_intel.providers.")
        or module == "code_intel.providers"
        or module.startswith("code_intel.providers.")
    )


def _is_existing_agent_import(module: str) -> bool:
    return any(module == forbidden or module.startswith(f"{forbidden}.") for forbidden in EXISTING_AGENT_MODULES)


def test_package_path_is_code_intel_not_codeintel() -> None:
    assert CODE_INTEL_ROOT.is_dir()
    assert not FORBIDDEN_LEGACY_PACKAGE.exists()


def _public_exports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        exports_targeted = any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        if not exports_targeted or not isinstance(node.value, ast.List):
            continue

        exports: list[str] = []
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                exports.append(element.value)
        return exports

    return []


def test_public_package_imports_kernel_facade_only() -> None:
    code_intel = importlib.import_module("src.code_intel")

    assert _public_exports(CODE_INTEL_ROOT / "__init__.py") == ["CodeIntelKernel"]
    assert hasattr(code_intel, "CodeIntelKernel")


def test_core_does_not_import_higher_layers() -> None:
    core_root = CODE_INTEL_ROOT / "core"
    offenders: list[str] = []

    for path in _python_files(core_root):
        for module, level in _imported_modules(path):
            imports_high_layer = (
                module.startswith("src.code_intel.")
                and _first_segment(module.removeprefix("src.code_intel.")) in HIGHER_LAYER_NAMES
            ) or (
                module.startswith("code_intel.")
                and _first_segment(module.removeprefix("code_intel.")) in HIGHER_LAYER_NAMES
            ) or (level > 0 and _first_segment(module) in HIGHER_LAYER_NAMES)
            if imports_high_layer:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {module or '<relative>'}")

    assert offenders == []


def test_providers_do_not_import_each_other() -> None:
    providers_root = CODE_INTEL_ROOT / "providers"
    provider_names = {path.name for path in providers_root.iterdir() if path.is_dir()}
    offenders: list[str] = []

    for provider_root in sorted(path for path in providers_root.iterdir() if path.is_dir()):
        current_provider = provider_root.name
        other_providers = provider_names - {current_provider}
        for path in _python_files(provider_root):
            for module, level in _imported_modules(path):
                absolute_peer_import = any(
                    module == f"src.code_intel.providers.{name}"
                    or module.startswith(f"src.code_intel.providers.{name}.")
                    or module == f"code_intel.providers.{name}"
                    or module.startswith(f"code_intel.providers.{name}.")
                    for name in other_providers
                )
                relative_peer_import = level > 0 and _first_segment(module) in other_providers
                if absolute_peer_import or relative_peer_import:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {module or '<relative>'}")

    assert offenders == []


def test_tools_do_not_directly_import_providers() -> None:
    tools_root = CODE_INTEL_ROOT / "tools"
    offenders: list[str] = []

    for path in _python_files(tools_root):
        for module, level in _imported_modules(path):
            imports_provider = _is_code_intel_provider_import(module) or (
                level > 0 and _first_segment(module) == "providers"
            )
            if imports_provider:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {module or '<relative>'}")

    assert offenders == []


def test_only_workflow_may_import_existing_agents_or_nodes() -> None:
    workflow_root = CODE_INTEL_ROOT / "workflow"
    offenders: list[str] = []

    for path in _python_files(CODE_INTEL_ROOT):
        if path == workflow_root or workflow_root in path.parents:
            continue
        for module, _level in _imported_modules(path):
            if _is_existing_agent_import(module):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {module}")

    assert offenders == []
