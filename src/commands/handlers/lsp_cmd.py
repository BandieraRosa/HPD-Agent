"""Handler for the /lsp command — inspect and manage configured LSP runtimes."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.agents import QueryAgent
from src.code_intel.config import load_code_intel_config
from src.code_intel.providers.lsp import default_language_server_specs
from src.code_intel.providers.lsp.manager import LSPManager

VALID_SUBS = ("status", "start", "stop", "restart")
_RUNTIME_ATTRS = (
    "code_intel_runtime",
    "code_intel_lsp_provider",
    "_code_intel_lsp_provider",
    "lsp_provider",
    "_lsp_provider",
    "code_intel_lsp_manager",
    "_code_intel_lsp_manager",
    "lsp_manager",
    "_lsp_manager",
)


@dataclass(frozen=True)
class _RunningServer:
    language: str
    name: str
    capabilities: str


async def run(raw: str, agent: QueryAgent) -> bool:
    """Dispatch /lsp subcommands without starting servers from status."""

    parts = raw.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "status"
    if sub not in VALID_SUBS:
        print("用法: /lsp [status|start <language>|stop|restart <language>]")
        print(f"可用子命令: {', '.join(VALID_SUBS)}")
        return False

    loaded = load_code_intel_config()
    if loaded.error is not None:
        print(loaded.error.format())
        return False
    config = loaded.config
    if not config.enabled or not config.lsp.enabled:
        print("语言服务器 LSP: code_intel.lsp 已在配置中禁用。")
        return False

    runtime = _runtime_from_agent(agent)
    if sub == "status":
        _print_status(config.lsp.languages, runtime)
        return False
    if sub == "start":
        if len(parts) < 3:
            print("用法: /lsp start <language>")
            print(f"可用语言: {', '.join(config.lsp.languages)}")
            return False
        await _start(runtime, agent, parts[2].lower())
        return False
    if sub == "stop":
        language = parts[2].lower() if len(parts) > 2 else None
        await _stop(runtime, language)
        return False
    if len(parts) < 3:
        print("用法: /lsp restart <language>")
        print(f"可用语言: {', '.join(config.lsp.languages)}")
        return False
    await _restart(runtime, parts[2].lower())
    return False


def _print_status(languages: list[str], runtime: object | None) -> None:
    specs = default_language_server_specs()
    running = _running_servers(runtime)
    print("语言服务器 LSP 状态（status 不会启动 server）:")
    if runtime is None:
        print("  runtime: 未绑定（没有运行中的 LSP manager）")
    else:
        print("  runtime: 已绑定")
    for language in languages:
        spec = specs.get(language)
        server_name = spec.name if spec is not None else "unknown"
        active = running.get(language)
        if active is None:
            print(f"  - {language} / {server_name}: 未运行")
        else:
            print(
                f"  - {language} / {active.name}: 运行中；capabilities: {active.capabilities}"
            )


async def _start(runtime: object | None, agent: QueryAgent, language: str) -> None:
    start_lsp = _callable_attr(runtime, "start_lsp")
    if start_lsp is not None:
        try:
            handle = await _maybe_await(
                cast(Callable[[str], object], start_lsp)(language)
            )
        except Exception as error:
            print(f"语言服务器 LSP 启动失败: {language}")
            print(f"原因: {error.__class__.__name__} - {error}")
            return
        _print_started(language, handle)
        return

    manager = (
        runtime if runtime is not None and hasattr(runtime, "ensure_client") else None
    )
    if manager is None:
        try:
            manager = LSPManager(workspace_root=str(Path.cwd()))
        except Exception as error:
            print("语言服务器 LSP 启动失败: 无法创建 manager。")
            print(f"原因: {error.__class__.__name__} - {error}")
            return

    try:
        handle = await cast(LSPManager, manager).ensure_client(language)
    except Exception as error:
        print(f"语言服务器 LSP 启动失败: {language}")
        print(f"原因: {error.__class__.__name__} - {error}")
        return

    setattr(agent, "_code_intel_lsp_manager", manager)
    try:
        from src.code_intel.providers.lsp.provider import LSPProvider
        from src.code_intel.tools.runtime import get_code_intel_kernel

        get_code_intel_kernel().register_provider(
            LSPProvider(Path.cwd(), manager=cast(LSPManager, manager))
        )
    except Exception as error:
        print(f"语言服务器 LSP provider 注册警告: {error.__class__.__name__}")
    _print_started(language, handle)


def _print_started(language: str, handle: object) -> None:
    spec = cast(object | None, getattr(handle, "spec", None))
    name = _str_attr(spec, "name") or "unknown"
    caps = getattr(handle, "capabilities", None)
    cap_names = _capability_summary(caps) if caps is not None else "未协商"
    print(f"语言服务器 LSP 已启动: {language} / {name}")
    print(f"  capabilities: {cap_names}")


async def _stop(runtime: object | None, language: str | None) -> None:
    if runtime is None:
        print("语言服务器 LSP 停止: 当前没有运行中的 LSP runtime。")
        return
    stop = (
        _callable_attr(runtime, "stop_lsp")
        or _callable_attr(runtime, "shutdown")
        or _callable_attr(_manager_from_runtime(runtime), "shutdown")
    )
    if stop is None:
        print("语言服务器 LSP 停止: runtime 不支持 shutdown。")
        return
    try:
        _ = await _maybe_await(cast(Callable[[str | None], object], stop)(language))
    except Exception as error:
        print("语言服务器 LSP 停止失败。")
        print(f"原因: {error.__class__.__name__}")
        return
    target = language or "全部语言"
    print(f"语言服务器 LSP 已停止: {target}")


async def _restart(runtime: object | None, language: str) -> None:
    if runtime is None:
        print(
            f"语言服务器 LSP 重启: 未绑定 runtime，未启动真实 server；无法重启 {language}。"
        )
        return
    restart = (
        _callable_attr(runtime, "restart_lsp")
        or _callable_attr(runtime, "restart")
        or _callable_attr(_manager_from_runtime(runtime), "restart")
    )
    if restart is None:
        print("语言服务器 LSP 重启: runtime 不支持 restart。")
        return
    try:
        health = await _maybe_await(cast(Callable[[str], object], restart)(language))
    except Exception as error:
        print(f"语言服务器 LSP 重启失败: {language}")
        print(f"原因: {error.__class__.__name__}")
        return
    message = _health_message(health)
    print(f"语言服务器 LSP 重启完成: {language}{message}")


def _runtime_from_agent(agent: QueryAgent) -> object | None:
    for attr in _RUNTIME_ATTRS:
        candidate = cast(object | None, getattr(agent, attr, None))
        if candidate is not None:
            return candidate
    return None


def _manager_from_runtime(runtime: object | None) -> object | None:
    if runtime is None:
        return None
    return cast(object | None, getattr(runtime, "manager", None))


def _running_servers(runtime: object | None) -> dict[str, _RunningServer]:
    manager = _manager_from_runtime(runtime) or runtime
    handles = cast(object | None, getattr(manager, "handles", None))
    if handles is None:
        return {}
    servers: dict[str, _RunningServer] = {}
    try:
        iterable = tuple(cast(Iterable[object], handles))
    except TypeError:
        return servers
    for handle in iterable:
        spec = cast(object | None, getattr(handle, "spec", None))
        language = _str_attr(spec, "language") or _str_attr(handle, "language")
        if not language:
            continue
        name = _str_attr(spec, "name") or "unknown"
        capabilities = _capability_summary(
            cast(object | None, getattr(handle, "capabilities", None))
        )
        servers[language] = _RunningServer(
            language=language, name=name, capabilities=capabilities
        )
    return servers


def _capability_summary(capabilities: object | None) -> str:
    if capabilities is None:
        return "未协商"
    names: list[str] = []
    for attr, label in (
        ("definition_provider", "definition"),
        ("references_provider", "references"),
        ("hover_provider", "hover"),
        ("document_symbol_provider", "document_symbols"),
        ("diagnostic_provider", "diagnostics"),
        ("text_document_sync", "diagnostics"),
    ):
        value = cast(object | None, getattr(capabilities, attr, None))
        if value not in (None, False) and label not in names:
            names.append(label)
    return ",".join(names) if names else "无已声明 capability"


def _str_attr(obj: object | None, attr: str) -> str:
    if obj is None:
        return ""
    value = cast(object | None, getattr(obj, attr, None))
    if isinstance(value, str):
        return value
    enum_value = cast(object | None, getattr(value, "value", None))
    return enum_value if isinstance(enum_value, str) else ""


def _callable_attr(obj: object | None, attr: str) -> object | None:
    if obj is None:
        return None
    value = cast(object | None, getattr(obj, attr, None))
    return value if callable(value) else None


async def _maybe_await(result: object) -> object:
    if inspect.isawaitable(result):
        return await cast(Awaitable[object], result)
    return result


def _health_message(health: object) -> str:
    status = _str_attr(health, "status")
    message = _str_attr(health, "message")
    suffix = ""
    if status:
        suffix += f"；status={status}"
    if message:
        suffix += f"；message={message}"
    return suffix


__all__ = ["VALID_SUBS", "run"]
