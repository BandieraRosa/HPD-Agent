"""Handler for the /index command — inspect and maintain code_intel symbol index."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from src.agents import QueryAgent
from src.code_intel.config import (
    CodeIntelConfig,
    code_intel_index_db_path,
    load_code_intel_config,
)
from src.code_intel.core import Symbol

VALID_SUBS = ("status", "build", "clear")


@dataclass(frozen=True)
class _IndexStatus:
    exists: bool
    db_path: Path
    file_count: int = 0
    symbol_count: int = 0
    last_indexed_at: float | None = None
    fts_available: bool = False
    error: str | None = None


async def run(raw: str, agent: QueryAgent) -> bool:
    """Dispatch /index subcommands."""

    _ = agent
    parts = raw.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "status"
    if sub not in VALID_SUBS:
        print("用法: /index [status|build|clear]")
        print(f"可用子命令: {', '.join(VALID_SUBS)}")
        return False

    loaded = load_code_intel_config()
    if loaded.error is not None:
        print(loaded.error.format())
        return False
    config = loaded.config
    if not config.enabled:
        print("代码索引: code_intel 已在配置中禁用。")
        return False

    if sub == "status":
        await _run_status(config)
    elif sub == "build":
        await _run_build(config)
    else:
        await _run_clear(config)
    return False


async def _run_status(config: CodeIntelConfig) -> None:
    workspace_root = Path.cwd()
    db_path = code_intel_index_db_path(workspace_root, config)
    status = await asyncio.to_thread(_inspect_index_db, db_path)
    print("代码索引状态:")
    print(f"  workspace: {workspace_root}")
    print(f"  db: {status.db_path}")
    if not status.exists:
        print("  状态: 未建立（运行 /index build 创建索引）")
        return
    if status.error is not None:
        print(f"  状态: 读取失败（不会自动重建）：{status.error}")
        return
    print("  状态: 已建立")
    print(f"  文件数: {status.file_count}")
    print(f"  符号数: {status.symbol_count}")
    if status.last_indexed_at is not None:
        ts = status.last_indexed_at
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
        print(f"  最近索引时间: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  FTS: {'可用' if status.fts_available else '未启用或不可用'}")


async def _run_build(config: CodeIntelConfig) -> None:
    workspace_root = Path.cwd()
    db_path = code_intel_index_db_path(workspace_root, config)
    try:
        from src.code_intel.index import SymbolIndexer
        from src.code_intel.providers.tree_sitter import (
            TREE_SITTER_INDEX_VERSION,
            TREE_SITTER_QUERY_VERSION,
            TreeSitterProvider,
        )

        provider = TreeSitterProvider(workspace_root)

        async def extract_symbols(
            workspace_root: Path, path: str, language: str
        ) -> Sequence[Symbol]:
            _ = workspace_root, language
            return await provider.outline(path)

        indexer = SymbolIndexer(
            workspace_root,
            extractor=extract_symbols,
            db_path=db_path,
            grammar_version=TREE_SITTER_INDEX_VERSION,
            query_version=TREE_SITTER_QUERY_VERSION,
            max_file_size_bytes=config.index.max_file_size_bytes,
        )
        result = await indexer.index_workspace()
        await indexer.store.close()
    except ModuleNotFoundError as error:
        print("代码索引构建失败。")
        print(f"原因: 缺少模块 - {error.name}")
        print(
            f"提示: 请确认 {error.name} 已安装（pip install {error.name} 或 pip install -e .[code-intel-full]）"
        )
        return
    except ImportError as error:
        print("代码索引构建失败。")
        print(f"原因: 导入失败 - {error}")
        print("提示: 请确认所有依赖已安装，或稍后使用 /index status 查看已有索引。")
        return
    except Exception as error:
        print("代码索引构建失败。")
        print(f"原因: {error.__class__.__name__} - {error}")
        print("提示: 如果问题持续，请查看完整堆栈或使用 /index status 查看已有索引。")
        return

    print("代码索引构建完成:")
    print(f"  db: {db_path}")
    print(f"  indexed: {result.indexed}")
    print(f"  rebuilt: {result.rebuilt}")
    print(f"  reused: {result.reused}")
    print(f"  skipped: {result.skipped}")
    print(f"  errors: {result.errors}")
    print(f"  FTS: {'可用' if result.fts_available else '降级'}")


async def _run_clear(config: CodeIntelConfig) -> None:
    db_path = code_intel_index_db_path(Path.cwd(), config)
    removed = await asyncio.to_thread(_clear_index_files, db_path)
    if removed == 0:
        print("代码索引清理: 当前 workspace 没有可删除的索引文件。")
        return
    print(f"代码索引清理完成: 删除 {removed} 个文件。")
    print(f"  db: {db_path}")


def _inspect_index_db(db_path: Path) -> _IndexStatus:
    if not db_path.exists():
        return _IndexStatus(exists=False, db_path=db_path)
    try:
        connection = sqlite3.connect(
            db_path.resolve(strict=False).as_uri() + "?mode=ro", uri=True
        )
    except sqlite3.Error as error:
        return _IndexStatus(
            exists=True, db_path=db_path, error=error.__class__.__name__
        )
    try:
        tables = _table_names(connection)
        if "files" not in tables or "symbols" not in tables:
            return _IndexStatus(
                exists=True, db_path=db_path, error="索引 schema 不完整"
            )
        file_count = _first_int(connection, "SELECT COUNT(*) FROM files")
        symbol_count = _first_int(connection, "SELECT COUNT(*) FROM symbols")
        last_indexed_at = _first_float(connection, "SELECT MAX(indexed_at) FROM files")
        return _IndexStatus(
            exists=True,
            db_path=db_path,
            file_count=file_count,
            symbol_count=symbol_count,
            last_indexed_at=last_indexed_at,
            fts_available="symbols_fts" in tables,
        )
    except sqlite3.Error as error:
        return _IndexStatus(
            exists=True, db_path=db_path, error=error.__class__.__name__
        )
    finally:
        connection.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = cast(
        list[tuple[object, ...]],
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall(),
    )
    return {str(row[0]) for row in rows}


def _first_value(connection: sqlite3.Connection, sql: str) -> object | None:
    row = cast(tuple[object, ...] | None, connection.execute(sql).fetchone())
    if row is None:
        return None
    return row[0]


def _first_int(connection: sqlite3.Connection, sql: str) -> int:
    value = _first_value(connection, sql)
    if value is None:
        return 0
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _first_float(connection: sqlite3.Connection, sql: str) -> float | None:
    value = _first_value(connection, sql)
    if value is None:
        return None
    return float(cast(float | int | str, value))


def _clear_index_files(db_path: Path) -> int:
    removed = 0
    for candidate in (
        db_path,
        Path(str(db_path) + "-wal"),
        Path(str(db_path) + "-shm"),
    ):
        try:
            if candidate.exists():
                candidate.unlink()
                removed += 1
        except OSError:
            continue
    return removed


__all__ = ["VALID_SUBS", "run"]
