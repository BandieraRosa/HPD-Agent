# Code Intelligence 模块文档

## 目标和实现边界

`code_intel` 是 HPD-Agent 旁挂的代码智能子系统。目标很具体：用结构化符号、语义查询和诊断结果帮助 Agent 理解代码、定位修改点，并在补丁后发现新增问题。

当前实现不包含这些能力：embeddings、autocomplete、语义 tokens、MCP bridge、Web UI、自动安装 LSP servers。`Capability.RENAME` 只是在核心枚举中保留的能力名，当前没有 rename 工具、provider 路由或批量改名流程。

LSP server 由用户或运行环境安装。系统只检测和提示，不下载二进制，也不替用户修改全局环境。

## 架构总览

本节架构内容从 `new_feature_architecture.md` 提炼，并与 `src/code_intel/` 的实际实现交叉核对；核心设计是「先 Kernel 和契约，再接 provider」：

```text
HPD Agent workflow
  src/tools/__init__.py 中的 agent tools
  src/code_intel/workflow/ 中的 PatchGate 和 edit policy
        |
        v
CodeIntelKernel
  capability routing, provider registry, optional symbol index
        |
        +-- providers/tree_sitter/  TreeSitterProvider
        +-- providers/text_search/  TextSearchProvider
        +-- providers/lsp/          LSPProvider
        +-- providers/fake/         FakeSyntaxProvider, FakeSemanticProvider
        |
        +-- index/      SQLite symbol index, target resolver, context extractor
        +-- verifier/   diagnostics delta, baseline cache, repair policy
        +-- tracing.py  safe trace metadata and redaction
```

依赖方向保持单向：

```text
tools/ -> kernel.py -> providers/ -> core/
                  |-> index/    -> core/
                  |-> verifier/ -> core/
workflow/ -> tools/ and existing HPD workflow modules
```

`core/` 是数据和接口边界，不依赖 provider、index、verifier、workflow 或 tools。`workflow/` 是唯一允许接触现有 agent/node 模块的 `code_intel` 层。

## 数据边界

跨模块传递的数据使用 Pydantic 模型，不把裸 dict、raw LSP response 或 tree-sitter Node 暴露到 provider 外面。主要模型在 `src/code_intel/core/`：

- `models.py` 定义 `Range`、`Location`、`Symbol`、`Diagnostic`、`ToolError`、`ToolMeta`、`ToolResult`、`CodeContext`、`HoverInfo`。
- `capabilities.py` 定义 `Capability`、`Provider` protocol、provider health、confidence class 和各能力的 protocol。
- `errors.py` 定义 `CodeIntelError`、`LanguageNotSupported`、`ProviderUnavailable`、`IndexStale`、`SymbolNotFound`、`LSPTimeout`，并映射为中文安全的 `ToolError`。
- `anchors.py` 定义 `TextAnchor`、`CodeTarget`、`TargetResolver`。定位优先级是 `symbol_id`，然后 `anchor`，最后 `location`。
- `workspace.py` 只保存 workspace 和语言识别契约，不扫描文件系统。

路径都要求是工作区相对路径，使用正斜杠，拒绝绝对路径、反斜杠、空片段、`.`、`..` 和 Windows drive-relative 形式。

## CodeIntelKernel

`CodeIntelKernel` 在 `src/code_intel/kernel.py`。默认内核不注册 provider，工具运行时通过 `set_code_intel_kernel()` 显式注入。它的主要职责是：

1. 接收 `Capability` 和 language。
2. 调用 provider 的 `supports()`、`health()` 和 `confidence_for()`。
3. 按 confidence 和 health 排序。
4. 调用对应能力方法。
5. 遇到 `ProviderUnavailable` 或 `LSPTimeout` 时尝试下一个 provider。
6. 返回统一的 `ToolResult`，并记录 `last_trace`。

如果配置了 `SymbolIndexStore`，Kernel 还会走索引路径处理部分能力：`DOCUMENT_SYMBOLS` 可以从 SQLite 取当前文件符号，`CONTEXT_EXTRACT` 可以通过 `IndexBackedCodeContext` 提取上下文。没有索引时，这些能力仍按普通 provider 路由。

## Providers

### TreeSitterProvider

`TreeSitterProvider` 位于 `src/code_intel/providers/tree_sitter/`。它支持 Python、TypeScript、JavaScript 的 `outline` 和 `document_symbols`。实现通过 `tree-sitter-language-pack` 加载 grammar 和 parser，使用 `.scm` query 抽取 module、import、export、class、interface、function、method 等符号。

同步的文件读取和 parse 被包在 `asyncio.to_thread()` 中，provider 对外仍是 async。返回值是 core `Symbol`，`source="tree_sitter"`，不返回 tree-sitter 内部对象。

### TextSearchProvider

`TextSearchProvider` 位于 `src/code_intel/providers/text_search/`。它是纯 Python workspace 文本搜索 provider，支持 literal 和 regex 搜索，返回 core `Location` 列表。

它会读取根目录 `.gitignore`，跳过 `.git`、symlink、二进制文件、超限文件和工作区外路径。正则无效时返回 `invalid_regex` 类型错误，不抛原始 traceback 给工具层。

### LSPProvider

`LSPProvider` 位于 `src/code_intel/providers/lsp/`。它通过 `LSPManager` 懒启动语言服务器，通过自写 `LSPTransport` 使用 stdio 和 `Content-Length` framing 通信，通过 `LSPClient` 把 LSP 类型转成 core 模型。

默认 registry 包含：

| language     | server                       | install hint                                     |
| ------------ | ---------------------------- | ------------------------------------------------ |
| `python`     | `pyright`                    | `npm i -g pyright`                               |
| `typescript` | `typescript-language-server` | `npm i -g typescript-language-server typescript` |
| `javascript` | `typescript-language-server` | `npm i -g typescript-language-server typescript` |

`LSPManager` 在启动前运行完整 `detect_command`。检测失败只返回安装提示，不自动安装。`LSPProvider` 只根据 server initialize 后协商到的 capability 暴露 `definition`、`references`、`hover`、`document_symbols`、`diagnostics`。单个语言 server 失败会标记该语言 unhealthy，不应影响其它语言或主进程。

当前 LSP 语义查询需要明确的 `CodeTarget.location`。索引可以解析 `symbol_id` 和 `TextAnchor`，但不是每条 LSP provider 调用都会自动先做解析。

### Fake providers

`FakeSyntaxProvider`、`FakeSemanticProvider` 和 `create_fake_providers()` 位于 `src/code_intel/providers/fake/`。它们是确定性的测试和演示 provider，需要显式注入。它们不是生产语义智能，也不代表真实 LSP、索引质量或模型推理能力。

## SQLite symbol index

索引实现位于 `src/code_intel/index/`。

- `SymbolIndexStore` 使用 `aiosqlite`，初始化 WAL、`synchronous=NORMAL` 和 foreign keys。
- 索引默认路径是 `~/.hpagent/index/<workspace_sha256>/symbols.db`，也可通过 config 的 `cache_dir` 计算。
- `files` 表保存 workspace 相对路径、language、SHA-256、mtime、size、grammar/query/schema version 和 indexed timestamp。
- `symbols` 表保存拆开的 `Symbol` 字段。
- `symbol_id_history` 保留历史 symbol id 到 `(language, path, qualified_name, file_hash)` 的映射，用于文件改动后的恢复。
- FTS5 可用时用于 symbol 搜索，不可用时降级到 bounded LIKE。
- `SymbolIndexer` 注入 extractor，不直接依赖 `TreeSitterProvider`。它扫描安全候选文件，跳过 secret-like 文件、symlink、binary、oversized、ignored 和 unsupported files。
- `IndexBackedTargetResolver` 按 `symbol_id -> TextAnchor -> Location` 解析目标。`IndexBackedCodeContext` 基于索引和源码提取签名、body、parents、imports、nearby symbols，并按预算截断。

## Verifier 和 PatchGate

验证实现分两层。

`src/code_intel/verifier/` 是 provider-free 逻辑：

- `diagnostics_delta.py` 维护 agent/workflow 隔离的 baseline cache，计算新增、解决、未变化三类 diagnostics。
- delta 比较不直接依赖行号和完整 message。它会规范化 message，并优先用 enclosing symbol 作为 semantic anchor。
- `repair_policy.py` 根据 delta 给出 `proceed`、`repair` 或 `retreat`，默认最多 2 轮 repair。新 error 会阻塞；warning 默认是 partial，不阻塞。

`PatchGate` 位于 `src/code_intel/workflow/patch_gate.py`。它观察 apply_patch 工具日志，提取 changed files，并按配置执行索引失效、LSP `didChange` 通知和 changed-file diagnostics。`PatchGate` 不改 `src/tools/apply_patch.py`，也不替代 apply_patch 的解析、写入、原子回滚或安全边界。

当前实现要点：

- `src/tools/apply_patch.py` 保持原样，patch verification 逻辑在 `src/code_intel/workflow/`。
- `PatchGateConfig.verify_changed`、`invalidate_syntax`、`invalidate_index`、`notify_lsp_did_change` 默认都是 `False`，需要上层显式打开。
- PatchGate 只处理 changed files，不自动启动 workspace-wide verify。
- PatchGate 不自动运行 tests 或 lint。
- `code_verify` 的 agent 入口当前只对显式 `paths` 运行 LSP diagnostics。`scope="changed"` 和 `scope="workspace"` 可以作为参数传入，但自动收集路径会被记录为 skipped。

## 五个 Agent 工具

五个工具定义在 `src/code_intel/tools/`，并由 `src/tools/__init__.py` 加入 `tool_list`：`code_search`、`code_outline`、`code_context`、`code_semantic`、`code_verify`。

| 工具            | 当前用途                         | 实现说明                                                                                                                             |
| --------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `code_search`   | 按 symbol 名或文本搜索           | 支持 `mode="symbol"`、`mode="text"`、`mode="mixed"`。symbol 搜索走 `Capability.SYMBOL_SEARCH`，文本搜索走 `Capability.TEXT_SEARCH`。 |
| `code_outline`  | 返回单文件结构化 symbol 列表     | 先请求 `OUTLINE`，失败后尝试 `DOCUMENT_SYMBOLS`。按 `max_depth` 裁剪嵌套结果。                                                       |
| `code_context`  | 按 `CodeTarget` 提取修改前上下文 | 默认提取 `signature`、`body`、`imports`。可请求 `parents` 和 `nearby_symbols`。                                                      |
| `code_semantic` | LSP 语义查询入口                 | 支持 `definition`、`references`、`hover`、`document_symbols`。`references` 会按文件分组。                                            |
| `code_verify`   | 修改后的诊断验证入口             | 当前只对显式 `paths` 跑 `lsp_diagnostics`。`tests` 和 `lint` 会记录为 skipped，不会在工具内执行。                                    |

所有工具返回 JSON 字符串形式的 `ToolResult`，中文错误对 Agent 可读，机器字段保持英文 code。

## 配置

配置文件是 `~/.hpagent/config.json`。没有该文件，或没有 `code_intel` 段时使用默认值。配置 JSON 损坏或 `code_intel` 段非法时，REPL 命令会打印中文错误和修复提示。

常用结构：

```json
{
  "code_intel": {
    "enabled": true,
    "locale": "zh-CN",
    "cache_dir": "~/.hpagent/index",
    "index": {
      "auto_build_on_startup": true,
      "respect_gitignore": true,
      "max_file_size_bytes": 1000000
    },
    "lsp": {
      "enabled": true,
      "languages": ["python", "typescript", "javascript"],
      "request_timeout_ms": 5000,
      "idle_shutdown_minutes": 10,
      "max_restart_count": 3
    },
    "providers": {
      "tree_sitter": true,
      "text_search": true,
      "lsp": true
    },
    "verify": {
      "auto_verify_on_patch": true,
      "max_repair_rounds": 2,
      "retreat_on_max_rounds": true
    },
    "tools": {
      "code_search_default_limit": 20,
      "code_context_default_max_tokens": 4000,
      "code_semantic_default_max_results": 50
    }
  }
}
```

## `/index` 命令

`/index` 管理当前 workspace 的 SQLite symbol index。

| 命令            | 行为                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| `/index status` | 只读查看 DB 路径、文件数、符号数、最近索引时间和 FTS 状态。不会自动重建。    |
| `/index build`  | 使用 `TreeSitterProvider` 和 `SymbolIndexer` 建立或更新当前 workspace 索引。 |
| `/index clear`  | 删除当前 workspace 的 `symbols.db`、`symbols.db-wal`、`symbols.db-shm`。     |

## `/lsp` 命令

`/lsp` 只查看或管理已经绑定的 LSP runtime。`/lsp status` 不启动 server，也不安装 server。

| 命令                      | 行为                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------- |
| `/lsp status`             | 显示配置语言、server 名称、运行状态和已协商 capability。没有 runtime 时只报告未绑定。 |
| `/lsp stop [language]`    | 停止已绑定 runtime。可以指定语言，不指定则停止全部语言。                              |
| `/lsp restart <language>` | 只重启已绑定 runtime 中的指定语言。没有 runtime 时不会启动真实 server。               |

缺少 LSP server 时，提示应给出安装命令，例如：`未找到 pyright。请先运行 npm i -g pyright，然后重新执行 /lsp status 或相关语义查询。` 不要写成 HPD-Agent 会替用户安装。

## Tracing 和 redaction

`src/code_intel/tracing.py` 复用项目 observability tracer，并对 metadata 做白名单过滤。允许记录 provider 名、capability、language、workspace 相对路径、fallback chain、result count、耗时、截断状态和安全错误信息。

Tracing 不记录源码正文、hover 内容、diagnostic 原文中的敏感值、绝对临时路径、raw LSP payload 或 tree-sitter 对象。diagnostic message 会被规范化成模板后再进入 trace。
