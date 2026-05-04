# Code Intelligence module documentation

## Goal and implementation boundaries

`code_intel` is a sidecar code-intelligence subsystem for HPD-Agent. Its job is narrow: help the Agent understand code through structured symbols, semantic queries, and diagnostics, then catch new problems after patches.

The current implementation does not include embeddings, autocomplete, semantic tokens, an MCP bridge, a Web UI, or automatic LSP server installation. `Capability.RENAME` exists only as a reserved core enum value. There is no rename tool, provider route, or workspace rename flow.

LSP servers are installed by the user or the runtime environment. The system detects them and prints hints, but it does not download binaries or modify the global environment.

## Architecture overview

This architecture section is distilled from `new_feature_architecture.md` and cross-checked against the implemented `src/code_intel/` code; the core design is "Kernel and contracts first, providers second":

```text
HPD Agent workflow
  agent tools from src/tools/__init__.py
  PatchGate and edit policy from src/code_intel/workflow/
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

Dependencies flow in one direction:

```text
tools/ -> kernel.py -> providers/ -> core/
                  |-> index/    -> core/
                  |-> verifier/ -> core/
workflow/ -> tools/ and existing HPD workflow modules
```

`core/` is the data and interface boundary. It does not depend on providers, index, verifier, workflow, or tools. `workflow/` is the only `code_intel` layer allowed to touch existing agent/node modules.

## Data boundary

Data crossing module boundaries uses Pydantic models. Provider internals do not leak raw dicts, raw LSP responses, or tree-sitter Node objects. The main contracts live in `src/code_intel/core/`:

- `models.py` defines `Range`, `Location`, `Symbol`, `Diagnostic`, `ToolError`, `ToolMeta`, `ToolResult`, `CodeContext`, and `HoverInfo`.
- `capabilities.py` defines `Capability`, the `Provider` protocol, provider health, confidence class, and capability-specific protocols.
- `errors.py` defines `CodeIntelError`, `LanguageNotSupported`, `ProviderUnavailable`, `IndexStale`, `SymbolNotFound`, and `LSPTimeout`, then maps them to safe Chinese `ToolError` values.
- `anchors.py` defines `TextAnchor`, `CodeTarget`, and `TargetResolver`. Resolution priority is `symbol_id`, then `anchor`, then `location`.
- `workspace.py` stores workspace and language-detection contracts only. It does not scan the filesystem.

Paths must be workspace-relative and use forward slashes. Absolute paths, backslashes, empty segments, `.`, `..`, and Windows drive-relative paths are rejected.

## CodeIntelKernel

`CodeIntelKernel` lives in `src/code_intel/kernel.py`. The default kernel has no providers. Runtime code injects providers explicitly through `set_code_intel_kernel()`.

The kernel does this for each call:

1. Accepts a `Capability` and language.
2. Calls each provider's `supports()`, `health()`, and `confidence_for()`.
3. Sorts candidates by confidence and health.
4. Calls the method for that capability.
5. Falls back after `ProviderUnavailable` or `LSPTimeout`.
6. Returns a `ToolResult` and stores `last_trace`.

When a `SymbolIndexStore` is configured, the kernel handles some calls through the index. `DOCUMENT_SYMBOLS` can read current file symbols from SQLite, and `CONTEXT_EXTRACT` can use `IndexBackedCodeContext`. Without an index, those capabilities use normal provider routing.

## Providers

### TreeSitterProvider

`TreeSitterProvider` lives in `src/code_intel/providers/tree_sitter/`. It supports `outline` and `document_symbols` for Python, TypeScript, and JavaScript. It loads grammars and parsers through `tree-sitter-language-pack`, then uses `.scm` queries to extract module, import, export, class, interface, function, and method symbols.

Synchronous file reads and parsing run inside `asyncio.to_thread()`, so the public provider methods stay async. Results are core `Symbol` models with `source="tree_sitter"`; tree-sitter internals do not cross the provider boundary.

### TextSearchProvider

`TextSearchProvider` lives in `src/code_intel/providers/text_search/`. It is a pure Python workspace text-search provider. It supports literal and regex search and returns core `Location` values.

It reads the root `.gitignore` and skips `.git`, symlinks, binary files, oversized files, paths outside the workspace, and ignored paths. Invalid regex input becomes an `invalid_regex` error instead of a raw traceback.

### LSPProvider

`LSPProvider` lives in `src/code_intel/providers/lsp/`. `LSPManager` lazy-starts language servers, `LSPTransport` handles stdio with `Content-Length` framing, and `LSPClient` converts LSP types to core models.

Default registry:

| language     | server                       | install hint                                     |
| ------------ | ---------------------------- | ------------------------------------------------ |
| `python`     | `pyright`                    | `npm i -g pyright`                               |
| `typescript` | `typescript-language-server` | `npm i -g typescript-language-server typescript` |
| `javascript` | `typescript-language-server` | `npm i -g typescript-language-server typescript` |

`LSPManager` runs the full `detect_command` before startup. Detection failure returns an install hint and does not install anything. `LSPProvider` exposes `definition`, `references`, `hover`, `document_symbols`, and `diagnostics` only when the initialized server reports those capabilities. One language server failure marks that language unhealthy and should not stop other languages or the main process.

Current LSP semantic calls need an explicit `CodeTarget.location`. The index can resolve `symbol_id` and `TextAnchor`, but not every LSP provider call automatically resolves first.

### Fake providers

`FakeSyntaxProvider`, `FakeSemanticProvider`, and `create_fake_providers()` live in `src/code_intel/providers/fake/`. They are deterministic providers for tests and demos that opt in explicitly. They are not production semantic intelligence and should not be described as real LSP, index quality, or model reasoning.

## SQLite symbol index

The index implementation lives in `src/code_intel/index/`.

- `SymbolIndexStore` uses `aiosqlite` and initializes WAL, `synchronous=NORMAL`, and foreign keys.
- The default index path is `~/.hpagent/index/<workspace_sha256>/symbols.db`; config can change the cache root.
- The `files` table stores workspace-relative path, language, SHA-256, mtime, size, grammar/query/schema version, and indexed timestamp.
- The `symbols` table stores decomposed `Symbol` fields.
- `symbol_id_history` preserves symbol ID history for `(language, path, qualified_name, file_hash)` recovery after edits.
- FTS5 is used for symbol search when available. Otherwise search falls back to bounded LIKE.
- `SymbolIndexer` accepts an injected extractor and does not directly depend on `TreeSitterProvider`. It skips secret-like files, symlinks, binary files, oversized files, ignored files, and unsupported languages.
- `IndexBackedTargetResolver` resolves targets in `symbol_id -> TextAnchor -> Location` order. `IndexBackedCodeContext` extracts signature, body, parents, imports, and nearby symbols from indexed symbols plus source text, then applies a token budget.

## Verifier and PatchGate

Verification has two layers.

`src/code_intel/verifier/` is provider-free logic:

- `diagnostics_delta.py` keeps isolated agent and workflow baseline caches, then computes new, resolved, and unchanged diagnostics.
- Delta matching does not rely only on raw line numbers or full messages. It normalizes messages and prefers an enclosing-symbol semantic anchor.
- `repair_policy.py` returns `proceed`, `repair`, or `retreat`. The default repair limit is 2 rounds. New errors block; warnings are partial and non-blocking by default.

`PatchGate` lives in `src/code_intel/workflow/patch_gate.py`. It observes apply_patch tool logs, extracts changed files, and can run configured index invalidation, LSP `didChange` notifications, and changed-file diagnostics. `PatchGate` does not modify `src/tools/apply_patch.py`, and it does not replace apply_patch parsing, writing, atomic rollback, or safety checks.

Current behavior:

- `src/tools/apply_patch.py` is unchanged. Patch verification lives in `src/code_intel/workflow/`.
- `PatchGateConfig.verify_changed`, `invalidate_syntax`, `invalidate_index`, and `notify_lsp_did_change` default to `False`; callers must enable them explicitly.
- PatchGate handles changed files only. It does not start workspace-wide verification.
- PatchGate does not run tests or lint automatically.
- The agent-facing `code_verify` currently runs LSP diagnostics only for explicit `paths`. `scope="changed"` and `scope="workspace"` are accepted as parameters, but automatic path collection is reported as skipped.

## Five Agent tools

The five tools live in `src/code_intel/tools/` and are added to `tool_list` by `src/tools/__init__.py`: `code_search`, `code_outline`, `code_context`, `code_semantic`, and `code_verify`.

| Tool            | Current use                                 | Implementation notes                                                                                                                                   |
| --------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `code_search`   | Search by symbol name or text               | Supports `mode="symbol"`, `mode="text"`, and `mode="mixed"`. Symbol search uses `Capability.SYMBOL_SEARCH`; text search uses `Capability.TEXT_SEARCH`. |
| `code_outline`  | Return structured symbols for one file      | Tries `OUTLINE` first, then `DOCUMENT_SYMBOLS`. Trims nested results by `max_depth`.                                                                   |
| `code_context`  | Extract pre-edit context for a `CodeTarget` | Defaults to `signature`, `body`, and `imports`. Can also request `parents` and `nearby_symbols`.                                                       |
| `code_semantic` | Run semantic queries                        | Supports `definition`, `references`, `hover`, and `document_symbols`. References are grouped by file.                                                  |
| `code_verify`   | Verify diagnostics after edits              | Currently runs `lsp_diagnostics` only for explicit `paths`. `tests` and `lint` are recorded as skipped and are not executed inside the tool.           |

All five tools return a JSON-string `ToolResult`. Chinese errors stay readable for the Agent, while machine fields keep English codes.

## Configuration

The config file is `~/.hpagent/config.json`. Missing config, or a missing `code_intel` section, uses defaults. Malformed JSON or an invalid `code_intel` object prints a Chinese error and hint in REPL command paths.

Typical shape:

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

## `/index` command

`/index` manages the SQLite symbol index for the current workspace.

| Command         | Behavior                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `/index status` | Read-only status for DB path, file count, symbol count, last indexed timestamp, and FTS state. It does not rebuild automatically. |
| `/index build`  | Builds or updates the current workspace index through `TreeSitterProvider` and `SymbolIndexer`.                                   |
| `/index clear`  | Deletes current workspace sidecars: `symbols.db`, `symbols.db-wal`, and `symbols.db-shm`.                                         |

## `/lsp` command

`/lsp` inspects or manages an already bound LSP runtime. `/lsp status` does not start servers and does not install servers.

| Command                   | Behavior                                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `/lsp status`             | Shows configured languages, server names, running state, and negotiated capabilities. If no runtime is bound, it reports that state only. |
| `/lsp stop [language]`    | Stops the bound runtime. Pass a language to stop one language, or omit it to stop all languages.                                          |
| `/lsp restart <language>` | Restarts one language only when a runtime is already bound. Without a runtime, it does not start a real server.                           |

When an LSP server is missing, the user-facing hint should include the install command, for example: `未找到 pyright。请先运行 npm i -g pyright，然后重新执行 /lsp status 或相关语义查询。` Do not say HPD-Agent installs it for the user.

## Tracing and redaction

`src/code_intel/tracing.py` reuses the project observability tracer and applies an allowlist to metadata. It can record provider name, capability, language, workspace-relative paths, fallback chain, result count, elapsed time, truncation state, and safe error metadata.

Tracing does not record source bodies, hover contents, sensitive diagnostic values, absolute temp paths, raw LSP payloads, or tree-sitter objects. Diagnostic messages are normalized into templates before they enter trace metadata.
