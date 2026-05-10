# Code Intelligence 新增语言支持

本文档说明在 `code_intel` 中接入新语言的固定流程。新增语言时先确定支持层级，再修改对应入口。不要把没有 provider 能力的语言加入 provider 级映射；否则工具会认为该语言可解析，但实际调用会在 grammar、query 或 LSP 阶段失败。

## 支持层级

`code_intel` 的语言能力分为三层，可以分步接入。

| 层级       | 能力                                                                   | 入口                                      | 适用场景                           |
| ---------- | ---------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------- |
| 工具层识别 | 根据扩展名推断语言，允许 text search fallback                          | `core/languages.py`、`TextSearchProvider` | 只需要搜索文本或让工具识别文件类型 |
| 语法层     | `code_outline`、`DOCUMENT_SYMBOLS`、symbol index、index-backed context | Tree-sitter query、`SymbolIndexer`        | 需要结构化符号和索引               |
| 语义层     | definition、references、hover、diagnostics                             | LSP registry、LSP config                  | 需要语言服务器提供语义能力         |

新增语言通常按这个顺序落地：工具层识别，语法层，语义层。每一层都需要独立测试。

## 语言 ID

语言 ID 使用小写稳定字符串，例如：

```text
python
typescript
javascript
go
rust
java
```

同一语言在扩展名映射、Tree-sitter、LSP registry、config 和测试中必须使用同一个 ID。不要混用 `go` / `golang`、`typescript` / `ts` 这类别名。

## 相关文件

| 文件                                                 | 用途                                 |
| ---------------------------------------------------- | ------------------------------------ |
| `src/code_intel/core/languages.py`                   | 扩展名到语言 ID 的映射               |
| `src/code_intel/providers/text_search/provider.py`   | text search 支持的语言集合           |
| `src/code_intel/providers/tree_sitter/queries/*.scm` | Tree-sitter 符号抽取 query           |
| `src/code_intel/providers/tree_sitter/parser.py`     | grammar/query 加载和 `Symbol` 归一化 |
| `src/code_intel/providers/lsp/registry.py`           | 默认语言服务器注册表                 |
| `src/code_intel/config.py`                           | 默认 LSP 语言列表和运行配置          |
| `tests/code_intel/contracts/test_languages.py`       | 语言映射契约测试                     |
| `tests/code_intel/providers/lsp/test_manager.py`     | LSP registry 和 manager 测试         |

## 工具层识别

文件：`src/code_intel/core/languages.py`

### 只支持工具层

如果该语言暂时只用于 text search 或工具 fallback，把扩展名加入 `TOOL_LANGUAGE_BY_EXTENSION`。

```python
TOOL_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    **LANGUAGE_BY_EXTENSION,
    ".swift": "swift",
}
```

这种方式不会扩大 `SUPPORTED_CODE_LANGUAGES`，Tree-sitter 和 LSP provider 不会把该语言当作已支持语言。

### 支持 provider 能力

如果该语言已经具备 Tree-sitter query 或 LSP provider 能力，把扩展名加入 `LANGUAGE_BY_EXTENSION`。

```python
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}
```

`SUPPORTED_CODE_LANGUAGES` 从 `LANGUAGE_BY_EXTENSION.values()` 自动派生。加入这里意味着 provider 层会认为该语言可被解析或路由。

### 契约测试

文件：`tests/code_intel/contracts/test_languages.py`

provider 级语言需要覆盖 `language_for_path(...)`：

```python
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/module.py", "python"),
        ("cmd/app/main.go", "go"),
    ],
)
def test_language_for_path_covers_supported_extensions_case_insensitively(
    path: str, expected: str
) -> None:
    assert language_for_path(path) == expected
```

只做工具层识别时，测试应确认不会扩大 provider 语言集合：

```python
def test_tool_language_mapping_covers_extra_tool_extensions_without_expanding_provider_languages() -> None:
    assert language_for_path("src/main.swift", TOOL_LANGUAGE_BY_EXTENSION) == "swift"
    assert "swift" not in SUPPORTED_CODE_LANGUAGES
```

## TextSearchProvider

文件：`src/code_intel/providers/text_search/provider.py`

`TextSearchProvider` 有独立的 `_SUPPORTED_LANGUAGES`。新增只用于 text search 的语言时，需要确认该集合包含对应语言 ID。

```python
_SUPPORTED_LANGUAGES = {
    "python",
    "typescript",
    "javascript",
    "go",
    "rust",
    "text",
}
```

如果语言不在集合中，kernel 语言路由时不会选择 text search provider，除非 provider 配置为支持通配语言。

## Tree-sitter 语法层

语法层用于 outline、document symbols、symbol index 和 index-backed context。

### 前置条件

确认 `tree-sitter-language-pack` 能加载目标语言：

```python
from tree_sitter_language_pack import get_language, get_parser

get_language("go")
get_parser("go")
```

如果目标语言不在 `tree-sitter-language-pack` 中，先处理依赖或加载策略，再接入 query。不要只添加扩展名和 `.scm` 文件。

### 新增 query

目录：`src/code_intel/providers/tree_sitter/queries/`

新增文件命名必须与语言 ID 一致：

```text
src/code_intel/providers/tree_sitter/queries/go.scm
```

示例：

```scheme
; Tree-sitter query for Go syntax outline symbols.

(function_declaration
  name: (identifier) @name) @function.definition

(method_declaration
  name: (field_identifier) @name) @method.definition

(import_declaration) @import.definition
```

query 中使用的 node 名称必须来自目标语言 grammar。不要从其它语言 query 直接复制后只改文件名。

### 支持的 capture

`TreeSitterParser` 当前识别这些 capture：

```text
class.definition
function.definition
method.definition
interface.definition
import.definition
export.definition
name
```

它们映射到 `SymbolKind`。如果目标语言需要新的 capture 类型，先确认 `SymbolKind`、序列化、索引和工具输出是否已有对应模型。新增 `SymbolKind` 属于模型变更，需要单独评估兼容性。

### 版本处理

文件：`src/code_intel/providers/tree_sitter/parser.py`

```python
TREE_SITTER_QUERY_VERSION = "tree-sitter-query-v1"
TREE_SITTER_INDEX_VERSION = f"tree-sitter:{TREE_SITTER_QUERY_VERSION}"
```

处理规则：

- 新增从未索引过的语言：通常不需要 bump query version。
- 修改已有语言的 query 输出：需要评估并通常 bump `TREE_SITTER_QUERY_VERSION`。
- 修改 `Symbol` 生成逻辑、稳定 ID 相关字段或索引 schema：按索引迁移规则处理，不只改 query version。

### 语法层测试

至少覆盖这些行为：

- parser 能加载目标语言 grammar 和 query。
- parser 能从真实源码 fixture 抽取 module 和至少一个业务符号。
- `TreeSitterProvider.document_symbols(...)` 返回 core `Symbol`，不暴露 tree-sitter 对象。
- `SymbolIndexer.index_file(...)` 能写入并查询该语言符号。

示例：

```python
def test_tree_sitter_extracts_go_symbols(tmp_path: Path) -> None:
    source = tmp_path / "main.go"
    source.write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

    parser = TreeSitterParser()
    symbols = parser.extract_symbols(tmp_path, "main.go")

    assert any(symbol.name == "main" for symbol in symbols)
```

## LSP 语义层

语义层通过语言服务器提供 definition、references、hover、diagnostics 和部分 document symbols。

### 注册语言服务器

文件：`src/code_intel/providers/lsp/registry.py`

新增 `LanguageServerSpec`：

```python
LanguageServerSpec(
    language="go",
    name="gopls",
    detect_command=["gopls", "version"],
    launch_command=["gopls"],
    install_hint="go install golang.org/x/tools/gopls@latest",
    root_markers=["go.work", "go.mod", ".git"],
)
```

字段要求：

| 字段             | 要求                                            |
| ---------------- | ----------------------------------------------- |
| `language`       | 与 `LANGUAGE_BY_EXTENSION` 中的语言 ID 完全一致 |
| `name`           | 语言服务器名称，用于提示和 trace                |
| `detect_command` | 快速退出的检测命令，不得启动常驻进程            |
| `launch_command` | stdio LSP server 启动命令                       |
| `install_hint`   | 用户可复制执行的安装命令                        |
| `root_markers`   | 用于识别项目根目录的文件名                      |
| `init_options`   | 需要传给 initialize 的初始化选项                |

### 默认启用语言

文件：`src/code_intel/config.py`

默认启用的 LSP 语言来自：

```python
_DEFAULT_LSP_LANGUAGES = ("python", "typescript", "javascript")
```

如果新语言应默认启用，把语言 ID 加入该 tuple。只注册 spec 但不加入默认列表时，用户仍可通过配置显式启用。

### 路由条件

LSP 能被选中需要同时满足：

1. `LANGUAGE_BY_EXTENSION` 能从路径推断语言。
2. `LSPManager` 注册了该语言的 `LanguageServerSpec`。
3. `CodeIntelLSPConfig.languages` 包含该语言。
4. `detect_command` 能成功执行。
5. server initialize 返回所需 capability。

`LSPProvider` 只暴露 server 协商到的能力。注册了语言服务器不等于 definition、references、hover、diagnostics 全部可用。

### LSP 测试

更新 `tests/code_intel/providers/lsp/test_manager.py`，覆盖默认 registry：

```python
def test_default_registry_specs_include_go() -> None:
    specs = default_language_server_specs()

    assert specs["go"].detect_command == ["gopls", "version"]
    assert specs["go"].launch_command == ["gopls"]
    assert specs["go"].install_hint == "go install golang.org/x/tools/gopls@latest"
```

还应覆盖：

- 缺失 executable 时返回中文安装提示。
- manager key 包含 workspace、language、launch command 和 init options hash。
- provider 能根据新语言文件路径选择对应 server。
- capabilities 映射符合 server initialize 返回值。

## 配置示例

用户配置位于 `~/.hpagent/config.json` 的 `code_intel` 段。示例：

```json
{
  "code_intel": {
    "lsp": {
      "enabled": true,
      "languages": ["python", "typescript", "javascript", "go"],
      "request_timeout_ms": 5000,
      "idle_shutdown_minutes": 10
    },
    "providers": {
      "tree_sitter": true,
      "text_search": true,
      "lsp": true
    }
  }
}
```

配置只控制运行时是否启用对应 provider。新增语言的源码映射、Tree-sitter query 和 LSP registry 仍需要在代码中接入。

## 验证命令

语言映射：

```bash
python3 -m pytest -q tests/code_intel/contracts/test_languages.py
```

Tree-sitter 和索引：

```bash
python3 -m pytest -q tests/code_intel/providers/tree_sitter
python3 -m pytest -q tests/code_intel/index
```

LSP registry 和 provider：

```bash
python3 -m pytest -q tests/code_intel/providers/lsp/test_manager.py
python3 -m pytest -q tests/code_intel/providers/lsp/test_provider.py
```

工具层：

```bash
python3 -m pytest -q tests/code_intel/tools
python3 -m pytest -q tests/code_intel/unit/test_kernel_routing.py
```

最终验证：

```bash
python3 -m pytest -q tests/code_intel
```

## 提交拆分

新增完整语言支持时，按能力边界拆分提交。

```text
1. 添加语言映射和契约测试
2. 添加 Tree-sitter query、fixture 和 parser/provider/index 测试
3. 注册 LSP server，并补充 manager/provider 测试
4. 补充工具层或集成测试
```

不要把扩展名、query、LSP registry 和工具测试混在一个大提交里。每个提交都应能说明一个独立能力边界。

## 常见错误

### 只加扩展名，不加 query

把语言加入 `LANGUAGE_BY_EXTENSION` 后，Tree-sitter provider 会认为该语言属于 provider 级支持。如果没有 `queries/<language>.scm`，outline/index 会在 query 加载阶段失败。

只需要工具识别时，把扩展名放进 `TOOL_LANGUAGE_BY_EXTENSION`。

### LSP registry 和 config 不一致

`LSPManager` 注册了 server spec，但 `config.lsp.languages` 没有该语言时，`LSPProvider` 不会路由该语言。

### query 没有真实源码测试

Tree-sitter query 写错 node 名称时常见表现是空结果，不一定抛异常。每个新语言都要有真实源码 fixture，至少验证一个函数、类或 import 被捕获。

### semantic 和 syntax 混淆

Tree-sitter 支持只代表 outline/index 可用，不代表 definition、references、hover、diagnostics 可用。语义能力必须由 LSP server 提供，并且取决于 initialize 返回的 capabilities。

### raw provider payload 泄漏

新增语言时不得把 raw LSP response、tree-sitter `Node`、源码全文或绝对路径放入 `ToolResult`、trace metadata 或工具 JSON 输出。跨边界数据必须归一化为 core models。

## 完成标准

新增语言达到对应层级后，需要满足以下条件：

- 语言映射契约测试通过。
- 新语言不会改变已有语言的映射和默认行为。
- provider 返回 core model，不返回 raw provider 对象。
- trace 中不包含源码、绝对路径、LSP 原始请求或响应。
- 缺失外部依赖时返回安全错误和安装提示。
- `python3 -m pytest -q tests/code_intel` 通过。
