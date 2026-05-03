"""Verify Code Intelligence baseline dependencies are importable."""


def test_baseline_dependencies_importable():
    """基线依赖必须真实可导入，而不是只存在于 pyproject 声明中。"""
    from tree_sitter import Language, Parser, Query
    from tree_sitter_language_pack import get_language, get_parser
    import aiosqlite
    import networkx as nx
    from lsprotocol import converters, types
    from pathspec import GitIgnoreSpec, PathSpec

    assert Language is not None
    assert Parser is not None
    assert Query is not None
    assert get_language is not None
    assert get_parser is not None
    assert aiosqlite is not None
    assert nx is not None
    assert converters is not None
    assert types is not None
    assert GitIgnoreSpec is not None
    assert PathSpec is not None
