"""Built-in tools shipped with the framework."""

from langchain_core.tools import tool


@tool
def read_file(path: str, lines: int = 100) -> str:
    """Read the contents of a file from the filesystem.

    Args:
        path: Absolute path to the file to read.
        lines: Maximum number of lines to read (default 100). Pass 0 or
               negative to read the entire file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            if lines > 0:
                content = "".join(f.readlines()[:lines])
            else:
                content = f.read()
        return content
    except FileNotFoundError:
        return f"[Error] File not found: {path}"
    except IsADirectoryError:
        return f"[Error] Path is a directory, not a file: {path}"
    except Exception as exc:
        return f"[Error] Failed to read {path}: {exc}"
