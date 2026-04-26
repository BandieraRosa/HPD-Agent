"""Write tool: create or append content to a file."""

from langchain_core.tools import tool


@tool
def write_file(filename: str, content: str, append: bool = True) -> str:
    """Write content to a file on the filesystem.

    Args:
        filename: Absolute path of the file to write.
        content: The text content to write.
        append: If True (default), append to existing file. If False, overwrite.
    """
    try:
        mode = "a" if append else "w"
        with open(filename, mode, encoding="utf-8") as f:
            f.write(content)
        action = "appended to" if append else "written to"
        return f"[OK] Content {action} {filename}"
    except IsADirectoryError:
        return f"[Error] Path is a directory, not a file: {filename}"
    except PermissionError:
        return f"[Error] Permission denied: {filename}"
    except FileNotFoundError:
        return f"[Error] Directory does not exist: {filename}"
    except Exception as exc:
        return f"[Error] Failed to write {filename}: {exc}"
