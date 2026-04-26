"""Quick demo: ask the agent to read a file using the read_file tool."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.models import get_store
get_store().switch("qwen-plus")

from src.llm import invoke_with_tools
from src.tools import read_file


async def main():
    prompt = (
        "请读取 /root/projects/evo_agent/README.md 的前 10 行内容，"
        "然后用中文总结文件开头写的是什么。"
    )

    print("=== Tool Use Demo ===\n")
    result, tool_log = await invoke_with_tools(
        prompt,
        tools=[read_file],
    )
    print("--- Final Response ---")
    print(result)
    if tool_log:
        print(f"\n--- Tool Results ({len(tool_log)} chars) ---")
        print(tool_log[:500])


if __name__ == "__main__":
    asyncio.run(main())
