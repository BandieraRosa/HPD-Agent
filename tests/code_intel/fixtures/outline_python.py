import os
from pathlib import Path


class Greeter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix}, {name}"


def build_greeter(prefix: str) -> Greeter:
    return Greeter(prefix)
