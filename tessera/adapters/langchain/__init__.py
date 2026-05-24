"""LangChain bridge — call any LangChain Tool as a Tessera tool.

Usage in .t.md:

    ```tsr:tool
    tool web_search(query: String) -> String from langchain_community.tools.DuckDuckGoSearchRun
    ```

At runtime the interpreter resolves the dotted path via importlib, instantiates
if it's a class, and calls `.invoke(args)` (LangChain convention) or the
caller-specified `via <method>` override.

This file is intentionally tiny — the actual `Tool.Invoke` handler lives in
``tessera/interp/eval.py``. The LangChain pkg is NOT imported here; we only
resolve dotted paths at call time, so installing tessera does not require
langchain.
"""
from __future__ import annotations

import importlib
from typing import Any


def resolve_callable(import_path: str) -> Any:
    """Resolve a dotted path like `module.sub.Class` or `module.fn` to an object."""
    parts = import_path.split(".")
    # Try progressively shorter module prefixes
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_path = parts[i:]
        try:
            obj = importlib.import_module(module_path)
        except ImportError:
            continue
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ImportError(f"could not resolve {import_path!r}")


def _fallback_search(query: str) -> str:
    """Built-in callable for examples — no network access, deterministic output.

    Swap to `langchain_community.tools.DuckDuckGoSearchRun` (or any LangChain
    BaseTool) in the .t.md `tool` block once LangChain is installed.
    """
    return (
        f"[fallback search results for: {query}]\n"
        "Standard contractor payment flow in commercial construction: "
        "GC bills owner on progress, owner pays GC, GC retains 5-10%, "
        "GC pays subs on net-30 to net-90 terms, subs pay sub-subs after that. "
        "Retainage typically released at substantial completion."
    )


def invoke_tool(tool_obj: Any, args: list[Any], invoke_method: str = "invoke") -> Any:
    """Call a resolved tool object with a list of positional args.

    Handles three shapes:
      - class with .invoke / .run / __call__ → instantiate then call
      - bare callable → call directly
      - LangChain BaseTool → instantiate then .invoke(arg_dict_or_str)
    """
    # If it's a class, instantiate it (LangChain Tools are typically classes).
    if isinstance(tool_obj, type):
        tool_obj = tool_obj()

    method = getattr(tool_obj, invoke_method, None)
    if method is None:
        # bare callable
        if callable(tool_obj):
            return tool_obj(*args)
        raise TypeError(f"tool {tool_obj!r} has no {invoke_method!r} and is not callable")

    # LangChain Tools take either a string or a dict
    if len(args) == 1:
        return method(args[0])
    return method(args)
