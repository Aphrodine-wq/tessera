"""Shared parsing helpers used by adapters that want to inspect a ParsedModule
without committing to a full lower() pass.

Currently used by the Obsidian adapter's vault scan, where we want a fast,
non-failing summary of what each agent file declares — fail-soft so a single
broken file doesn't poison a whole vault listing.
"""
from __future__ import annotations

import re

from ..parser.module import ParsedModule

_AGENT_NAME_RE = re.compile(r"agent\s+(\w+)\s*\{")
_PROMPT_NAME_RE = re.compile(r"prompt\s+(\w+)\s*\(")
_TOOL_NAME_RE = re.compile(r"tool\s+(\w+)\s*\(")
_MODEL_NAME_RE = re.compile(r"model\s+(\w+)\s*\{")


def sniff_block_features(pm: ParsedModule) -> dict:
    """Return a lightweight feature summary of the parsed module.

    Keys:
      - substrates: list of substrate strings used (deduped, sorted)
      - agent_names: list of agent declarations found in agent blocks
      - prompts / tools / neural_models: list of declared names
    """
    substrates: set[str] = set()
    agent_names: list[str] = []
    prompts: list[str] = []
    tools: list[str] = []
    neural_models: list[str] = []

    for block in pm.blocks:
        substrates.add(block.substrate)
        if block.substrate == "agent":
            agent_names.extend(_AGENT_NAME_RE.findall(block.body))
        elif block.substrate == "prompt":
            prompts.extend(_PROMPT_NAME_RE.findall(block.body))
        elif block.substrate == "tool":
            tools.extend(_TOOL_NAME_RE.findall(block.body))
        elif block.substrate == "neural":
            neural_models.extend(_MODEL_NAME_RE.findall(block.body))

    return {
        "substrates": sorted(substrates),
        "agent_names": agent_names,
        "prompts": prompts,
        "tools": tools,
        "neural_models": neural_models,
    }
