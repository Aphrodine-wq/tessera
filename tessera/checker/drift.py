"""Tessera ↔ live-engine drift checker.

Parses a `.t.md` agent spec, extracts each agent's declared capability set
(from frontmatter `capabilities_requested` and any `spawn X with [...]`
clauses), then scans the corresponding live engine implementation for code
patterns that would require a capability the spec doesn't grant.

The mapping from agent name → engine file is convention-based: an agent
named `Foo` is looked up first at `~/.claude/<foo>/engine.js`, then at a
small registry of known overrides (immune system uses `healer.js`, etc.).

Heuristic, not formal: greps for fs/subprocess/network primitives and
infers capabilities. False positives are possible; false negatives mean
"keep your spec honest."
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

HOME = Path.home()
CLAUDE_DIR = HOME / ".claude"
VAULT_DIR = HOME / "Desktop" / "TheVault"


# ----- spec parsing ---------------------------------------------------------

_FRONTMATTER_CAPS_RE = re.compile(
    r"^capabilities_requested:\s*\[([^\]]*)\]", re.MULTILINE
)
_SPAWN_RE = re.compile(
    r"spawn\s+(\w+)\s+with\s*\[([^\]]*)\]"
)
_AGENT_BLOCK_RE = re.compile(
    r"^agent\s+(\w+)\s*\{", re.MULTILINE
)


@dataclass
class AgentSpec:
    name: str
    declared_caps: set[str] = field(default_factory=set)
    source_file: Path | None = None


def parse_tsr(path: Path) -> list[AgentSpec]:
    text = path.read_text(encoding="utf-8")
    file_caps = _extract_caps(_FRONTMATTER_CAPS_RE.search(text))

    # Each agent block gets the frontmatter caps as a floor plus any spawn-
    # supplied caps where it is the *spawner*. The grant is the union: an
    # agent named in a `spawn X with [...]` line is being given those caps.
    spawn_grants: dict[str, set[str]] = {}
    for m in _SPAWN_RE.finditer(text):
        agent_name = m.group(1)
        caps = _extract_caps(re.match(r"(.*)", m.group(2)))
        spawn_grants.setdefault(agent_name, set()).update(caps)

    specs: list[AgentSpec] = []
    for m in _AGENT_BLOCK_RE.finditer(text):
        name = m.group(1)
        granted = set(file_caps) | spawn_grants.get(name, set())
        specs.append(AgentSpec(name=name, declared_caps=granted, source_file=path))
    return specs


def _extract_caps(match) -> set[str]:
    if not match:
        return set()
    raw = match.group(1) if match.lastindex else ""
    return {c.strip() for c in raw.split(",") if c.strip()}


# ----- engine resolution ----------------------------------------------------

# Agent name → engine path (relative to ~/.claude/). When the convention
# `~/.claude/<lowercased-name>/engine.js` doesn't match, override here.
ENGINE_OVERRIDES: dict[str, str] = {
    "ImmuneSystem": "immune/healer.js",
    "UnifiedMemory": "memory-layer/engine.js",
    "DeepReasoning": "reasoning/engine.js",
    "ProjectPlanner": "planner/engine.js",
    "TeachingEngine": "teaching/engine.js",
    "CodeReview": "codereview/engine.js",
    "HabitTracker": "habits/engine.js",
    "DecisionJournal": "decisions/engine.js",
    "CompetitiveMonitor": "competitive/engine.js",
    "InnovationEngine": "innovation/engine.js",
    "ReputationEngine": "reputation/engine.js",
    "RelationshipManager": "relationships/engine.js",
    "FinancialAwareness": "financial/engine.js",
    "SimulationEngine": "simulation/engine.js",
    "EmotionalBaseline": "emotional/engine.js",
    "KnowledgeAcquisition": "knowledge/engine.js",
    "MetaCognition": "metacognition/engine.js",
    "ProactiveAgent": "proactive/engine.js",
    "GoalsEngine": "goals/engine.js",
    "Orchestrator": "orchestrator/engine.js",
    "ClaudeEyes": None,  # lives in ~/Projects/walt/eyes/, not ~/.claude/
    "ConstructionAIMonitor": None,  # ~/Projects/constructionai/scripts/monitor.py
    "WALTSystem": None,  # umbrella, no single file
    "TwinBrainSystem": None,  # cross-machine, no single file
}


def resolve_engine(name: str) -> Path | None:
    override = ENGINE_OVERRIDES.get(name)
    if override is None and name in ENGINE_OVERRIDES:
        return None  # explicit "no engine"
    if override:
        path = CLAUDE_DIR / override
        return path if path.exists() else None
    # Convention: ~/.claude/<lowercase>/engine.js
    guess = CLAUDE_DIR / name.lower() / "engine.js"
    if guess.exists():
        return guess
    return None


# ----- capability detection in JS ------------------------------------------

# Maps capability label → list of regex patterns that, if found, imply the
# implementation needs that capability.
CAP_PATTERNS: dict[str, list[re.Pattern]] = {
    "FileSystem": [
        re.compile(r"\bfs\.\s*(writeFile|writeFileSync|unlink|mkdir|rmdir|rm|appendFile)"),
        re.compile(r"\brequire\(['\"]fs['\"]\)"),
        re.compile(r"\bfrom\s+['\"]fs['\"]"),
    ],
    "Subprocess": [
        re.compile(r"\bchild_process"),
        re.compile(r"\bspawn\s*\("),
        re.compile(r"\bexecSync\b"),
        re.compile(r"\bexec\s*\("),
    ],
    "NetworkOut": [
        re.compile(r"\bfetch\s*\("),
        re.compile(r"\baxios\."),
        re.compile(r"\bhttps?\.\s*request"),
        re.compile(r"\brequire\(['\"]https?['\"]\)"),
    ],
    "VaultWrite": [
        re.compile(r"TheVault[^'\"]*['\"][^)]*\bfs\.\s*write"),
        re.compile(r"writeFile\w*\s*\(\s*[^)]*TheVault"),
    ],
    "VaultRead": [
        re.compile(r"readFile\w*\s*\(\s*[^)]*TheVault"),
        re.compile(r"TheVault[^'\"]*['\"][^)]*\bfs\.\s*read"),
    ],
}


def detect_used_caps(engine_path: Path) -> set[str]:
    try:
        text = engine_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return set()
    used: set[str] = set()
    for cap, patterns in CAP_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            used.add(cap)
    # Heuristic: a VaultRead/Write requires the underlying FileSystem too.
    # Don't double-count — caller treats VaultRead as a "tighter" subset.
    if used & {"VaultRead", "VaultWrite"}:
        used.discard("FileSystem") if "FileSystem" in used and not _has_non_vault_fs(text) else None
    return used


def _has_non_vault_fs(text: str) -> bool:
    """True if the engine touches fs outside TheVault paths."""
    # Look for fs write/read with a path that doesn't mention TheVault.
    for m in re.finditer(r"fs\.\s*(?:writeFile\w*|readFile\w*|appendFile)\s*\(([^,)]*)", text):
        arg = m.group(1)
        if "TheVault" not in arg:
            return True
    return False


# ----- reporting ------------------------------------------------------------

@dataclass
class DriftReport:
    agent: str
    engine: Path | None
    declared: set[str]
    used: set[str]
    missing_grants: set[str]  # used but not declared
    unused_grants: set[str]   # declared but never used


def check_agent(spec: AgentSpec) -> DriftReport:
    engine = resolve_engine(spec.name)
    used: set[str] = detect_used_caps(engine) if engine else set()
    missing = used - spec.declared_caps
    unused = spec.declared_caps - used if engine else set()
    return DriftReport(
        agent=spec.name,
        engine=engine,
        declared=spec.declared_caps,
        used=used,
        missing_grants=missing,
        unused_grants=unused,
    )


def format_report(reports: Iterable[DriftReport]) -> tuple[str, int]:
    lines: list[str] = []
    exit_code = 0
    for r in reports:
        head = f"{r.agent}"
        if r.engine is None:
            lines.append(f"  ?  {head}  (no engine file found — spec-only)")
            continue
        engine_rel = str(r.engine).replace(str(HOME), "~")
        if r.missing_grants:
            lines.append(f"  ✗  {head}  ({engine_rel})")
            lines.append(f"        DRIFT: engine uses {sorted(r.missing_grants)} but spec doesn't grant them")
            exit_code = 1
        elif r.unused_grants:
            lines.append(f"  ⚠  {head}  ({engine_rel})")
            lines.append(f"        declared {sorted(r.unused_grants)} but engine never uses them")
        else:
            lines.append(f"  ✔  {head}  ({engine_rel})")
        if r.declared or r.used:
            lines.append(f"        declared={sorted(r.declared) or '∅'}  used={sorted(r.used) or '∅'}")
    return "\n".join(lines), exit_code


# ----- CLI ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessera-drift",
        description="Check Tessera agent specs against their live engine implementations.",
    )
    parser.add_argument(
        "paths", nargs="*",
        help=".t.md files (defaults to scanning the vault's 200 Projects/)",
    )
    parser.add_argument(
        "--vault-dir", default=str(VAULT_DIR),
        help=f"Vault root (default: {VAULT_DIR})",
    )
    args = parser.parse_args(argv)

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        vault = Path(args.vault_dir)
        files = sorted((vault / "200 Projects").glob("*.t.md"))

    if not files:
        print("No .t.md files found.", file=sys.stderr)
        return 2

    print(f"Tessera drift check — {len(files)} spec(s)")
    print()
    overall_exit = 0
    for f in files:
        print(f"# {f.name}")
        specs = parse_tsr(f)
        reports = [check_agent(s) for s in specs]
        body, exit_code = format_report(reports)
        print(body)
        print()
        if exit_code:
            overall_exit = exit_code
    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
