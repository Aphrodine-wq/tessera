"""Obsidian vault adapter — Tessera works directly on Obsidian vaults.

An Obsidian vault is just a directory of markdown files. Tessera treats any
``.tsr.md`` file as an agent declaration. This adapter lets you:

- **Scan** a vault: walk the tree, find every Tessera agent, report what it
  declares (substrates used, capabilities requested, prompts/tools/models).
- **Run** an agent from its vault path: ``vault run ~/Vault/Agents/Foo.tsr.md``
  is the same as ``compile`` but resolves relative paths against the vault root.
- **Scaffold** a new agent in the vault from a starter template, so you can
  drop into the vault folder and have a working `.tsr.md` ready to edit.

Wiki-link resolution (``[[NoteName]]`` inside templates) is best-effort: we
scan the vault for any ``.md`` file whose stem matches the link, and inline
its content. Multiple matches → first by alphabetical path.

Safety: scan + resolve are pure reads. ``scaffold_agent`` writes ONE file to
an explicit path you provide — same model as the Synapse adapter's opt-in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ...parser.module import ParsedModule, parse_file


# ---------- data shapes ----------


@dataclass
class VaultAgent:
    """One Tessera agent found in an Obsidian vault."""
    vault_root: Path
    file_path: Path
    agent_name: str
    substrates: list[str] = field(default_factory=list)
    capabilities_requested: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    neural_models: list[str] = field(default_factory=list)

    @property
    def vault_relative(self) -> Path:
        try:
            return self.file_path.relative_to(self.vault_root)
        except ValueError:
            return self.file_path


@dataclass
class VaultScan:
    vault_root: Path
    agents: list[VaultAgent] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)


# ---------- scan ----------


def _looks_like_tessera(path: Path) -> bool:
    """Quick filter — true if the file is named .tsr.md OR has Tessera blocks."""
    if path.name.endswith(".tsr.md"):
        return True
    if path.suffix != ".md":
        return False
    try:
        head = path.read_text(errors="ignore")[:4096]
    except OSError:
        return False
    return "```tsr:" in head


def scan_vault(vault_root: str | Path, *, max_files: int = 5000,
               parallel: bool = True, max_workers: int = 8) -> VaultScan:
    """Walk vault_root looking for Tessera agent files.

    max_files caps the walk — protects against accidentally scanning ~ or /.
    parallel=True uses a ThreadPoolExecutor for I/O parallelism on parse step.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    root = Path(vault_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"vault root not found: {root}")

    scan = VaultScan(vault_root=root)
    candidate_paths: list[Path] = []
    count = 0
    for path in root.rglob("*.md"):
        count += 1
        if count > max_files:
            scan.skipped.append((path, f"hit max_files={max_files}; stopping walk"))
            break
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not _looks_like_tessera(path):
            continue
        candidate_paths.append(path)

    def _process(path: Path):
        try:
            from ...cache import parse_file_cached
            pm = parse_file_cached(path)
            return path, _agents_from_parsed(pm, root, path), None
        except Exception as e:
            return path, None, f"parse failed: {type(e).__name__}: {e}"

    if parallel and len(candidate_paths) > 4:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for fut in as_completed(pool.submit(_process, p) for p in candidate_paths):
                path, agents, err = fut.result()
                if err:
                    scan.skipped.append((path, err))
                else:
                    scan.agents.extend(agents)
    else:
        for path in candidate_paths:
            path, agents, err = _process(path)
            if err:
                scan.skipped.append((path, err))
            else:
                scan.agents.extend(agents)

    # Stable ordering for reproducible output.
    scan.agents.sort(key=lambda a: (str(a.file_path), a.agent_name))
    return scan


def _agents_from_parsed(pm: ParsedModule, vault_root: Path, file_path: Path) -> list[VaultAgent]:
    """One ParsedModule may declare multiple agent blocks; one row per agent."""
    from ..build_helpers import sniff_block_features  # local helper below

    sniff = sniff_block_features(pm)
    declared_name = pm.frontmatter.get("agent")

    rows: list[VaultAgent] = []
    for agent_name in sniff["agent_names"] or ([declared_name] if declared_name else []):
        rows.append(VaultAgent(
            vault_root=vault_root,
            file_path=file_path,
            agent_name=agent_name,
            substrates=sniff["substrates"],
            capabilities_requested=list(pm.frontmatter.get("capabilities_requested") or []),
            prompts=sniff["prompts"],
            tools=sniff["tools"],
            neural_models=sniff["neural_models"],
        ))
    return rows


# ---------- scaffold ----------


_AGENT_TEMPLATES: dict[str, str] = {
    "basic": '''---
agent: {agent}
capabilities_requested: []
max_cost: {{ dollars: 0.00, tokens: 0 }}
---

# {agent}

Describe what this agent does. The first paragraph is shown in `vault scan`.

```tsr:agent
agent {agent} {{
  beliefs:
    @last_write input: String

  intentions:
    plan respond {{
      let reply = "echo: " + input
      return reply
    }}
}}
```
''',

    "llm": '''---
agent: {agent}
capabilities_requested: [NetworkOut]
max_cost: {{ dollars: 0.05, tokens: 2000 }}
---

# {agent}

An LLM-backed agent. Picks up the configured backend via TESSERA_LLM_BACKEND.

```tsr:prompt
prompt answer(question: String) -> String = "Answer concisely: {{question}}"
```

```tsr:agent
agent {agent} {{
  beliefs:
    @last_write question: String

  intentions:
    plan think {{
      let reply = answer(question)
      return reply
    }}
}}
''' + "```\n",

    "journal": '''---
agent: {agent}
capabilities_requested: []
max_cost: {{ dollars: 0.00, tokens: 0 }}
---

# {agent}

A journaling agent — logs every input to its episodic memory.

```tsr:memory:episodic
episodic {{
  event Entry(content: String)
}}
```

```tsr:agent
agent {agent} {{
  beliefs:
    @last_write entry: String

  intentions:
    plan record {{
      log Entry(entry)
      return "logged"
    }}
}}
''' + "```\n",
}


def scaffold_agent(
    target_path: str | Path,
    agent_name: str,
    template: str = "basic",
    overwrite: bool = False,
) -> Path:
    """Write a starter .tsr.md at target_path. Raises if it exists and overwrite=False."""
    if template not in _AGENT_TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; available: {sorted(_AGENT_TEMPLATES)}"
        )
    path = Path(target_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass overwrite=True)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_AGENT_TEMPLATES[template].format(agent=agent_name))
    return path


# ---------- wiki-link resolution (best-effort) ----------


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]+)?(?:#[^\]]+)?\]\]")


def resolve_wikilinks(template: str, vault_root: str | Path) -> str:
    """Replace [[NoteName]] in template with the matching note's content.

    Best-effort — first matching .md file wins by alphabetical path. Falls
    through to the literal link if no match found.
    """
    root = Path(vault_root).expanduser().resolve()
    if not root.exists():
        return template

    cache: dict[str, str | None] = {}

    def _replace(m: re.Match) -> str:
        link = m.group(1).strip()
        if link not in cache:
            cache[link] = _find_note_text(link, root)
        return cache[link] or m.group(0)

    return _WIKILINK_RE.sub(_replace, template)


def _find_note_text(stem: str, root: Path) -> str | None:
    matches = sorted(root.rglob(f"{stem}.md"))
    if not matches:
        return None
    return matches[0].read_text(errors="ignore")
