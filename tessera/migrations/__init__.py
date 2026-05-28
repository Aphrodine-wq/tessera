"""Migration framework for .t.md files across Tessera language versions.

A `.t.md` file declares its language version in frontmatter:

    tessera_version: 0.2

Files without that field are assumed to be 0.1 (the version before
per-block persistence on memory:semantic shipped). When the parser
loads a file, it runs every migration whose `from_version` is at or
above the file's declared version and whose `to_version` is at or
below the language's CURRENT_VERSION, in order.

A migration is a function `(ParsedModule) -> ParsedModule`. It can
edit `frontmatter`, `blocks`, `attrs`, or `prose`. After running,
the parser hands the (possibly-rewritten) module to the SIR builder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..parser.module import ParsedModule

CURRENT_VERSION = "0.2"
DEFAULT_VERSION = "0.1"


@dataclass(frozen=True)
class Migration:
    from_version: str
    to_version: str
    transform: Callable[[ParsedModule], ParsedModule]
    note: str


def _migrate_0_1_to_0_2(mod: ParsedModule) -> ParsedModule:
    """0.1 → 0.2: every memory:semantic block gains an explicit `persistent`
    attribute. Default is `true` (matches 0.1 behavior where semantic memory
    always persisted). Authors who want ephemeral semantic memory in 0.2+
    set `persistent=false` on the fence.
    """
    for block in mod.blocks:
        if block.substrate == "memory:semantic":
            block.attrs.setdefault("persistent", "true")
    return mod


REGISTRY: list[Migration] = [
    Migration(
        from_version="0.1",
        to_version="0.2",
        transform=_migrate_0_1_to_0_2,
        note="memory:semantic gains explicit persistent attribute",
    ),
]


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def apply_migrations(mod: ParsedModule) -> ParsedModule:
    declared = str(mod.frontmatter.get("tessera_version") or DEFAULT_VERSION)
    current = _version_key(CURRENT_VERSION)
    cursor = _version_key(declared)
    while cursor < current:
        step = next(
            (m for m in REGISTRY if _version_key(m.from_version) == cursor),
            None,
        )
        if step is None:
            raise RuntimeError(
                f"no migration from {'.'.join(str(c) for c in cursor)} "
                f"toward {CURRENT_VERSION}"
            )
        mod = step.transform(mod)
        cursor = _version_key(step.to_version)
    mod.frontmatter["tessera_version"] = CURRENT_VERSION
    return mod
