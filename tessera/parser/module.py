"""ParsedModule — frontmatter + substrate blocks extracted from a .tsr.md file.

This is the minimum needed to feed the SIR emitter. The grammar of the substrate
bodies themselves is parsed later (sir/build.py); here we only split the file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    file: str
    line_start: int
    line_end: int


@dataclass
class SubstrateBlock:
    substrate: str
    body: str
    span: SourceSpan
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedModule:
    path: str
    frontmatter: dict[str, Any]
    blocks: list[SubstrateBlock]
    prose: str

    def blocks_of(self, substrate: str) -> list[SubstrateBlock]:
        return [b for b in self.blocks if b.substrate == substrate]


_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FENCE = re.compile(
    r"^```tsr:([\w:]+)(?:[ \t]+([^\n]+))?\n(.*?)^```\s*$",
    re.DOTALL | re.MULTILINE,
)


def _parse_yaml_lite(text: str) -> dict[str, Any]:
    """Toy YAML — enough for our frontmatter. No external deps."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            out[key] = None
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [s.strip().strip('"\'') for s in inner.split(",")] if inner else []
        elif value.startswith("{") and value.endswith("}"):
            out[key] = _parse_inline_map(value[1:-1])
        else:
            out[key] = value.strip('"\'')
    return out


def _parse_inline_map(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in text.split(","):
        if ":" not in part:
            continue
        k, _, v = part.partition(":")
        out[k.strip()] = v.strip().strip('"\'')
    return out


def _parse_attrs(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    attrs: dict[str, str] = {}
    for chunk in raw.split():
        if "=" in chunk:
            k, _, v = chunk.partition("=")
            attrs[k.strip()] = v.strip().strip('"\'[]')
    return attrs


def parse_source(source: str, path: str = "<string>") -> ParsedModule:
    frontmatter: dict[str, Any] = {}
    rest = source
    m = _FRONTMATTER.match(source)
    if m:
        frontmatter = _parse_yaml_lite(m.group(1))
        rest = source[m.end():]

    blocks: list[SubstrateBlock] = []
    line_offset = source[: m.end() if m else 0].count("\n")
    for fm in _FENCE.finditer(rest):
        substrate = fm.group(1).strip()
        attrs = _parse_attrs(fm.group(2))
        body = fm.group(3)
        start_line = line_offset + rest[: fm.start()].count("\n") + 1
        end_line = start_line + body.count("\n") + 2
        blocks.append(
            SubstrateBlock(
                substrate=substrate,
                body=body,
                span=SourceSpan(file=path, line_start=start_line, line_end=end_line),
                attrs=attrs,
            )
        )

    prose = _FENCE.sub("", rest).strip()

    return ParsedModule(path=path, frontmatter=frontmatter, blocks=blocks, prose=prose)


def parse_file(path: str | Path) -> ParsedModule:
    p = Path(path)
    return parse_source(p.read_text(), path=str(p))
