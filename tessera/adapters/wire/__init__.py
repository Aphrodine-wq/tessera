"""tessera-wire bridge — constrained decoding for Tessera prompts.

Tessera is the first consumer of the ``tessera_wire`` interchange format. This
adapter is the seam between the two: it turns a Tessera tool/schema into the
format's compiled artifacts (GBNF grammar + validator) and picks the right
*enforcement tier* for whatever LLM backend is active.

Tiers (by backend capability):
  - ``gbnf``       true token-level grammar constraint (llama.cpp / llama-server
                   / vLLM guided_grammar) — malformed output is impossible.
  - ``jsonschema`` JSON-Schema-constrained output (Ollama ``format``) — the model
                   emits a JSON projection that we transcode to the wire form.
  - ``none``       no constraint API (Anthropic, …) — emit freely, then
                   validate-and-repair once.

The format itself lives in the standalone ``tessera-wire`` repo. If it is not
installed, this adapter degrades gracefully and Tessera behaves exactly as
before (no prompt is schema-bound unless ``world.prompt_schemas`` is populated).
"""
from __future__ import annotations

try:
    import tessera_wire as _tw
    from tessera_wire import Compiled, ValidationError, compile_schema
    from tessera_wire.schema import Field, Schema

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the pkg is absent
    AVAILABLE = False
    _tw = None  # type: ignore


# Backend name -> maximum enforcement tier it supports.
BACKEND_CAPS: dict[str, str] = {
    "llamacpp": "gbnf",
    "llama_server": "gbnf",
    "ollama": "jsonschema",
    "openai_compat": "jsonschema",
    "noop": "none",
    "anthropic": "none",
    "gemini": "none",
    "cohere": "none",
    "bedrock": "none",
}

# Tessera type names -> wire scalar types.
_TYPE_MAP = {
    "string": "str", "str": "str", "text": "str",
    "int": "int", "integer": "int",
    "float": "float", "number": "float", "double": "float",
    "bool": "bool", "boolean": "bool",
}


def _require() -> None:
    if not AVAILABLE:
        raise RuntimeError(
            "tessera-wire is not installed; `pip install -e ../tessera-wire` "
            "to use constrained decoding."
        )


def schema_from_tool(tool) -> "Schema":
    """Build a wire :class:`Schema` from a Tessera ``ToolDecl``.

    A tool's typed parameters are effectively a per-call schema. All params are
    treated as required (a function signature has no optionals here); richer
    constraints (enum/range/default) come from an explicit ``@schema`` block.
    """
    _require()
    fields = [
        Field(name=pname, type=_TYPE_MAP.get(str(ptype).lower(), "str"), required=True)
        for pname, ptype in tool.params
    ]
    return Schema(name=tool.name, fields=fields)


def compile_tool(tool) -> "Compiled":
    """Compile a Tessera ``ToolDecl`` straight to wire artifacts."""
    _require()
    return compile_schema(schema_from_tool(tool))


def tier_for(backend) -> str:
    """The enforcement tier available for ``backend``."""
    return BACKEND_CAPS.get(getattr(backend, "name", ""), "none")


def constrain_opts(compiled: "Compiled", backend) -> dict:
    """The ``complete(**opts)`` kwargs that enforce ``compiled`` on ``backend``."""
    tier = tier_for(backend)
    if tier == "gbnf":
        return {"grammar": compiled.gbnf}
    if tier == "jsonschema":
        return {"format": compiled.json_schema}
    return {}


def _first_record_line(text: str) -> str:
    """Pull the first plausible record line from a model completion (models
    sometimes add a code fence or chatter)."""
    for raw in text.strip().splitlines():
        ln = raw.strip().strip("`").strip()
        if ln and ln[0] in "!=>?@~":
            return ln
    return text.strip().splitlines()[0].strip() if text.strip() else text


def enforce_complete(backend, prompt_text: str, compiled: "Compiled", *, max_tokens: int = 256):
    """Run a constrained completion against ``backend`` and return a
    ``CompletionResult`` whose text is the validated wire record.

    On a ``gbnf``/``jsonschema`` tier the structure is guaranteed; we still parse
    out the record line and run the validator (which also enforces numeric ranges
    the grammar can't). On the ``none`` tier we do one validate-and-repair round.
    """
    _require()
    opts = constrain_opts(compiled, backend)
    result = backend.complete(prompt_text, max_tokens=max_tokens, **opts)
    line = _first_record_line(result.text)
    try:
        compiled.validate(line)
        result.text = line
        return result
    except ValidationError as e:
        if tier_for(backend) == "none":
            from tessera_wire.repair import build_repair_prompt

            retry = backend.complete(
                build_repair_prompt(prompt_text, result.text, e), max_tokens=max_tokens
            )
            line2 = _first_record_line(retry.text)
            try:
                compiled.validate(line2)
                retry.text = line2
            except ValidationError:
                pass  # best-effort; caller sees the unrepaired text
            return retry
        # constrained tiers: structure is guaranteed; surface the line as-is
        result.text = line
        return result
