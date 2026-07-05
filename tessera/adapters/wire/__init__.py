"""tson bridge — constrained decoding for Tessera prompts.

Tessera is the first consumer of the ``tson`` interchange format. This
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

The format itself lives in the standalone ``tson`` repo. If it is not
installed, this adapter degrades gracefully and Tessera behaves exactly as
before (no prompt is schema-bound unless ``world.prompt_schemas`` is populated).
"""
from __future__ import annotations

try:
    import tson as _tw
    from tson import Compiled, ValidationError, compile_schema
    from tson.schema import Field, Schema

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
            "tson is not installed; `pip install -e ../tson` "
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
    """Compile a Tessera ``ToolDecl`` straight to wire artifacts.

    Pins the call address to a fixed literal (``fixed_address="c1"``) rather
    than the schema-compiler's default free-generated ``id`` rule
    (``[a-z] [a-z0-9_]*``, unbounded repetition). Each `enforce_complete()`
    call is one independent completion — there's no multi-call correlation
    that needs a variable address — so letting the grammar generate one is
    pure downside: an unbounded repetition the sampler has no pressure to
    terminate, which can run past max_tokens before ever reaching the
    mandatory field(s) that follow it, producing a truncated, field-missing
    "valid prefix" record. Observed directly: a real completion ran ~230
    tokens into the address alone and never reached its one required field.
    """
    _require()
    return compile_schema(schema_from_tool(tool), fixed_address="c1")


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


def parse_emitted_record(text: str):
    """Parse a validated wire record line back into a tson Record (for dispatch)."""
    _require()
    return _tw.parse_record(_first_record_line(text))


def _first_record_line(text: str) -> str:
    """Pull the first plausible record line from a model completion (models
    sometimes add a code fence or chatter)."""
    for raw in text.strip().splitlines():
        ln = raw.strip().strip("`").strip()
        if ln and ln[0] in "!=>?@~":
            return ln
    return text.strip().splitlines()[0].strip() if text.strip() else text


def enforce_complete(
    backend, prompt_text: str, compiled: "Compiled", *, max_tokens: int = 256, max_repairs: int = 1
):
    """Run a constrained completion against ``backend`` and return a
    ``CompletionResult`` whose text is the validated wire record.

    On a ``gbnf``/``jsonschema`` tier the *shape* is guaranteed by construction —
    but a large/open numeric range is one of the documented honest limits (GBNF
    enumerates small ranges exactly; a large one falls back to a generic
    int/float rule and only the validator enforces the bound). A semantic miss
    there still gets up to ``max_repairs`` bounded retries, same mechanism as
    the ``none`` tier's retry: quote the validator's exact error back to the
    model and re-ask. The grammar/schema constraint stays active on the retry,
    so only the *value* needs another sample — structure was never in question.
    """
    _require()
    opts = constrain_opts(compiled, backend)
    result = backend.complete(prompt_text, max_tokens=max_tokens, **opts)
    line = _first_record_line(result.text)
    attempt_prompt = prompt_text

    for _ in range(max_repairs + 1):
        try:
            compiled.validate(line)
            result.text = line
            return result
        except ValidationError as e:
            from tson.repair import build_repair_prompt

            attempt_prompt = build_repair_prompt(attempt_prompt, line, e)
            result = backend.complete(attempt_prompt, max_tokens=max_tokens, **opts)
            line = _first_record_line(result.text)

    # Retries exhausted: best-effort, caller sees the last (possibly still
    # invalid) attempt rather than raising.
    result.text = line
    return result
