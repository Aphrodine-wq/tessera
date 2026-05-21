"""AEON adapter — hands SIR to AEON for verification.

Strategy (per docs/aeon-bridge.md):

1. Write the textual SIR to a temp `.sir.txt` file.
2. Call `aeon.language_adapter.verify_file` if AEON is importable in the
   current env, OR shell out to the `aeon` CLI as fallback.
3. Map AEON diagnostics back to Tessera's RFC §Appendix-C error codes.

When AEON is unreachable (not installed / no `tessera_adapter.py` in tree yet),
this returns an empty list with a warning rather than blowing up — the local
passes in tessera.verify.passes still run regardless.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from ...verify.passes import Diagnostic


_AEON_CATEGORY_TO_RFC_CODE = {
    "taint_analysis": "E301",
    "auth_access_control": "E102",
    "resource_logic": "E201",
    "secret_detection": "E301",
}


def _map_finding(f: dict) -> Diagnostic:
    cat = f.get("category", "")
    code = _AEON_CATEGORY_TO_RFC_CODE.get(cat, f"AEON-{f.get('id', '?')}")
    return Diagnostic(
        code=code,
        severity=f.get("severity", "warning"),
        region=str(f.get("location", {}).get("file", "?")),
        node=str(f.get("location", {}).get("line", "?")),
        message=f.get("message", "(no message)"),
    )


def _try_import_aeon():
    try:
        from aeon.language_adapter import verify_file  # type: ignore
        return verify_file
    except ImportError:
        return None


def verify_sir_text(sir_text: str) -> list[Diagnostic]:
    """Run AEON over the given SIR text. Returns mapped Tessera diagnostics.

    Wrapped with the SIR-hash verify cache so the same SIR isn't re-verified
    on subsequent compiles. Disable via TESSERA_NO_VERIFY_CACHE=1.

    Canonicalizes the SIR text before hashing so two SIRs that differ only in
    random node ids share the cache slot — much higher hit rate.
    """
    from ...cache import verify_cache_get, verify_cache_put
    from ...sir.canonical import canonicalize

    cache_key = canonicalize(sir_text)
    cached = verify_cache_get(cache_key)
    if cached is not None:
        return [Diagnostic(**d) for d in cached]

    verify_file = _try_import_aeon()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sir", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(sir_text)
        tmp_path = Path(tmp.name)

    try:
        if verify_file is not None:
            try:
                result = verify_file(
                    str(tmp_path),
                    deep_verify=True,
                    prove_kwargs={
                        "taint_analysis": True,
                        "auth_check": True,
                        "secret_detection": True,
                    },
                )
                findings = list(getattr(result, "errors", [])) + list(getattr(result, "warnings", []))
                diagnostics = [_map_finding(f) for f in findings]
                verify_cache_put(cache_key, [d.__dict__ for d in diagnostics])
                return diagnostics
            except Exception as exc:
                warnings.warn(f"AEON in-process verify failed: {exc}; falling back to CLI")

        # CLI fallback
        aeon_bin = shutil.which("aeon")
        if aeon_bin is None:
            warnings.warn("AEON not reachable (no module, no CLI on PATH) — skipping AEON pass")
            return []
        proc = subprocess.run(
            [aeon_bin, "check", str(tmp_path), "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode not in (0, 1):  # 1 = found issues, still valid output
            warnings.warn(f"AEON CLI failed (rc={proc.returncode}): {proc.stderr[:200]}")
            return []
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            warnings.warn("AEON CLI returned non-JSON output — skipping")
            return []
        findings = payload.get("errors", []) + payload.get("warnings", [])
        diagnostics = [_map_finding(f) for f in findings]
        verify_cache_put(cache_key, [d.__dict__ for d in diagnostics])
        return diagnostics
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
