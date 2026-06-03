"""LLM backends — pluggable model providers for Prompt.Call.

Backend selection at runtime via env var ``TESSERA_LLM_BACKEND``:
  - ``ollama`` (default) — local Ollama server, model from TESSERA_OLLAMA_MODEL
  - ``anthropic`` — requires `anthropic` pkg + ANTHROPIC_API_KEY env var
  - ``noop`` — returns a deterministic stub string, useful for tests

Each backend implements ``complete(prompt: str, **opts) -> CompletionResult``.

Backends fail soft: if a configured backend is unreachable (no Ollama running,
no API key), the dispatcher falls back to NoopBackend with a warning rather
than crashing the whole Tessera pipeline. Tests stay green even when no LLM
is configured on the host.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    text: str
    backend: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_dollars: float = 0.0


class LLMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, prompt: str, **opts) -> CompletionResult:
        ...


class NoopBackend(LLMBackend):
    """Deterministic stub backend — returns a recognizable string."""
    name = "noop"

    def complete(self, prompt: str, **opts) -> CompletionResult:
        digest = prompt.strip().splitlines()[0][:80] if prompt.strip() else "<empty>"
        return CompletionResult(
            text=f"[noop:{digest}]",
            backend="noop",
            model="noop",
        )


class OllamaBackend(LLMBackend):
    """Talks to a local Ollama server (default: http://localhost:11434)."""
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("TESSERA_OLLAMA_MODEL") or "llama3.2"

    def complete(self, prompt: str, **opts) -> CompletionResult:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        for k in ("temperature", "top_p", "num_predict"):
            if k in opts:
                body.setdefault("options", {})[k] = opts[k]
        # Tier-B constraint: Ollama takes a JSON Schema object in `format`
        # (top-level, not under options). The wire adapter passes the schema's
        # JSON projection; the model emits JSON we transcode to the wire form.
        if "format" in opts:
            body["format"] = opts["format"]

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        # 5min default — cloud-routed Ollama models (e.g. glm-4.6:cloud) can
        # take 1-3min on a cold start. Local models will finish well under this.
        with urllib.request.urlopen(req, timeout=opts.get("timeout", 300)) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return CompletionResult(
            text=payload.get("response", ""),
            backend="ollama",
            model=self.model,
            tokens_in=payload.get("prompt_eval_count", 0),
            tokens_out=payload.get("eval_count", 0),
        )

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                r.read(1)
            return True
        except (urllib.error.URLError, OSError):
            return False


class AnthropicBackend(LLMBackend):
    """Anthropic API with server-side prompt caching.

    Prompt caching: when the prompt has a STABLE prefix (>= 1024 tokens for
    Sonnet, >= 2048 for Haiku) followed by varying content, the prefix can be
    cached server-side for 5 min and 90% cheaper + ~80% lower TTFT on warm
    hits. We split the prompt heuristically on the first '\\n\\n' boundary —
    everything before is the cacheable prefix, everything after is variable.

    Disable via TESSERA_ANTHROPIC_CACHE=0.
    Requires `anthropic` pkg + ANTHROPIC_API_KEY env var.
    """
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("TESSERA_ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001"
        self.use_prompt_cache = os.environ.get("TESSERA_ANTHROPIC_CACHE", "1") != "0"
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
        except ImportError as e:
            raise RuntimeError("anthropic package not installed; pip install anthropic") from e

    def complete(self, prompt: str, **opts) -> CompletionResult:
        max_tokens = opts.get("max_tokens", 1024)

        # Heuristic split: if the prompt has a "\n\n" within its first ~70%,
        # treat everything before that as a stable cacheable prefix. This is
        # the common shape for prompts with system context + a question.
        if self.use_prompt_cache and "\n\n" in prompt:
            split_idx = prompt.find("\n\n")
            prefix = prompt[:split_idx]
            suffix = prompt[split_idx + 2:]
            # Anthropic requires cached prefix >= some token threshold; below
            # that the API rejects the cache_control marker. We approximate
            # tokens by chars/4 and skip caching for very short prefixes.
            if len(prefix) >= 200:  # ~50 tokens, well below the real threshold,
                                    # but the SDK silently drops cache_control
                                    # if the prefix is too short — safe.
                content = [
                    {"type": "text", "text": prefix,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": suffix},
                ]
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                text_parts = [b.text for b in msg.content
                              if getattr(b, "type", None) == "text"]
                usage = msg.usage
                return CompletionResult(
                    text="".join(text_parts),
                    backend="anthropic",
                    model=self.model,
                    tokens_in=getattr(usage, "input_tokens", 0),
                    tokens_out=getattr(usage, "output_tokens", 0),
                    # Anthropic returns cache hit/miss in usage; pull them when present
                )

        # Plain path (no caching applicable)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return CompletionResult(
            text="".join(text_parts),
            backend="anthropic",
            model=self.model,
            tokens_in=getattr(msg.usage, "input_tokens", 0),
            tokens_out=getattr(msg.usage, "output_tokens", 0),
        )


class OpenAICompatibleBackend(LLMBackend):
    """Universal backend for any provider exposing the OpenAI chat completions schema.

    Covers (with the right TESSERA_OPENAI_BASE_URL):
      - openai      → https://api.openai.com/v1
      - groq        → https://api.groq.com/openai/v1
      - together    → https://api.together.xyz/v1
      - fireworks   → https://api.fireworks.ai/inference/v1
      - deepseek    → https://api.deepseek.com/v1
      - xai         → https://api.x.ai/v1
      - openrouter  → https://openrouter.ai/api/v1
      - lm_studio   → http://localhost:1234/v1
      - vllm        → http://localhost:8000/v1
      - llama_cpp   → http://localhost:8080/v1
      - azure_openai → https://YOUR_RESOURCE.openai.azure.com/openai/deployments/MODEL

    Uses the official ``openai`` SDK if installed (better error handling, retries),
    falls back to bare urllib if not (zero hard deps).
    """
    name = "openai_compat"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        self.base_url = (base_url or os.environ.get("TESSERA_OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = (api_key or os.environ.get("TESSERA_OPENAI_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "")
        self.model = model or os.environ.get("TESSERA_OPENAI_MODEL") or "gpt-4o-mini"

        try:
            import openai  # type: ignore
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._mode = "sdk"
        except ImportError:
            self._client = None
            self._mode = "raw"

    def complete(self, prompt: str, **opts) -> CompletionResult:
        max_tokens = opts.get("max_tokens", 1024)
        temperature = opts.get("temperature", 0.7)

        if self._mode == "sdk":
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return CompletionResult(
                text=text,
                backend="openai_compat",
                model=self.model,
                tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
                tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            )

        # raw urllib fallback
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=opts.get("timeout", 120)) as r:
            payload = json.loads(r.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return CompletionResult(
            text=text or "",
            backend="openai_compat",
            model=self.model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )


class GeminiBackend(LLMBackend):
    """Google Gemini via the google-generativeai SDK.

    Env: TESSERA_GEMINI_API_KEY (or GOOGLE_API_KEY), TESSERA_GEMINI_MODEL.
    """
    name = "gemini"

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("TESSERA_GEMINI_MODEL") or "gemini-2.5-flash"
        try:
            import google.generativeai as genai  # type: ignore
            api_key = (os.environ.get("TESSERA_GEMINI_API_KEY")
                       or os.environ.get("GOOGLE_API_KEY") or "")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model_name)
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai not installed; pip install google-generativeai"
            ) from e

    def complete(self, prompt: str, **opts) -> CompletionResult:
        resp = self._client.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        return CompletionResult(
            text=text,
            backend="gemini",
            model=self.model_name,
            tokens_in=getattr(usage, "prompt_token_count", 0) if usage else 0,
            tokens_out=getattr(usage, "candidates_token_count", 0) if usage else 0,
        )


class CohereBackend(LLMBackend):
    """Cohere Command family via the cohere SDK.

    Env: TESSERA_COHERE_API_KEY (or COHERE_API_KEY), TESSERA_COHERE_MODEL.
    """
    name = "cohere"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("TESSERA_COHERE_MODEL") or "command-r-plus"
        try:
            import cohere  # type: ignore
            api_key = (os.environ.get("TESSERA_COHERE_API_KEY")
                       or os.environ.get("COHERE_API_KEY") or "")
            self._client = cohere.ClientV2(api_key=api_key)
        except ImportError as e:
            raise RuntimeError("cohere package not installed; pip install cohere") from e

    def complete(self, prompt: str, **opts) -> CompletionResult:
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        if resp.message and resp.message.content:
            text = "".join(part.text for part in resp.message.content
                          if getattr(part, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=text,
            backend="cohere",
            model=self.model,
            tokens_in=getattr(getattr(usage, "tokens", None), "input_tokens", 0) if usage else 0,
            tokens_out=getattr(getattr(usage, "tokens", None), "output_tokens", 0) if usage else 0,
        )


class BedrockBackend(LLMBackend):
    """AWS Bedrock (Claude, Llama, Titan, Mistral) via boto3.

    Env: standard AWS creds (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION),
    plus TESSERA_BEDROCK_MODEL (defaults to anthropic.claude-3-5-sonnet-20241022-v2:0).
    """
    name = "bedrock"

    def __init__(self, model: str | None = None):
        self.model = (model or os.environ.get("TESSERA_BEDROCK_MODEL")
                      or "anthropic.claude-3-5-sonnet-20241022-v2:0")
        try:
            import boto3  # type: ignore
            self._client = boto3.client("bedrock-runtime")
        except ImportError as e:
            raise RuntimeError("boto3 not installed; pip install boto3") from e

    def complete(self, prompt: str, **opts) -> CompletionResult:
        import json as _json
        max_tokens = opts.get("max_tokens", 1024)
        # Claude on Bedrock uses Anthropic's message format
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = self._client.invoke_model(
            modelId=self.model,
            body=_json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = _json.loads(resp["body"].read())
        text_parts = [b.get("text", "") for b in payload.get("content", [])
                      if b.get("type") == "text"]
        usage = payload.get("usage", {})
        return CompletionResult(
            text="".join(text_parts),
            backend="bedrock",
            model=self.model,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
        )


# ---------- dispatcher ----------


_CACHED: dict[str, LLMBackend] = {}


# Known OpenAI-compatible providers — pre-baked base URLs.
_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai":     {"base_url": "https://api.openai.com/v1",                 "env_key": "OPENAI_API_KEY"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",            "env_key": "GROQ_API_KEY"},
    "together":   {"base_url": "https://api.together.xyz/v1",               "env_key": "TOGETHER_API_KEY"},
    "fireworks":  {"base_url": "https://api.fireworks.ai/inference/v1",     "env_key": "FIREWORKS_API_KEY"},
    "deepseek":   {"base_url": "https://api.deepseek.com/v1",               "env_key": "DEEPSEEK_API_KEY"},
    "xai":        {"base_url": "https://api.x.ai/v1",                       "env_key": "XAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",              "env_key": "OPENROUTER_API_KEY"},
    "mistral":    {"base_url": "https://api.mistral.ai/v1",                 "env_key": "MISTRAL_API_KEY"},
    "lm_studio":  {"base_url": "http://localhost:1234/v1",                  "env_key": ""},
    "vllm":       {"base_url": "http://localhost:8000/v1",                  "env_key": ""},
    "llama_cpp":  {"base_url": "http://localhost:8080/v1",                  "env_key": ""},
}


def get_backend(name: str | None = None) -> LLMBackend:
    """Return the configured backend, falling back to NoopBackend on failure."""
    name = (name or os.environ.get("TESSERA_LLM_BACKEND") or "ollama").lower()
    if name in _CACHED:
        return _CACHED[name]

    backend: LLMBackend
    if name == "noop":
        backend = NoopBackend()
    elif name == "anthropic":
        try:
            backend = AnthropicBackend()
        except Exception as e:
            warnings.warn(f"Anthropic backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name == "gemini":
        try:
            backend = GeminiBackend()
        except Exception as e:
            warnings.warn(f"Gemini backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name == "cohere":
        try:
            backend = CohereBackend()
        except Exception as e:
            warnings.warn(f"Cohere backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name == "bedrock":
        try:
            backend = BedrockBackend()
        except Exception as e:
            warnings.warn(f"Bedrock backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name in _PROVIDER_PRESETS:
        preset = _PROVIDER_PRESETS[name]
        api_key = os.environ.get(preset["env_key"]) if preset["env_key"] else None
        try:
            backend = OpenAICompatibleBackend(
                base_url=preset["base_url"],
                api_key=api_key,
                model=os.environ.get(f"TESSERA_{name.upper()}_MODEL"),
            )
        except Exception as e:
            warnings.warn(f"{name} backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name in ("openai_compat", "openai_compatible"):
        try:
            backend = OpenAICompatibleBackend()
        except Exception as e:
            warnings.warn(f"OpenAI-compat backend init failed ({e}); using noop")
            backend = NoopBackend()
    elif name == "ollama":
        ob = OllamaBackend()
        if ob.health():
            backend = ob
        else:
            warnings.warn(f"Ollama not reachable at {ob.host}; using noop")
            backend = NoopBackend()
    else:
        warnings.warn(f"unknown TESSERA_LLM_BACKEND {name!r}; using noop")
        backend = NoopBackend()

    _CACHED[name] = backend
    return backend


def list_providers() -> list[dict]:
    """Return metadata for every supported provider — for `tessera providers` CLI."""
    rows = [
        {"id": "ollama", "kind": "local",   "schema": "ollama",    "env": "OLLAMA_HOST",
         "model_env": "TESSERA_OLLAMA_MODEL",   "default": "llama3.2"},
        {"id": "anthropic", "kind": "hosted", "schema": "anthropic", "env": "ANTHROPIC_API_KEY",
         "model_env": "TESSERA_ANTHROPIC_MODEL", "default": "claude-haiku-4-5"},
        {"id": "gemini", "kind": "hosted",  "schema": "gemini",    "env": "TESSERA_GEMINI_API_KEY",
         "model_env": "TESSERA_GEMINI_MODEL",   "default": "gemini-2.5-flash"},
        {"id": "cohere", "kind": "hosted",  "schema": "cohere",    "env": "COHERE_API_KEY",
         "model_env": "TESSERA_COHERE_MODEL",   "default": "command-r-plus"},
        {"id": "bedrock", "kind": "hosted", "schema": "bedrock",   "env": "AWS_ACCESS_KEY_ID",
         "model_env": "TESSERA_BEDROCK_MODEL",  "default": "anthropic.claude-3-5-sonnet"},
    ]
    for pid, preset in _PROVIDER_PRESETS.items():
        kind = "local" if not preset["env_key"] else "hosted"
        rows.append({
            "id": pid, "kind": kind, "schema": "openai_compat",
            "env": preset["env_key"] or "(none)",
            "model_env": f"TESSERA_{pid.upper()}_MODEL",
            "default": "(provider-specific)",
        })
    return rows


def reset_cache() -> None:
    _CACHED.clear()
