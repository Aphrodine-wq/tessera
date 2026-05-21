# Providers

Tessera talks to every major LLM provider through a single `LLMBackend`
interface. Pick a backend via `TESSERA_LLM_BACKEND=<id>` env var. Same `.tsr.md`
file runs against any of them.

## Quick start

```bash
# Local — free, no API key
TESSERA_LLM_BACKEND=ollama TESSERA_OLLAMA_MODEL=llama3.2 tessera compile ...

# OpenAI
TESSERA_LLM_BACKEND=openai TESSERA_OPENAI_MODEL=gpt-4o-mini tessera compile ...
# (uses OPENAI_API_KEY)

# Anthropic
TESSERA_LLM_BACKEND=anthropic tessera compile ...
# (uses ANTHROPIC_API_KEY)

# Google Gemini
TESSERA_LLM_BACKEND=gemini TESSERA_GEMINI_MODEL=gemini-2.5-flash tessera compile ...

# Groq (fast)
TESSERA_LLM_BACKEND=groq TESSERA_GROQ_MODEL=llama-3.3-70b-versatile tessera compile ...

# OpenRouter (every model behind one API key)
TESSERA_LLM_BACKEND=openrouter TESSERA_OPENROUTER_MODEL=anthropic/claude-sonnet-4 tessera compile ...
```

## Full matrix

| Provider | Backend id | Schema | API key env | Model env | Notes |
|---|---|---|---|---|---|
| **Ollama (local)** | `ollama` | ollama native | (none) | `TESSERA_OLLAMA_MODEL` | Default. No setup if `ollama serve` is running. |
| **Anthropic** | `anthropic` | anthropic | `ANTHROPIC_API_KEY` | `TESSERA_ANTHROPIC_MODEL` | Native SDK; supports prompt caching. |
| **Google Gemini** | `gemini` | gemini | `TESSERA_GEMINI_API_KEY` or `GOOGLE_API_KEY` | `TESSERA_GEMINI_MODEL` | Default `gemini-2.5-flash`. |
| **Cohere** | `cohere` | cohere | `COHERE_API_KEY` | `TESSERA_COHERE_MODEL` | Default `command-r-plus`. |
| **AWS Bedrock** | `bedrock` | bedrock (boto3) | AWS creds + region | `TESSERA_BEDROCK_MODEL` | Defaults to Claude 3.5 Sonnet. |
| **OpenAI** | `openai` | openai-compat | `OPENAI_API_KEY` | `TESSERA_OPENAI_MODEL` | Default `gpt-4o-mini`. |
| **Groq** | `groq` | openai-compat | `GROQ_API_KEY` | `TESSERA_GROQ_MODEL` | Fastest hosted inference. |
| **Together AI** | `together` | openai-compat | `TOGETHER_API_KEY` | `TESSERA_TOGETHER_MODEL` | Big open-model catalog. |
| **Fireworks** | `fireworks` | openai-compat | `FIREWORKS_API_KEY` | `TESSERA_FIREWORKS_MODEL` | Fast OSS inference. |
| **DeepSeek** | `deepseek` | openai-compat | `DEEPSEEK_API_KEY` | `TESSERA_DEEPSEEK_MODEL` | DeepSeek V3 / R1. |
| **xAI** | `xai` | openai-compat | `XAI_API_KEY` | `TESSERA_XAI_MODEL` | Grok family. |
| **OpenRouter** | `openrouter` | openai-compat | `OPENROUTER_API_KEY` | `TESSERA_OPENROUTER_MODEL` | One key for every model. |
| **Mistral** | `mistral` | openai-compat | `MISTRAL_API_KEY` | `TESSERA_MISTRAL_MODEL` | Mistral hosted. |
| **LM Studio (local)** | `lm_studio` | openai-compat | (none) | `TESSERA_LM_STUDIO_MODEL` | `localhost:1234`. |
| **vLLM (self-hosted)** | `vllm` | openai-compat | (none) | `TESSERA_VLLM_MODEL` | `localhost:8000`. |
| **llama.cpp (local)** | `llama_cpp` | openai-compat | (none) | `TESSERA_LLAMA_CPP_MODEL` | `localhost:8080`. |
| **Azure OpenAI** | `openai_compat` | openai-compat | `TESSERA_OPENAI_API_KEY` | `TESSERA_OPENAI_MODEL` | Set `TESSERA_OPENAI_BASE_URL` to your endpoint. |
| **Any other compat** | `openai_compat` | openai-compat | `TESSERA_OPENAI_API_KEY` | `TESSERA_OPENAI_MODEL` | Set `TESSERA_OPENAI_BASE_URL` to whatever. |

## Install the SDKs you need (all optional)

```bash
pip install openai                       # → openai, groq, together, fireworks, deepseek, xai, openrouter, mistral, lm_studio, vllm, llama_cpp
pip install anthropic                    # → anthropic
pip install google-generativeai          # → gemini
pip install cohere                       # → cohere
pip install boto3                        # → bedrock
```

Tessera has **zero hard LLM dependencies**. If the SDK isn't installed, the
backend uses a raw urllib fallback for OpenAI-compatible providers (no
streaming, no retries, but it works). Native-schema providers (Anthropic,
Gemini, Cohere, Bedrock) require their respective SDKs.

## Failure mode

If a backend's init fails (missing API key, unreachable host), Tessera
**falls back to the noop backend** with a warning. Your pipeline still runs;
LLM calls return deterministic stub strings. This keeps test suites green
even when no real provider is configured on the host.

## Discovering what's reachable

```bash
tessera providers          # lists every supported provider
tessera providers --check  # probes each one, reports reachable / not
```

## Speed knobs

| Env var | Effect |
|---|---|
| `TESSERA_NO_SEMANTIC_CACHE=1` | Disable semantic prompt cache. |
| `TESSERA_NO_VERIFY_CACHE=1`   | Disable AEON verify cache. |
| `TESSERA_NO_PARSE_CACHE=1`    | Disable Markdown parse cache. |
| `TESSERA_CACHE_DIR=<path>`    | Override cache directory (default `~/.cache/tessera/`). |

The semantic prompt cache uses `sentence-transformers` if installed (real
384-d embeddings, catches paraphrases), or falls back to a hashed bag of
tokens (only catches identical-or-near-identical prompts).
