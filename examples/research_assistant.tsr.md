---
agent: ResearchAssistant
capabilities_requested: [NetworkOut]
max_cost: { dollars: 0.10, tokens: 4000 }
---

# Research Assistant

A one-agent program that demonstrates the three new substrates:

- **`tsr:prompt`** — declares an LLM template (`reframe_question`, `summarize_findings`).
- **`tsr:tool`** — binds an external search tool (LangChain DuckDuckGo or a
  python fallback). The interpreter resolves the dotted import path lazily,
  so you can run this example without LangChain installed by swapping in any
  callable that takes a string and returns one.
- **`tsr:agent`** — orchestrates prompt + tool calls into a real task.

Run with:

```
TESSERA_LLM_BACKEND=ollama tessera compile examples/research_assistant.tsr.md \
    --run ResearchAssistant --set topic="how do construction subs typically get paid"
```

```tsr:prompt
prompt reframe_question(topic: String) -> String = "Reframe this research question into a precise web search query (one line, no quotes): {topic}"
```

```tsr:prompt
prompt summarize_findings(topic: String, context: String) -> String = "Topic: {topic}\n\nContext from web:\n{context}\n\nWrite a 3-sentence summary of what the context says about the topic."
```

```tsr:tool
tool web_search(query: String) -> String from tessera.adapters.langchain._fallback_search
```

```tsr:agent
agent ResearchAssistant {
  beliefs:
    @last_write topic: String

  intentions:
    plan answer {
      let query = reframe_question(topic)
      let context = web_search(query)
      let summary = summarize_findings(topic, context)
      return summary
    }
}
```
