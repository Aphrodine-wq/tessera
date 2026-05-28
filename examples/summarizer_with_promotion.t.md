---
agent: Summarizer
tessera_version: 0.2
capabilities_requested: []
max_cost: { dollars: 0.05, tokens: 5000 }
---

# Summarizer with skill promotion — decision 16 plumbing

A summarization agent declared as a procedural skill rather than a bare
prompt call. The skill carries `promote_to: neural { threshold: 3 }`,
which means: after the third call to `summarize`, the runtime emits a
one-shot `skill_promotion_pending` audit event into the operational
audit store. A future training job reads those events, assembles
(input, output) pairs from the audit corpus, kicks off a vast.ai
fine-tune, and swaps the skill binding to the new neural model.

This file only exercises the SIGNAL — the language now knows that the
skill is a candidate for promotion. Actual training is a follow-up
commit.

```
tessera compile examples/summarizer_with_promotion.t.md --run Summarizer \
  --set article="..."

tessera audit query --action skill_promotion_pending --count
```

```tsr:logic
fn brief(text: String) -> String = "summary: " + text
```

```tsr:prompt
prompt summarize_prompt(text: String) -> String = "Summarize: {text}"
```

```tsr:memory:procedural
procedural {
  skill summarize(text: String) -> String from prompt summarize_prompt promote_to: neural { threshold: 3 }
}
```

```tsr:agent
agent Summarizer {
  beliefs:
    @last_write article: String
  intentions:
    plan run {
      let s1 = summarize(article)
      let s2 = summarize(article)
      let s3 = summarize(article)
      let s4 = summarize(article)
      return s4
    }
}
```
