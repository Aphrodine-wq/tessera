---
agent: CodingAgent
capabilities_requested: []
max_cost: { dollars: 0.05, tokens: 2000 }
---

# Coding agent MVP — grammar-guaranteed bash tool-calling loop

The scoped MVP: `read_file` / `edit_file` / `bash` as real Tessera tools, a
contract guardrail refusing destructive bash commands before they ever run,
and a multi-step loop — not the single emit-and-return the wire examples show.

One real constraint shaped this design: plan bodies have no `if`/branching
(only straight-line statements + `until`), so a single loop iteration can't
dynamically choose between three different `emits=<tool>` prompts — `emits`
binds one prompt to one tool at compile time. The honest fix: the loop drives
`bash` alone (which can `cat`/`sed`/`grep` its way through read-and-edit work
same as a real shell), while `read_file` runs once directly up front to seed
context, and `edit_file` is a fully declared, tool-invokable, unit-tested tool
callable via the exact same direct-call path proven by `read_file` here — it
just isn't threaded into *this* particular loop's iteration.

```tsr:tool
tool read_file(path: String) -> String from tessera.adapters.coding.read_file
```

```tsr:tool
tool edit_file(path: String, old: String, new: String) -> String from tessera.adapters.coding.edit_file
```

```tsr:tool
tool bash(cmd: String) -> String from tessera.adapters.coding.bash
```

```tsr:contract
contract no_destructive_bash on tool:bash {
  before: not matches("rm -rf|push --force|force-with-lease|cat .env|.git/config")
  on_violation: refuse
}
```

```tsr:prompt emits=bash execute=true
prompt next_step(task: String, context: String) -> String = "Task: {task}\nWhat you've seen so far:\n{context}\n\nEmit exactly one line, a call record of the form:\n!bash #c1 cmd:<shell command>\nPick one shell command that makes progress on the task. Prefer read-only inspection (cat, ls, grep) unless the task requires a change."
```

```tsr:agent
agent CodingAgent {
  beliefs:
    @last_write task: String
    @last_write target_path: String

  intentions:
    plan do_task {
      let context = read_file(target_path)
      let steps = 0
      until steps >= 3 {
        let steps = steps + 1
        let result = next_step(task, context)
        let context = context + "\n---\n" + result
      }
      return context
    }
}
```

Run it against a local grammar-capable backend:

```bash
TESSERA_LLM_BACKEND=llamacpp TESSERA_WIRE_GGUF=/path/to/model.gguf \
  tessera compile examples/coding_agent_mvp.t.md --run CodingAgent \
  --set task="Summarize what this file does" --set target_path=README.md
```
