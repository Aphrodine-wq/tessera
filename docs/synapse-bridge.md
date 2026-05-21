# Synapse Bridge — Tessera Κ Target

How Tessera's Κ (knowledge vault) compilation target writes into Synapse's graph.

## Synapse surface (as of 2026-05-21)

Synapse lives at `~/Projects/synapse/`. MCP server `synapse-mcp` exposes 8 tools (the README documents 7; `synapse_related` is undocumented but present).

| Tool | Used by Tessera? |
|---|---|
| `synapse_create` | Yes — one call per SIR node when batches < 100 |
| `synapse_link` | Yes — one call per data-dependency edge when batches < 100 |
| `synapse_log_trace` | Yes — once per compile, records parse→SIR→verify→emit visits |
| `synapse_search`, `synapse_get`, `synapse_related`, `synapse_recall_path`, `synapse_recent_activity` | Read-only, used by tooling not by the compiler |

Vault path: `~/Library/Application Support/Synapse/vault.sqlite` (GRDB).

## Write strategy: hybrid

| Batch size | Path | Why |
|---|---|---|
| < 100 nodes | MCP `synapse_create` + `synapse_link` | Enforces invariants, writes ChangeLog, auto-triggers ActivityEvents (the breathing canvas reflects compile in real time) |
| ≥ 100 nodes | Direct SQLite transaction on `vault.sqlite` | ~100–500× faster; manually populate `author_kind`, `author_id`, `hlc_timestamp`, `review_status` |
| Trace (1 per compile) | Always MCP `synapse_log_trace` | Centralizes audit trail; future replay UI depends on it |

## SIR → Synapse schema mapping

### Blocks

| SIR concept | Synapse Block fields |
|---|---|
| `agent FooAgent { ... }` | `title="agent:FooAgent"`, `tags=["tessera","agent"]`, `folder="Tessera Compile #N"`, `blockType=text`, `content=<rendered agent source>` |
| `intentions: plan find_papers { ... }` | `title="plan:find_papers"`, `tags=["tessera","intention"]`, edge from parent agent |
| `beliefs: @last_write topic: String` | `title="belief:topic"`, `tags=["tessera","belief","@last_write"]`, edge from parent agent |
| SIR node (e.g., `tsr.tool.invoke`) | `title="sir:Tool.Invoke"`, `tags=["tessera","sir-node"]`, `content=<textual SIR line>`, edge from enclosing region |

All Blocks get:
- `author = Author.agent("tessera-compiler")` (auto-creates on first call)
- `reviewStatus = .pending` (default — James approves or it stays pending)
- `embedding = None` (Synapse computes via NaturalLanguage on first read)

### Edges

| SIR concept | Edge fields |
|---|---|
| Data dependency (output of N1 → input of N2) | `edgeType=explicit`, `polarity=reinforcing`, `weight=0.7` |
| Effect-set conflict (e.g., `~pii.transmit` with no `TransmitPII` cap) | `polarity=inhibiting`, `weight=1.0` |
| Capability grant via `Spawn` | `polarity=reinforcing`, `weight=0.9`, `edgeType=pathway` |
| Region containment (region → child node) | `edgeType=implicit`, `polarity=neutral`, `weight=0.5` |

### Trace per compile

```python
synapse_log_trace(
    agent_id="tessera-compiler",
    summary=f"Tessera compile {trinity_hash} — {n_nodes} SIR nodes, {n_errors} errors",
    visits=[
        {"block_id": <parsed_module_block>, "note": "parsed N substrate blocks"},
        {"block_id": <sir_root>, "note": "emitted SIR DAG"},
        {"block_id": <verification_block>, "note": f"AEON: {n_errors} err / {n_warnings} warn"},
        ...
    ]
)
```

## What Tessera does NOT do to Synapse

- Does not call `ActivationSpreader` or `DreamProcessor` automatically. Those are James's tools, not the compiler's. (Future: an `eval:dream` kind could opt in.)
- Does not auto-approve. Every compile artifact starts `.pending` until James reviews in the Synapse UI.
- Does not delete blocks. Compile artifacts are immutable; subsequent compiles add new blocks under a new `folder="Tessera Compile #N+1"`.

## Tradeoffs we picked

- **Reuse Synapse's review queue as our approval flow.** No new UI needed — pending blocks already show up in the sidebar.
- **Hybrid MCP/direct-SQLite.** MCP is cleaner; direct SQLite is faster. Pick per batch size.
- **One folder per compile.** Lets James diff two compiles side-by-side and roll back to an earlier folder if needed.
