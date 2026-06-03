---
agent: WeatherActor
capabilities_requested: []
max_cost: { dollars: 0.01, tokens: 500 }
---

# Weather Actor — close the loop (tson constrained emit → execute)

Like `wire_tool.t.md`, but the `tsr:prompt` block adds `execute=true`. Now the
model's grammar-constrained `!get_weather …` record is not just returned as
text — it is parsed and the `get_weather` tool is **dispatched**, and the plan
binds the tool's *result*. Constrained generation becomes a real agent action.

The dispatch is tagged `called_via=wire` in the audit trail; a call lifted from
untrusted text (`called_via=text`) is refused by default (instruction/data
boundary, PRD §12) unless a `tsr:autonomy` block lists the tool in
`allow_text_calls`.

```tsr:tool
tool get_weather(location: String, units: String) -> String from tessera.adapters.langchain._fallback_weather
```

```tsr:prompt emits=get_weather execute=true
prompt weather_call(place: String) -> String = "Call get_weather for {place} in Fahrenheit. Emit exactly one record line: !get_weather #c1 location:<city> units:<f|c>"
```

```tsr:agent
agent WeatherActor {
  beliefs:
    @last_write place: String

  intentions:
    plan act_weather {
      let report = weather_call(place)
      return report
    }
}
```
