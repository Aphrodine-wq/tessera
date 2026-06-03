---
agent: WeatherCaller
capabilities_requested: []
max_cost: { dollars: 0.01, tokens: 500 }
---

# Weather Caller — constrained tool-call emission (tson)

Demonstrates **tson constrained decoding**. The `tsr:prompt` block carries the
attribute `emits=get_weather`, which binds the prompt's output to the
`get_weather` tool's call grammar. On a grammar-capable backend (llama.cpp /
llama-server) the model can ONLY emit a structurally valid `!get_weather …`
record — malformed output is impossible. The prompt returns that record line.

Run it:

```
TESSERA_LLM_BACKEND=llamacpp TESSERA_WIRE_GGUF=/path/to/model.gguf \
  tessera compile examples/wire_tool.t.md --run WeatherCaller --set place="Oxford, MS"
```

The tool's typed parameters ARE the schema — `location`/`units` become required
fields of the call grammar. The tool is declared for its signature; this example
emits the call rather than invoking it.

```tsr:tool
tool get_weather(location: String, units: String) -> String from tessera.adapters.langchain._fallback_search
```

```tsr:prompt emits=get_weather
prompt weather_call(place: String) -> String = "Call get_weather for {place} in Fahrenheit. Emit exactly one record line: !get_weather #c1 location:<city> units:<f|c>"
```

```tsr:agent
agent WeatherCaller {
  beliefs:
    @last_write place: String

  intentions:
    plan call_weather {
      let record = weather_call(place)
      return record
    }
}
```
