# ContextLens 🔍

**Chrome DevTools' memory profiler, but for your LLM agent's context window.**

[![PyPI version](https://img.shields.io/pypi/v/contextlens.svg)](https://pypi.org/project/contextlens/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Every long-running agent eventually hits the same wall: the context window fills
up with stale tool results, the framework silently summarizes or drops messages,
recall quality degrades, and your input-token bill climbs — and you have **no
visibility into any of it**.

ContextLens intercepts your agent's LLM calls and produces a **per-turn token
ledger** of exactly what's inside the context window: what entered, what got
evicted, what got compacted, what's stale, and what it all costs. It then renders
everything into a **self-contained HTML report** you can open in any browser.

```
pip install contextlens
```

## The questions it answers

| Question | Where you see it |
|---|---|
| What fraction of my context is system prompt vs. tool results vs. conversation? | Composition stream chart |
| Which tool result from turn 3 is still burning 8K tokens at turn 20? | Stale burn analysis |
| Did my framework silently drop or summarize messages? When? | Eviction / compaction events |
| After compaction, can the summary still answer what the original could? | Compaction quality scoring |
| Is my client-side token estimate diverging from what the provider bills? | Client vs. provider divergence chart |
| Which source category drives my input-token bill? | Cost attribution |

## What makes it different

ContextLens detects **every kind of context mutation**, including ones most
tools can't see:

| Event | What happened | Detected via |
|---|---|---|
| `entered` | New blocks joined the context | content-hash diff |
| `evicted` | Blocks silently dropped (e.g. `trim_messages`) | content-hash diff |
| `compaction` | Blocks replaced by a summary (lossy) | hash diff + summary classification |
| `server_edit` | **Server-side** context editing (Anthropic `clear_tool_uses`) | `applied_edits` in the API response |
| `server_compaction` | **Server-side** compaction (Anthropic `compact`) | `stop_reason` / usage drop |
| `reversible_evict` | Reversible compression ([Headroom](https://github.com/chopratejas/headroom)) — content compressed but retrievable, **not** lost | exact `tokens_before`/`tokens_after` from the compressor |

Server-side mechanisms never touch your client-side `messages` list — hash
diffing alone sees *nothing*. ContextLens reads the API response metadata
instead, so nothing escapes the ledger.

## Quickstart

### Raw Anthropic SDK — one line

```python
from anthropic import Anthropic
from contextlens import ContextProfiler, wrap_anthropic

profiler = ContextProfiler("run.jsonl", label="text2sql")
client = wrap_anthropic(Anthropic(), profiler)
# ... run your agent exactly as before ...
```

Using Anthropic's server-side context management (context editing /
compaction)? Use `wrap_anthropic_beta` instead — it additionally captures
`server_edit` and `server_compaction` events from the response metadata.

### LangChain / LangGraph

```python
from contextlens import ContextProfiler
from contextlens.integrations.langchain_handler import ContextLensCallbackHandler

profiler = ContextProfiler("run.jsonl", label="text2sql")
handler = ContextLensCallbackHandler(profiler)
graph.invoke(state, config={"callbacks": [handler]})
```

Bonus: `langgraph_node` metadata gives you **per-node context attribution** —
see exactly which subagent is bloating the context.

### Headroom (reversible compression)

```python
from headroom import compress
from contextlens import ContextProfiler
from contextlens.integrations.headroom_adapter import wrap_headroom_compress

profiler = ContextProfiler("run.jsonl", label="my-agent")
headroom_compress = wrap_headroom_compress(compress, profiler)

result = headroom_compress(raw_tool_results, model="claude-sonnet-4-6")
# every compression that saves tokens is recorded as a `reversible_evict`
# event with exact (not estimated) token counts
```

### Framework-free — bring your own loop

```python
profiler = ContextProfiler("run.jsonl")
profiler.record_turn(messages, system=system_prompt, model=model_name,
                     usage={"input_tokens": ..., "output_tokens": ...})
```

The profiler only ever sees `[{role, content}]` — it works with any provider
and any framework.

## The report

```bash
contextlens report run.jsonl -o report.html
```

One self-contained HTML file (no server, no dependencies, works offline) with:

- **Composition stream** — stacked per-turn token bands by source (system /
  user / assistant / tool results / compaction summaries), with event markers:
  rose ◆ for lossy compaction, teal ↺ for reversible compression
- **Client vs. provider divergence** — your token estimate against what the
  API actually reported, per turn
- **Turn inspector** — click any turn to see every block, its age, size, and
  a content preview
- **Stale burn** — blocks that entered long ago and are still paying rent

## CLI

```bash
contextlens report run.jsonl -o report.html    # render standalone HTML
contextlens stats run.jsonl                    # summary stats as JSON
contextlens quality run.jsonl --model claude-haiku-4-5   # score compactions
```

`quality` replays probe questions against pre- and post-compaction context to
score how much recall the compaction actually destroyed.

## Try the demo (no API key needed)

```bash
python examples/demo_agent.py
open examples/demo-report.html
```

Simulates a 24-turn Text-to-SQL session: growing context, large tool results,
staleness, a mid-session compaction, and post-compaction growth.

## Validated against real mechanisms

The [`examples/validation/`](examples/validation/) directory contains seven
runnable scripts, one per real-world context-mutation mechanism:

| # | Mechanism | Expected events |
|---|---|---|
| 1 | LangChain `SummarizationMiddleware` | `evicted` + `compaction` |
| 2 | langmem `SummarizationNode` | `evicted` + `compaction` |
| 3 | Manual `RemoveMessage` pattern | `compaction` (with classifier hook) |
| 4 | `trim_messages` (negative control) | `evicted`, **no** `compaction` |
| 5 | Anthropic `clear_tool_uses` (server-side) | `server_edit` |
| 6 | Anthropic `compact` (server-side) | `server_compaction` |
| 7 | Headroom reversible compression | `reversible_evict` |

## Install

```bash
pip install contextlens                  # core — stdlib only, zero dependencies
pip install "contextlens[anthropic]"     # + Anthropic SDK wrapper
pip install "contextlens[langchain]"     # + LangChain/LangGraph callback
pip install "contextlens[tiktoken]"      # + exact tokenizer for OpenAI models
```

## Design principles

- **Zero-rewrite adoption.** One wrapper line around an existing client/graph.
- **Zero core dependencies.** The base install is pure stdlib.
- **Plain-JSONL ledger.** Any tool — pandas, jq, your dashboard — can consume it.
- **Model-agnostic core.** The profiler only ever sees `[{role, content}]`.
- **Exact where possible, honest where not.** Provider-reported `usage` for
  turn totals; per-block counts come from a pluggable tokenizer and are
  labeled as estimates.
- **Never breaks the agent.** All recording is wrapped in try/except
  (`strict=False` by default); a profiler crash never crashes your agent.

## Development

```bash
git clone https://github.com/sackri10/contextlens.git
cd contextlens
pip install -e ".[dev]"
pytest
```

See [`docs/architecture.md`](docs/architecture.md) for the full design spec —
data model, event detection, integration internals, and build order.

## License

MIT — see [LICENSE](LICENSE).
