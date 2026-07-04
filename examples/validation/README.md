# ContextWatch Validation Examples — LangGraph & Anthropic Native Compaction

Runnable examples for validating ContextWatch against every real compaction
mechanism: LangChain `SummarizationMiddleware`, langmem `SummarizationNode`,
the manual `RemoveMessage` pattern, `trim_messages` (negative control),
Anthropic's **server-side** context editing (`clear_tool_uses_20250919`) and
**server-side compaction** (`compact_20260112`), and Headroom's
**reversible compression** ([chopratejas/headroom](https://github.com/chopratejas/headroom)).

> **Headroom is not compaction.** It compresses tool outputs/logs/files/RAG
> chunks and caches the original so the model can retrieve it later —
> nothing is deleted or summarized away. That's a third failure mode,
> distinct from lossy `compaction` and plain `evicted`, so it gets its own
> event type: `reversible_evict`. See `ex7_headroom_ccr.py` below.

> **Architecture note:** Anthropic's context editing and compaction run
> **server-side** — the client-side `messages` list never changes.
> ContextWatch's content-hash diffing at the model-call boundary sees
> *nothing* for these. `wrap_anthropic_beta` (in
> `contextwatch/integrations/anthropic_client.py`) instead reads
> `response.context_management.applied_edits` and
> `response.stop_reason == "compaction"`, surfacing them as two new event
> types: `server_edit` and `server_compaction`. See
> [`docs/architecture.md`](../../docs/architecture.md) §5/§9
> for the full design writeup.

## 0. Setup

```bash
pip install -e ./contextwatch                       # this package, editable
pip install "langchain>=1.1" langchain-anthropic langgraph langmem
export ANTHROPIC_API_KEY=sk-ant-...
```

Each `ex*.py` script is standalone and writes its own `<label>.jsonl` ledger
and `<label>-report.html` in the current directory. Run them from inside
this directory so `harness.py`/`tools.py` resolve as local imports:

```bash
cd examples/validation
python ex1_summarization_middleware.py
python ex2_langmem_node.py
python ex3_manual_removemessage.py            # HOOK_ENABLED=1 by default
EX3_HOOK=0 python ex3_manual_removemessage.py  # compare against no hook
python ex4_trim_negative_control.py
python ex5_anthropic_context_editing.py
python ex6_anthropic_server_compaction.py
python ex7_headroom_ccr.py                     # pip install "headroom-ai[all]" first
```

## Files

| File | Mechanism |
|---|---|
| `harness.py` | shared `make_profiler`/`finish` helpers |
| `tools.py` | shared "context bloater" tools (`fetch_report`, `run_sql`) |
| `ex1_summarization_middleware.py` | LangChain `SummarizationMiddleware` |
| `ex2_langmem_node.py` | langmem `SummarizationNode` in a raw `StateGraph` |
| `ex3_manual_removemessage.py` | hand-rolled `RemoveMessage` compaction node |
| `ex4_trim_negative_control.py` | `trim_messages` (negative control) |
| `ex5_anthropic_context_editing.py` | Anthropic `clear_tool_uses_20250919` |
| `ex6_anthropic_server_compaction.py` | Anthropic `compact_20260112` |
| `ex7_headroom_ccr.py` | Headroom reversible compression (CCR) |

## Validation checklist

| # | Mechanism | Where compaction runs | Detected via | Expected events |
|---|---|---|---|---|
| 1 | SummarizationMiddleware | client (middleware) | hash diff | `evicted` + `compaction` |
| 2 | langmem SummarizationNode | client (graph node) | hash diff | `evicted` + `compaction` |
| 3 | Manual RemoveMessage | client (your node) | hash diff + hook | `compaction` only with hook |
| 4 | trim_messages | client (deletion) | hash diff | `evicted`, **no** `compaction` |
| 5 | clear_tool_uses_20250919 | **server** | `applied_edits` in response | `server_edit` |
| 6 | compact_20260112 | **server** | `stop_reason` / usage drop | `server_compaction` |
| 7 | Headroom CCR | client (`compress()`, pre-append) | exact `tokens_before`/`tokens_after` | `reversible_evict` |

Pass criteria across all seven:

1. Every mechanism produces its expected event type, and *only* that type
2. §4 (`ex4`) produces zero compaction events (no false positives) — the
   script self-asserts this and prints a pass/fail line
3. §5/§6 show the client-estimate vs. provider-usage divergence in the
   report's "Client estimate vs. provider usage" panel
4. §3 demonstrates hook registration flipping detection on — run once with
   `EX3_HOOK=0` and once with the default, and diff the two reports
5. §7's `reversible_evict` events carry exact (not estimated) token counts,
   emit nothing when zero tokens are saved, and render with their own teal
   "retrievable" marker — never the rose "lossy" diamond used for
   `compaction`/`server_compaction`
6. Each run's HTML report is legible without explanation

## Note on running these in CI / sandboxed environments

Every script here makes real Anthropic API calls (directly or through
LangChain/LangGraph) and needs `ANTHROPIC_API_KEY` plus network access, and
`ex1`/`ex2` need `langchain>=1.1`, `langchain-anthropic`, `langgraph`, and
`langmem` installed; `ex7` additionally needs `headroom-ai[all]`. None of
that is available in a network-isolated sandbox, so these scripts are
provided as **runnable specs** rather than something exercised by `pytest`.

What *is* covered by the package's own test suite (`tests/test_profiler.py`,
no network or extra dependencies required) is the detection logic every
example depends on:

- `test_negative_control_trim_produces_no_compaction` — mirrors ex4's
  assertion directly against `ContextProfiler`
- `test_manual_marker_needs_hook_to_be_detected` — mirrors ex3's
  hook-on/hook-off comparison
- `test_server_edit_event_promoted_from_meta`,
  `test_server_compaction_event_promoted_from_meta` — the meta → event
  promotion that `wrap_anthropic_beta` relies on
- `test_wrap_anthropic_beta_captures_applied_edits`,
  `test_wrap_anthropic_beta_captures_server_compaction` — the wrapper
  itself, against a fake `client.beta.messages.create` (no network)
- `tests/test_headroom_adapter.py` — mirrors ex7 against a fake
  `compress_fn` (no `headroom` package or network needed): exact token
  accounting, no event when nothing was saved, and no confusion between
  `reversible_evict` and `compaction`

Run `pytest` from the package root to exercise these.
