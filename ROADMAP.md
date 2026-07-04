# ContextWatch — Future Upgrade Ideas

ContextWatch today is an **observational profiler**: you wrap your client
(`wrap_anthropic`, `ContextWatchCallbackHandler`, or manual `record_turn`),
it produces a JSONL ledger, and `contextwatch report` renders a static HTML
view after the fact. That's deliberately passive — see the "observational
only" invariant in [`docs/architecture.md`](docs/architecture.md) §2.

The two ideas below — a **proxy server** and an **editor/Copilot-style
surface** — are the two directions that turn ContextWatch from "a report you
generate afterward" into "a thing that watches every call live and tells you
what to do about it." They're independent; either can ship without the
other, but the proxy is the one that makes the second possible in real time
rather than after a `contextwatch report` run.

---

## 1. Context Gateway — a transparent LLM proxy

### Why

The current interception layer requires *some* code change: wrap the client,
add a callback handler, or call `record_turn` manually. That's fine for one
codebase you control, but it doesn't scale to:
- polyglot stacks (a Python backend + a Node worker both calling the same
  model),
- third-party agent frameworks you don't want to monkeypatch,
- wanting visibility *before* you've decided to instrument anything.

A **proxy** sits between every caller and the provider's API. Point
`ANTHROPIC_BASE_URL` (or the OpenAI/Gemini equivalent) at it, and every call
gets profiled with zero code change — the same zero-rewrite philosophy the
SDK wrappers already have, just moved up a layer.

### Shape of it

```
┌──────────────┐        ┌─────────────────────────────┐        ┌──────────────┐
│  Any client  │──────▶ │   contextwatch-gateway        │──────▶ │  Anthropic /  │
│  (any lang)  │  HTTPS │   (local or hosted proxy)     │  HTTPS │  OpenAI / …   │
└──────────────┘        │                               │        └──────────────┘
                         │ 1. capture request messages   │
                         │ 2. forward unmodified (default)│
                         │ 3. capture response + usage    │
                         │ 4. profiler.record_turn(...)   │
                         │ 5. stream ledger to live UI    │
                         └───────────┬───────────────────┘
                                     ▼
                          same JSONL ledger + same
                          ContextBlock/Event model as today
```

Key design constraint: **the gateway reuses the existing core untouched.**
It's just a new entry into `profiler.record_turn()` — a `TurnRecord` doesn't
care whether it was produced by `wrap_anthropic` or an HTTP proxy sitting in
front of the same API. This keeps the ledger schema, report renderer, and
quality checker all as-is.

### Two modes

1. **Pass-through (default, safest).** Forward every request byte-identical,
   just observe. This is the only mode for a v1 — it preserves the
   "profiler never mutates messages" invariant from the architecture doc.
2. **Active (opt-in, v2+).** The gateway can *apply* a strategy before
   forwarding — e.g., auto-invoke Headroom compression on oversized tool
   results, or warn-and-block a request that's about to blow the context
   budget. This is a meaningfully bigger trust boundary (see Risks) and
   should ship behind an explicit flag, never as a default.

### Build blocks

- Minimal HTTP reverse proxy (e.g. Starlette/FastAPI or a small ASGI app) —
  paths mirror Anthropic's `/v1/messages`, OpenAI's `/v1/chat/completions`.
- Streaming support: providers stream SSE; the gateway needs to buffer/relay
  chunks without breaking the client's stream while still accumulating the
  full response for profiling after the stream ends.
- Provider adapters translate each API's request/response shape into the
  same `[{role, content}]` + `usage` shape `profiler.record_turn` expects —
  this is a generalization of what `wrap_anthropic_beta` already does for
  `context_management.applied_edits` / `stop_reason`.
- Session correlation: group calls into one "session" ledger. Options:
  a client-supplied header (`X-ContextWatch-Session`), or falling back to
  one ledger per client IP/API-key pair with a timeout-based session
  boundary.
- Auth passthrough: never terminate or log the real API key in plaintext;
  forward it unmodified, store only a hash for correlation if needed.

### Where this plugs into the existing package

- New subpackage: `contextwatch/gateway/` (keeps `contextwatch` core
  dependency-free; the gateway pulls in its own optional deps like
  `fastapi`/`httpx` under a `contextwatch[gateway]` extra).
- New CLI verb: `contextwatch gateway --port 8787 --upstream anthropic`.
- Ledger destination becomes pluggable (today: local JSONL file). For a
  gateway serving multiple concurrent sessions, JSONL-per-session on disk
  still works for v1; a real backing store (SQLite, then Postgres) is the
  natural v2 once multiple sessions/users need querying together.

---

## 2. Live surface — "Copilot for your context window"

### Why

The HTML report is retrospective — you run your agent, then generate a
report, then look at it. The valuable moment to see "you're about to burn
40% of your context on a stale tool result" is **while the agent is
running**, ideally inline in the editor or terminal you're already looking
at, the same way Copilot surfaces suggestions inline instead of in a
separate document you read afterward.

### Three surfaces, roughly in order of effort

**a. Terminal / TUI dashboard (cheapest, ship first)**
A `contextwatch watch run.jsonl` command that tails the ledger file (or
attaches to a running gateway) and renders a live-updating summary in the
terminal: current turn's token composition bar, last event, staleness
warnings. This needs no new UI framework — a curses/`rich`-based live table
over the same ledger tail the HTML report already consumes.

**b. Browser dev-tools-style panel (matches the existing design language)**
The static HTML report already looks like a DevTools panel. A live version
of the same thing, served locally by the gateway (or by a `contextwatch
watch --serve` command), auto-refreshing via a small polling/WS loop against
the ledger. This is the natural next step after the TUI — same
`report_template.html` composition-stream/turn-inspector components, just
fed a live-updating `TURNS` array instead of a one-shot JSON blob.

**c. Editor extension (VS Code / Cursor / JetBrains)**
A status-bar item showing live context usage (e.g. `Context: 62% · 1
stale block · 1 compaction`) for whichever agent session is currently
running, plus a sidebar that mirrors the turn inspector. This is the
"Copilot" analog the user is describing — it's the highest-effort surface
and depends on (a) or (b) already existing as a data source to poll, so it
should come last, not first.

### What "strategies to improve context" means concretely

This is the recommendation layer that makes any of the above worth looking
at, rather than just a fancier chart. Concretely, a `contextwatch.advisor`
module that runs against a `TurnRecord` (or a live stream of them) and
emits **structured suggestions**, not just observations:

| Signal already in the ledger | Suggested strategy |
|---|---|
| A `tool_result` block's `age` (turns since `first_seen_turn`) is large and it hasn't been referenced since | "Evict or compress block `{id}` — stale for {N} turns, costing {tokens} tok/turn" |
| A block's `tokens` is large and `source == tool_result` with low entropy / repeated structure (the same heuristic that made Headroom's SmartCrusher succeed in `ex7`) | "This tool result looks compressible — try Headroom before it enters context" |
| Client-estimated tokens diverge significantly from provider-reported `usage.input_tokens` over several turns | "Your tokenizer estimate is off by {pct}% — switch to the `tiktoken` extra for this model" |
| No `compaction`/`server_compaction` event has fired despite tokens approaching a configured budget | "Context is at {pct} of budget with no compaction strategy active — consider `SummarizationMiddleware` or Anthropic's `compact_20260112`" |
| A `compaction` event's quality score (from `quality.py`) is low | "Last compaction scored {score} — the summary lost information the agent needed; consider raising the summarization trigger threshold or protecting recent blocks" |
| Repeated near-identical `tool_result` blocks across turns (e.g. the same SQL query re-run) | "Cache this tool result — {N} duplicate calls in this session" |

Each rule reuses fields *already present* in `TurnRecord`/`ContextBlock`
(`age`, `tokens`, `source`, `first_seen_turn`, plus `quality.py`'s score) —
no new data collection is required, only a rules layer on top. This should
ship as a **pure function over the ledger** (`advise(turns) -> list[Suggestion]`)
so it works identically whether it's called from the CLI (`contextwatch
advise run.jsonl`), the live TUI, or the gateway's real-time stream.

---

## 3. Suggested order

1. **`contextwatch advise`** — pure analysis over an existing ledger, no new
   infra. Cheapest to ship, and it's the payload every other surface below
   needs anyway.
2. **`contextwatch watch` (TUI)** — tails a ledger and renders the same
   advisor output live. Proves out "live" without building a proxy yet;
   works today against any ledger a `wrap_anthropic`-instrumented app is
   already writing.
3. **Context Gateway (pass-through only)** — zero-instrumentation capture.
   Unlocks live profiling for anything that can set a base-URL env var,
   including stacks you don't control.
4. **Browser live panel** — same report UI, fed by the gateway or a watched
   ledger instead of a one-shot render.
5. **Editor extension** — once (3) and (4) exist as a stable local data
   source to poll, wrap it in a status-bar/sidebar extension.
6. **Active gateway mode (auto-remediation)** — only after the above are
   solid and trusted; this is the one place ContextWatch would start
   *mutating* traffic instead of observing it.

## Risks to carry forward from the architecture doc

- **Never break the agent.** Everything here is additive to a codebase that
  currently guarantees a profiler crash can't crash the agent
  (`strict=False`). The gateway raises the stakes: a proxy *bug* can break
  every call passing through it, not just profiling. Pass-through mode
  should fail open (forward the request even if profiling/logging throws).
- **Active mode is a trust boundary.** Auto-compressing or blocking real
  traffic means the gateway can now change what the model sees. This needs
  to be opt-in, auditable (log what was changed and why, same as the
  `reversible_evict` event already does for Headroom), and reversible by
  default — never silently lossy.
- **API keys flow through the gateway.** Treat this like any secrets-handling
  proxy: no plaintext logging of keys, no persistence beyond what's needed
  for request correlation.
- **Keep the core dependency-free.** Gateway and editor-extension work
  should live in separate optional packages/extras so `pip install
  contextwatch` stays stdlib-only for people who just want the SDK wrapper.
