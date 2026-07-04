# ContextLens — Context Window Profiler for LLM Agents

**Complete architecture, design, and implementation specification**

A pip-installable Python package that intercepts agent LLM calls, produces a per-turn token ledger of what's inside the context window, detects evictions and compactions, scores compaction quality, and renders a self-contained HTML report.

---

## 1. Product definition

**One-liner:** Chrome DevTools' memory profiler, but for an agent's context window.

**Problems it makes visible:**

1. What fraction of the context is system prompt vs. tool results vs. conversation, per turn
2. Stale payloads — tool results from turn 3 still burning 8K tokens at turn 20
3. Silent evictions — which blocks got dropped, when, and how many tokens they freed
4. Compaction quality — after a summarization/compaction, can the summary still answer questions the original context could? (the killer differentiator)
5. Cost attribution — which source category is driving the input-token bill

**Non-goals (v1):** tracing tool latency, output evals, multi-session aggregation, hosted SaaS. Local-first, single-session, zero-infra.

**Design principles:**

- **Zero rewrite adoption.** One wrapper line around an existing client/graph. No agent code changes.
- **Plain-dict ledger.** JSONL on disk. Any tool (pandas, jq, your dashboard) can consume it.
- **Model-agnostic core.** Anthropic and LangChain/LangGraph integrations are thin adapters; the profiler only sees `[{role, content}]`.
- **Exact where possible, honest where not.** Use provider-reported `usage` for turn totals; per-block counts come from a pluggable tokenizer and are labeled as estimates.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR AGENT (unchanged)                    │
│   LangGraph nodes / raw Anthropic SDK / LangChain chains         │
└──────────────┬──────────────────────────────────────────────────┘
               │ every LLM call passes the full message list
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  INTERCEPTION LAYER (integrations/)                              │
│  • wrap_anthropic(client, profiler)      — monkeypatch msgs.create│
│  • wrap_anthropic_beta(client, profiler) — same, + reads          │
│    response.context_management.applied_edits / stop_reason for    │
│    server-side context editing & compaction (see §5 note, §9.3)   │
│  • ContextLensCallbackHandler        — LangChain callback        │
│  • profiler.record_turn(...)         — manual, framework-free    │
└──────────────┬──────────────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CORE PROFILER (profiler.py)                                     │
│  1. Normalize messages → ContextBlocks (stable hash identity)    │
│  2. Classify each block's source (heuristics + user hooks)       │
│  3. Count tokens per block (pluggable tokenizer)                 │
│  4. Diff against previous turn → entered / evicted / compaction  │
│  5. Promote meta flags → server_edit / server_compaction events  │
│  6. Track block age (first_seen_turn)                            │
│  7. Emit TurnRecord → JSONL sink                                 │
└──────┬───────────────────────────────────┬──────────────────────┘
       ▼                                   ▼
┌───────────────────┐          ┌────────────────────────────────┐
│  LEDGER (JSONL)   │          │  QUALITY CHECKER (quality.py)  │
│  session header    │          │  probes pre-compaction context │
│  + one line/turn   │          │  against summary via cheap LLM │
└──────┬────────────┘          └──────────────┬─────────────────┘
       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  REPORT (report.py + cli.py)                                     │
│  `contextlens report session.jsonl` → single self-contained     │
│  HTML: stacked composition chart, eviction markers, turn         │
│  inspector table, staleness heatmap, quality scores,              │
│  client-estimate-vs-provider-usage divergence panel               │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow invariant:** the profiler is *observational only*. It never mutates messages, never adds latency-critical work inline (token counting is cheap heuristics by default), and a profiler crash never breaks the agent (all recording wrapped in try/except with a `strict=False` default).

**Server-side context management note:** Anthropic's context editing (`clear_tool_uses_20250919`) and compaction (`compact_20260112`) run **server-side** — the client-side `messages` list never changes, so the profiler's content-hash diffing (step 4 above) sees nothing for these. `wrap_anthropic_beta` reads `response.context_management.applied_edits` and `response.stop_reason == "compaction"` instead and passes them through `meta`; the profiler promotes them to `server_edit` / `server_compaction` events (step 5). This was validated against real API calls in `examples/validation/ex5_anthropic_context_editing.py` and `ex6_anthropic_server_compaction.py` — see §9.3.

---

## 3. Package layout

```
contextlens/
├── pyproject.toml
├── README.md
├── src/contextlens/
│   ├── __init__.py            # public API: ContextProfiler, wrap_anthropic, ...
│   ├── models.py              # ContextBlock, TurnRecord, Event, SessionHeader
│   ├── tokenizers.py          # pluggable token counters
│   ├── classify.py            # source classification heuristics + hooks
│   ├── profiler.py            # ContextProfiler core
│   ├── quality.py             # CompactionQualityChecker
│   ├── report.py              # JSONL → standalone HTML
│   ├── cli.py                 # `contextlens report|stats`
│   └── integrations/
│       ├── __init__.py
│       ├── anthropic_client.py    # wrap_anthropic + wrap_anthropic_beta
│       ├── langchain_handler.py
│       └── headroom_adapter.py    # wrap_headroom_compress (reversible compression)
├── examples/
│   ├── demo_agent.py          # simulated 24-turn session with a compaction
│   └── validation/            # real-mechanism validation suite (needs API key)
│       ├── harness.py, tools.py
│       ├── ex1_summarization_middleware.py
│       ├── ex2_langmem_node.py
│       ├── ex3_manual_removemessage.py
│       ├── ex4_trim_negative_control.py
│       ├── ex5_anthropic_context_editing.py
│       ├── ex6_anthropic_server_compaction.py
│       └── ex7_headroom_ccr.py
└── tests/
    ├── test_profiler.py
    └── test_headroom_adapter.py
```

Dependency policy: **core has zero required dependencies** (stdlib only). Optional extras: `contextlens[anthropic]`, `contextlens[langchain]`, `contextlens[tiktoken]`.

---

## 4. Ledger schema (the contract everything shares)

JSONL, one JSON object per line.

**Line 0 — session header:**

```json
{"kind": "session", "schema": 1, "session_id": "a1b2c3d4e5",
 "started_at": 1751500000.0, "label": "text2sql-run-42",
 "tokenizer": "heuristic-chars4"}
```

**Line N — turn record:**

```json
{
  "kind": "turn",
  "turn": 7,
  "ts": 1751500042.5,
  "model": "claude-sonnet-4-6",
  "blocks": [
    {"id": "9f2c1ab0d3e4", "index": 0, "role": "system", "source": "system",
     "tokens": 1840, "chars": 7360, "first_seen_turn": 1, "age": 6,
     "preview": "You are a SQL analyst agent...", "tool_name": null},
    {"id": "b7e...", "index": 5, "role": "user", "source": "tool_result",
     "tokens": 3120, "chars": 12480, "first_seen_turn": 4, "age": 3,
     "preview": "[tool_result] rows: [...]", "tool_name": "run_sql"}
  ],
  "events": [
    {"type": "entered", "block_ids": ["b7e..."], "tokens": 3120, "detail": ""},
    {"type": "evicted", "block_ids": ["01aa..", "02bb.."], "tokens": 4200,
     "detail": "2 blocks left context"}
  ],
  "totals": {"tokens": 14350, "blocks": 9,
             "by_source": {"system": 1840, "tool_result": 7900,
                            "assistant": 2400, "user": 2210}},
  "usage": {"input_tokens": 14512, "output_tokens": 380},
  "meta": {"node": "sql_writer"}
}
```

Key decisions:

- **`usage` vs `totals.tokens`:** `usage` is the provider's exact count when available; `totals` is the sum of per-block estimates. The report shows both, and the ratio is itself a diagnostic (a big gap means your tokenizer estimate or content normalization is off) — **or, once server-side context management is in play, the gap is the signal itself** (see below).
- **Block identity = `sha1(role + normalized_content)[:12]`.** Content-addressed identity is what makes cross-turn diffing trivial and robust: identical block → same id → "still present"; missing id → eviction; new id → entry. A block whose text is edited in place is correctly modeled as evict + enter.
- **`age`** enables the staleness heatmap: `tokens × age` is the "stale burn" metric — the single most shareable number the tool produces ("turn-3 tool output has cost you 47K cumulative input tokens").

**Event types:** `entered`, `evicted`, `compaction` are all detected client-side by hash-diffing `blocks` across turns (§8). Two more are detected purely from response metadata, never from the diff:

- **`server_edit`** — Anthropic cleared old tool results server-side (`clear_tool_uses_20250919`). `detail` carries the clear-strategy type and tool-use count; `tokens` is `cleared_input_tokens` from `context_management.applied_edits`.
- **`server_compaction`** — Anthropic compacted the conversation server-side (`compact_20260112`), detected via `response.stop_reason == "compaction"`.

Both are produced by `wrap_anthropic_beta` (§9.3) and promoted from `meta.server_edits` / `meta.server_compaction` inside `profiler._record` — the client `blocks` list for these turns shows **no evictions**, because nothing on the client side ever changed. That's the whole point: without this second detection path, server-side management would be completely invisible to the ledger.

A sixth event type covers a fundamentally different failure mode:

- **`reversible_evict`** — reversible compression (e.g. [Headroom](https://github.com/chopratejas/headroom)) shrank a tool output/log/file/RAG chunk without deleting or summarizing it; the original is cached and retrievable. `Event.type` is already a free-form string, so this needed no schema change. `tokens` is the **exact** savings (`tokens_before - tokens_after` from the compressor's own accounting, never estimated); `detail` follows the convention `"headroom compressed {before}->{after} tok ({ratio:.0%} reduction); retrievable via headroom_retrieve"`.

`reversible_evict` must never be conflated with `compaction`/`server_compaction`: those are lossy (the original text is gone, replaced by a summary); `reversible_evict` is not (the original is cached, just not in the current context window). The report (§11) gives it a distinct teal marker and a "retrievable" badge specifically so a reader can't mistake one for the other. See §9.4 for the adapter that produces it.

---

## 5. Core models (`models.py`)

```python
"""Data models for the ContextLens ledger."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SCHEMA_VERSION = 1

SOURCES = (
    "system", "user", "assistant", "assistant_action",
    "tool_result", "compaction_summary", "retrieval", "other",
)


def _normalize_content(content: Any) -> str:
    """Flatten message content (str or Anthropic-style block list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    parts.append(
                        f"[tool_use:{block.get('name','?')}] "
                        + json.dumps(block.get("input", {}), sort_keys=True, default=str)
                    )
                elif btype == "tool_result":
                    parts.append(
                        f"[tool_result:{block.get('tool_use_id','?')}] "
                        + _normalize_content(block.get("content"))
                    )
                else:
                    parts.append(json.dumps(block, sort_keys=True, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def block_id(role: str, content: Any) -> str:
    """Stable identity for a message across turns: hash(role + content)."""
    text = _normalize_content(content)
    return hashlib.sha1(f"{role}\x00{text}".encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class ContextBlock:
    id: str
    index: int
    role: str
    source: str
    tokens: int
    chars: int
    first_seen_turn: int
    age: int                      # turns since entry; 0 = entered this turn
    preview: str                  # first ~140 chars for the report UI
    tool_name: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    type: str                     # "entered" | "evicted" | "compaction"
    block_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnRecord:
    kind: str
    turn: int
    ts: float
    model: Optional[str]
    blocks: list[ContextBlock]
    events: list[Event]
    totals: dict
    usage: Optional[dict]
    meta: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionHeader:
    kind: str
    schema: int
    session_id: str
    started_at: float
    label: str
    tokenizer: str

    @staticmethod
    def new(label: str, tokenizer: str) -> "SessionHeader":
        return SessionHeader(
            kind="session", schema=SCHEMA_VERSION,
            session_id=uuid.uuid4().hex[:10],
            started_at=time.time(), label=label, tokenizer=tokenizer,
        )

    def to_dict(self) -> dict:
        return asdict(self)
```

---

## 6. Tokenizers (`tokenizers.py`)

Pluggable `Callable[[str], int]`. Ship three:

```python
"""Pluggable token counting. Exactness ladder:
heuristic (free, ~±15%) → tiktoken (fast, GPT-family exact) →
anthropic count_tokens API (exact, network call — off the hot path only).
"""

from __future__ import annotations
from typing import Callable, Protocol

TokenCounter = Callable[[str], int]


def heuristic_counter(text: str) -> int:
    """chars/4 blended with words*1.3 — good enough for composition ratios."""
    if not text:
        return 0
    by_chars = len(text) / 4.0
    by_words = len(text.split()) * 1.3
    return max(1, int((by_chars + by_words) / 2))


def tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter:
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text)) if text else 0


def anthropic_counter(client, model: str) -> TokenCounter:
    """Exact via the count_tokens endpoint. Use for offline re-scoring of a
    ledger (`contextlens rescore`), never inline in the agent loop."""
    def count(text: str) -> int:
        if not text:
            return 0
        r = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}])
        return r.input_tokens
    return count
```

Rationale: composition percentages and trends are insensitive to ±15% error, so the free heuristic is the default and adds ~0 latency. Exact per-block numbers are an offline enrichment step, keeping the hot path pure.

---

## 7. Source classification (`classify.py`)

Heuristics first, hooks for the rest:

```python
"""Classify a message into a source category.

Order of precedence:
1. user-registered classifier hook (returns a source string or None)
2. structural heuristics (content block types)
3. compaction markers (registered prefixes)
4. role fallback
"""

from __future__ import annotations
from typing import Any, Callable, Optional

Classifier = Callable[[dict, int], Optional[str]]

DEFAULT_COMPACTION_MARKERS = (
    "[conversation summary]", "summary of prior conversation",
    "<compacted>", "previous context (summarized)",
)


def classify(message: dict, index: int, *,
             hooks: list[Classifier] = (),
             compaction_markers: tuple[str, ...] = DEFAULT_COMPACTION_MARKERS,
             ) -> tuple[str, Optional[str]]:
    """Returns (source, tool_name)."""
    for hook in hooks:
        result = hook(message, index)
        if result:
            return result, None

    role = message.get("role", "other")
    content = message.get("content")

    if role == "system":
        return "system", None

    # Anthropic-style content block lists
    if isinstance(content, list):
        types = {b.get("type") for b in content if isinstance(b, dict)}
        if "tool_result" in types:
            return "tool_result", _first_tool_id(content)
        if "tool_use" in types:
            return "assistant_action", _first_tool_name(content)

    text = content if isinstance(content, str) else ""
    lowered = text[:200].lower()
    if any(m in lowered for m in compaction_markers):
        return "compaction_summary", None

    if role in ("user", "assistant"):
        return role, None
    return "other", None


def _first_tool_name(blocks) -> Optional[str]:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            return b.get("name")
    return None


def _first_tool_id(blocks) -> Optional[str]:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return b.get("tool_use_id")
    return None
```

Users with custom compaction logic register a one-line hook:

```python
profiler.add_classifier(lambda msg, i:
    "compaction_summary" if msg.get("content", "").startswith("MEMORY:") else None)
```

---

## 8. Core profiler (`profiler.py`)

The heart of the package. Full implementation:

```python
"""ContextProfiler: turn message lists into a per-turn context ledger."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .models import (ContextBlock, Event, SessionHeader, TurnRecord,
                     _normalize_content, block_id)
from .classify import classify, Classifier
from .tokenizers import TokenCounter, heuristic_counter

# Heuristic: a compaction = several blocks evicted AND a summary-ish block
# entering in the same turn, with net token reduction.
COMPACTION_MIN_EVICTED = 3


class ContextProfiler:
    def __init__(
        self,
        sink: str | Path | None = "contextlens.jsonl",
        *,
        label: str = "session",
        token_counter: TokenCounter = heuristic_counter,
        tokenizer_name: str = "heuristic-chars4",
        strict: bool = False,          # False: never let profiling break the agent
        preview_chars: int = 140,
    ):
        self._counter = token_counter
        self._strict = strict
        self._preview_chars = preview_chars
        self._hooks: list[Classifier] = []
        self._lock = threading.Lock()

        self._turn = 0
        self._first_seen: dict[str, int] = {}     # block_id -> turn entered
        self._prev_ids: set[str] = set()
        self._prev_tokens: dict[str, int] = {}    # for eviction token accounting
        self.records: list[TurnRecord] = []       # in-memory copy

        self._fh = None
        if sink is not None:
            self._fh = open(Path(sink), "a", encoding="utf-8")
        self._header = SessionHeader.new(label=label, tokenizer=tokenizer_name)
        self._emit(self._header.to_dict())

    # -- public API --------------------------------------------------------

    def add_classifier(self, hook: Classifier) -> None:
        self._hooks.append(hook)

    def record_turn(
        self,
        messages: list[dict],
        *,
        system: Any = None,            # Anthropic passes system separately
        model: Optional[str] = None,
        usage: Optional[dict] = None,  # {"input_tokens":.., "output_tokens":..}
        meta: Optional[dict] = None,
    ) -> Optional[TurnRecord]:
        try:
            with self._lock:
                return self._record(messages, system, model, usage, meta or {})
        except Exception:
            if self._strict:
                raise
            return None

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals ----------------------------------------------------------

    def _record(self, messages, system, model, usage, meta) -> TurnRecord:
        self._turn += 1
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        blocks: list[ContextBlock] = []
        current_ids: set[str] = set()
        by_source: dict[str, int] = {}
        cur_tokens: dict[str, int] = {}

        for i, msg in enumerate(full):
            role = msg.get("role", "other")
            content = msg.get("content")
            text = _normalize_content(content)
            bid = block_id(role, content)
            source, tool_name = classify(msg, i, hooks=self._hooks)
            tokens = self._counter(text)

            first = self._first_seen.setdefault(bid, self._turn)
            blocks.append(ContextBlock(
                id=bid, index=i, role=role, source=source,
                tokens=tokens, chars=len(text),
                first_seen_turn=first, age=self._turn - first,
                preview=text[: self._preview_chars],
                tool_name=tool_name,
            ))
            current_ids.add(bid)
            cur_tokens[bid] = tokens
            by_source[source] = by_source.get(source, 0) + tokens

        # ---- diff against previous turn ----
        entered = current_ids - self._prev_ids
        evicted = self._prev_ids - current_ids
        events: list[Event] = []
        if entered:
            events.append(Event(
                type="entered", block_ids=sorted(entered),
                tokens=sum(cur_tokens[b] for b in entered)))
        if evicted:
            events.append(Event(
                type="evicted", block_ids=sorted(evicted),
                tokens=sum(self._prev_tokens.get(b, 0) for b in evicted),
                detail=f"{len(evicted)} blocks left context"))

        # ---- compaction detection ----
        summary_entered = [b for b in blocks
                           if b.id in entered and b.source == "compaction_summary"]
        if len(evicted) >= COMPACTION_MIN_EVICTED and summary_entered:
            freed = sum(self._prev_tokens.get(b, 0) for b in evicted)
            added = sum(b.tokens for b in summary_entered)
            events.append(Event(
                type="compaction",
                block_ids=[b.id for b in summary_entered],
                tokens=freed - added,
                detail=(f"compacted {len(evicted)} blocks ({freed} tok) into "
                        f"{len(summary_entered)} summary ({added} tok); "
                        f"ratio {added / max(freed, 1):.2f}")))

        record = TurnRecord(
            kind="turn", turn=self._turn, ts=time.time(), model=model,
            blocks=blocks, events=events,
            totals={"tokens": sum(cur_tokens.values()),
                    "blocks": len(blocks), "by_source": by_source},
            usage=usage, meta=meta,
        )
        self.records.append(record)
        self._emit(record.to_dict())
        self._prev_ids = current_ids
        self._prev_tokens = cur_tokens
        return record

    def _emit(self, obj: dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()
```

Design notes worth stating in the README:

- **Thread-safe** (lock around record) because LangGraph parallel branches can fire concurrent LLM calls; turn numbering is then "call order," which is the honest thing to report.
- **`strict=False` default** — observability must never take down the agent. Flip to `True` in tests.
- **Flush per line** — a crashed agent still leaves a valid ledger up to the crash, which is exactly when you want the ledger.

---

## 9. Integrations (`integrations/`)

### 9.1 Anthropic SDK — one line adoption

```python
"""wrap_anthropic(client, profiler): profile every messages.create call."""

from __future__ import annotations
import functools


def wrap_anthropic(client, profiler, *, meta_fn=None):
    """Monkeypatch client.messages.create to record each call.

    Records the *request* context (what the model actually saw) plus the
    response usage (exact input/output token counts).
    """
    original = client.messages.create

    @functools.wraps(original)
    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        profiler.record_turn(
            kwargs.get("messages", []),
            system=kwargs.get("system"),
            model=kwargs.get("model"),
            usage={"input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens} if usage else None,
            meta=meta_fn(kwargs) if meta_fn else None,
        )
        return response

    client.messages.create = create
    return client
```

Usage:

```python
from anthropic import Anthropic
from contextlens import ContextProfiler, wrap_anthropic

profiler = ContextProfiler("run.jsonl", label="text2sql")
client = wrap_anthropic(Anthropic(), profiler)
# ... run your agent exactly as before ...
```

Streaming variant: also patch `messages.stream` — record on `__enter__` with usage backfilled from the final message event. (v1.1; the pattern is identical.)

### 9.2 LangChain / LangGraph — callback handler

```python
"""ContextLensCallbackHandler: attach to any LangChain/LangGraph runnable."""

from __future__ import annotations
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
except ImportError as e:
    raise ImportError("pip install 'contextlens[langchain]'") from e


def _to_dict(msg: "BaseMessage") -> dict:
    role = {"human": "user", "ai": "assistant", "system": "system",
            "tool": "user"}.get(msg.type, msg.type)
    content = msg.content
    # LangChain ToolMessage → represent as a tool_result block so
    # classification works identically to the raw-SDK path.
    if msg.type == "tool":
        content = [{"type": "tool_result",
                    "tool_use_id": getattr(msg, "tool_call_id", "?"),
                    "content": msg.content}]
    return {"role": role, "content": content}


class ContextLensCallbackHandler(BaseCallbackHandler):
    def __init__(self, profiler):
        self.profiler = profiler

    def on_chat_model_start(self, serialized: dict,
                            messages: list[list["BaseMessage"]],
                            **kwargs: Any) -> None:
        model = (serialized or {}).get("kwargs", {}).get("model")
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        for batch in messages:
            self.profiler.record_turn(
                [_to_dict(m) for m in batch],
                model=model,
                meta={"node": node} if node else {},
            )
```

Usage with your Text-to-SQL LangGraph app:

```python
handler = ContextLensCallbackHandler(profiler)
graph.invoke(state, config={"callbacks": [handler]})
```

`langgraph_node` in metadata gives you **per-node context attribution** — which subagent (Sales vs Orders) is bloating context — for free in the report.

### 9.3 Anthropic beta SDK — server-side context management

Context editing and compaction are **server-side** features (beta headers `context-management-2025-06-27` and `compact-2026-01-12`): the client's `messages` list is never mutated, so `wrap_anthropic`'s hash-diffing has nothing to diff. `wrap_anthropic_beta` reads the *response* instead:

```python
# contextlens/integrations/anthropic_client.py — wrap_anthropic_beta

def wrap_anthropic_beta(client, profiler, *, meta_fn=None):
    """Wrap client.beta.messages.create; captures server-side context edits."""
    original = client.beta.messages.create

    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        meta = dict(meta_fn(kwargs) if meta_fn else {})

        cm = getattr(response, "context_management", None)
        applied = list(getattr(cm, "applied_edits", []) or [])
        if applied:
            meta["server_edits"] = [
                {"type": getattr(e, "type", "?"),
                 "cleared_tool_uses": getattr(e, "cleared_tool_uses", None),
                 "cleared_input_tokens": getattr(e, "cleared_input_tokens", None)}
                for e in applied
            ]
        if getattr(response, "stop_reason", None) == "compaction":
            meta["server_compaction"] = True

        profiler.record_turn(
            kwargs.get("messages", []),
            system=kwargs.get("system"),
            model=kwargs.get("model"),
            usage={"input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens} if usage else None,
            meta=meta,
        )
        return response

    client.beta.messages.create = create
    return client
```

`ContextProfiler._record` promotes these `meta` flags to first-class `server_edit` / `server_compaction` events (§4, §8) right after the client-side compaction-detection block, using the same `Event` model as everything else — the report code doesn't need to special-case them beyond drawing them (§11).

Usage:

```python
import anthropic
from contextlens import ContextProfiler, wrap_anthropic_beta

profiler = ContextProfiler("run.jsonl", label="clear-tool-uses")
client = wrap_anthropic_beta(anthropic.Anthropic(), profiler)

resp = client.beta.messages.create(
    model="claude-sonnet-4-6", max_tokens=1024,
    betas=["context-management-2025-06-27"],
    context_management={"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 8000},
        "keep": {"type": "tool_uses", "value": 2},
    }]},
    messages=messages,
)
```

Validated end-to-end (real API calls, no mocking) in `examples/validation/ex5_anthropic_context_editing.py` (context editing) and `ex6_anthropic_server_compaction.py` (server-side compaction); the wrapper's response-parsing logic itself is unit-tested against a fake `client.beta.messages.create` in `tests/test_profiler.py` (`test_wrap_anthropic_beta_captures_applied_edits`, `test_wrap_anthropic_beta_captures_server_compaction`) so it doesn't need network access to catch regressions.

### 9.4 Headroom — reversible compression (`headroom_adapter.py`)

[Headroom](https://github.com/chopratejas/headroom) compresses tool outputs/logs/files/RAG chunks via `compress(messages, model=...)`, returning a `CompressResult` with exact `tokens_before`/`tokens_after`, and caches the original so the model can retrieve it later. Unlike every other integration in this doc, there's no hash-diffing involved — Headroom reports the exact numbers itself, so none of the profiler's estimation machinery (`classify.py`, `tokenizers.py`) is touched:

```python
"""Wrap headroom.compress() so every call is profiled before the compressed
messages go to the model. Headroom reports tokens_before/after directly --
no need to infer compression from a hash diff, unlike every other integration."""

from __future__ import annotations
from contextlens.models import Event


def wrap_headroom_compress(compress_fn, profiler):
    """compress_fn: headroom.compress. Returns a wrapped callable with the
    same signature that also queues a `reversible_evict` event, which is
    flushed into the next record_turn() call on this profiler."""

    def compress(messages, **kwargs):
        result = compress_fn(messages, **kwargs)
        saved = result.tokens_before - result.tokens_after
        if saved > 0:
            ratio = saved / max(result.tokens_before, 1)
            profiler._pending_events = getattr(profiler, "_pending_events", [])
            profiler._pending_events.append(Event(
                type="reversible_evict",
                tokens=saved,
                detail=(f"headroom compressed {result.tokens_before}->"
                        f"{result.tokens_after} tok ({ratio:.0%} reduction); "
                        f"retrievable via headroom_retrieve"),
            ))
        return result

    return compress
```

**Why queueing, not direct promotion:** `compress()` runs on the raw tool output *before* it's appended to `messages` — it's called outside of `record_turn()` entirely, so it has no `TurnRecord` to attach an event to yet. It stashes the event on `profiler._pending_events` instead; `ContextProfiler._record` flushes the queue into `events` on the very next `record_turn()` call, right after the server-side event promotion (§8):

```python
# profiler._record(), immediately before `record = TurnRecord(...)`:
events.extend(getattr(self, "_pending_events", []))
self._pending_events = []
```

No other change to `_record`, `classify.py`, or `tokenizers.py` was needed — this is the smallest integration in the codebase precisely because Headroom already does the hard part (exact token accounting) itself.

Usage:

```python
from headroom import compress
from contextlens import ContextProfiler
from contextlens.integrations.headroom_adapter import wrap_headroom_compress

profiler = ContextProfiler("run.jsonl", label="headroom-ccr")
compress = wrap_headroom_compress(compress, profiler)

compressed = compress(raw_tool_results, model="claude-sonnet-4-6")
messages.append({"role": "user", "content": compressed.messages})
profiler.record_turn(messages, model="claude-sonnet-4-6", usage=usage)  # picks up the queued event
```

Validated in `examples/validation/ex7_headroom_ccr.py` (real Anthropic + Headroom calls) and unit-tested against a fake `compress_fn` in `tests/test_headroom_adapter.py` — no network or `headroom` package needed to catch regressions in the queueing/flush logic itself.

---

## 10. Compaction quality checker (`quality.py`)

The differentiator. Given the evicted text and the summary that replaced it, generate probe questions from the original, answer them from the summary alone, grade.

```python
"""CompactionQualityChecker: does the summary still answer what the
original context could? Probe-based scoring with any LLM callable."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable

LLM = Callable[[str], str]   # prompt in, text out — bring any model

PROBE_PROMPT = """From the following conversation context, write {n} short
factual questions that this context answers, with their answers.
Respond ONLY with JSON: [{{"q": "...", "a": "..."}}].

CONTEXT:
{context}"""

ANSWER_PROMPT = """Answer the question using ONLY the summary below.
If the summary does not contain the answer, reply exactly: UNKNOWN.

SUMMARY:
{summary}

QUESTION: {question}
Answer in one sentence."""

GRADE_PROMPT = """Reference answer: {ref}
Candidate answer: {cand}
Is the candidate factually consistent with the reference? Reply ONLY "yes" or "no"."""


@dataclass
class QualityReport:
    score: float                 # 0..1 — fraction of probes survived
    n_probes: int
    failures: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {"score": self.score, "n_probes": self.n_probes,
                "failures": self.failures}


def _parse_json(text: str):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text.strip())


class CompactionQualityChecker:
    def __init__(self, llm: LLM, n_probes: int = 5):
        self.llm, self.n = llm, n_probes

    def check(self, original_text: str, summary_text: str) -> QualityReport:
        probes = _parse_json(self.llm(
            PROBE_PROMPT.format(n=self.n, context=original_text[:20000])))
        failures, passed = [], 0
        for p in probes[: self.n]:
            cand = self.llm(ANSWER_PROMPT.format(
                summary=summary_text, question=p["q"])).strip()
            if cand.upper().startswith("UNKNOWN"):
                ok = False
            else:
                ok = self.llm(GRADE_PROMPT.format(
                    ref=p["a"], cand=cand)).strip().lower().startswith("yes")
            if ok:
                passed += 1
            else:
                failures.append({"q": p["q"], "expected": p["a"], "got": cand})
        n = max(len(probes[: self.n]), 1)
        return QualityReport(score=passed / n, n_probes=n, failures=failures)
```

Wiring it to the ledger (offline enrichment — never in the hot loop):

```python
# contextlens quality run.jsonl --model claude-haiku-4-5
# For each "compaction" event: reconstruct evicted text from the last turn
# where those block ids were present, take the summary blocks' text,
# run checker.check(), append a {"kind": "quality", "turn": N, ...} line.
```

The report then shows, at each compaction marker: **"Quality 0.6 — lost: 'what filter did the user request for Q3?'"** That screenshot is your launch post.

Cost control: probes run on a cheap model (Haiku-class); 5 probes ≈ a few thousand tokens per compaction.

---

## 11. Report (`report.py`) — the demo surface

Single self-contained HTML file (data embedded as JSON, zero network calls, opens from disk, shareable as an attachment). Vanilla JS + inline SVG — no build step.

**Visual design (deliberate, not default):** a light "instrument panel" aesthetic — cool paper background `#EEF1F4`, ink `#16232E`, monospace numerals for all token figures, hairline grid. Categorical palette keyed to source:

| source | color | | source | color |
|---|---|---|---|---|
| system | `#5B6EE1` indigo | | tool_result | `#E08A3C` amber |
| user | `#2E9E83` teal | | compaction_summary | `#C94F6D` rose |
| assistant | `#7A8699` slate | | assistant_action | `#B58A2A` ochre |

**Event styling** (not sources — a single `EVENT_STYLES` map in the template JS, used for both composition-stream markers and the events timeline) — deliberately encodes severity, not just identity:

| event | color | icon | meaning |
|---|---|---|---|
| `evicted` | `#7A8699` slate (= assistant) | ▾ | plain drop, unqualified |
| `compaction` | `#C94F6D` rose (= compaction_summary) | ◆ | **lossy** — original gone |
| `server_edit` | `#B58A2A` ochre (= assistant_action) | ✂ | server-side partial clear |
| `server_compaction` | `#C94F6D` rose — same as `compaction` | ◆ | **lossy** — same severity class |
| `reversible_evict` | `#2E9E83` teal (= user) | ↺ | **not lossy** — cached, retrievable |

`reversible_evict` (§9.4, Headroom) deliberately never gets the rose diamond — conflating "compressed but retrievable" with "summarized, original gone" would misrepresent what happened to the data. The events-timeline also attaches a small badge per row via `eventBadge(evt)`: `"◆ summarized (lossy)"` for `compaction`/`server_compaction`, `"↺ retrievable"` for `reversible_evict`, nothing otherwise — badges, not quality scores, since there's no lossiness to score for a reversible event.

**Signature element:** the **composition stream** — a stacked area chart of tokens-by-source across turns, with vertical rose markers at compaction events, slate downward ticks at evictions, ochre/rose dashed markers at `server_edit`/`server_compaction` turns, and a teal dashed marker at `reversible_evict` turns. Everything else stays quiet around it. A one-line legend note under the composition stream spells out the rose-vs-teal distinction for first-time readers: *"↺ teal = compressed but retrievable (Headroom CCR) · ◆ rose = summarized, original gone."*

**Second panel — the divergence chart:** client-side context editing and compaction (§8) show up as evictions in the composition stream. Server-side ones (§9.3) don't — the client `blocks` list never changes for those turns. The only way to *see* server-side management is to plot `totals.tokens` (client-side estimate, monotonically non-decreasing unless the client also prunes) against `usage.input_tokens` (provider-reported, drops the instant a `clear_tool_uses_20250919` or `compact_20260112` edit lands) as two overlaid lines across turns, with the same violet/plum markers at the turns where `server_edit`/`server_compaction` fired. The gap between the lines *is* the server-side clearing — screenshot-worthy on its own, and the reason this panel is first-class rather than a debug footnote.

**Layout:**

```
┌──────────────────────────────────────────────────────────────┐
│ contextlens · text2sql-run-42   24 turns · peak 41,203 ·      │
│                                  1 server edit · 1 server comp │
├──────────────────────────────────────────────────────────────┤
│  COMPOSITION STREAM  (stacked area, ~70% width of viewport)  │
│  ── compaction marker @ t15 · quality 0.60 ──                │
├──────────────────────────────────────────────────────────────┤
│  CLIENT ESTIMATE VS. PROVIDER USAGE  (divergence, two lines) │
│  ┈┈ server_edit @ t9  ┈┈ server_compaction @ t18 ┈┈           │
├──────────────┬───────────────────────────────────────────────┤
│ TURN 15      │  STALE BURN (top 5 blocks by tokens × age)    │
│ inspector    │  b7e2 run_sql result   3,120 tok · age 11     │
│ table: every │  ...                                          │
│ block, src,  ├───────────────────────────────────────────────┤
│ tokens, age  │  EVENTS TIMELINE (+ server_edit/compaction)   │
└──────────────┴───────────────────────────────────────────────┘
```

If no turn ever carried `usage`, the divergence panel renders a plain "no provider usage recorded" message instead of an empty chart.

Implementation sketch:

```python
"""report.py — render a ledger to a standalone HTML file."""

import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "report_template.html"


def load_ledger(path):
    header, turns, quality = None, [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        {"session": lambda: header,           # noqa — illustrative
         }.get(obj["kind"])
        if obj["kind"] == "session":
            header = obj
        elif obj["kind"] == "turn":
            turns.append(obj)
        elif obj["kind"] == "quality":
            quality.append(obj)
    return header, turns, quality


def render(ledger_path, out_path="contextlens-report.html"):
    header, turns, quality = load_ledger(ledger_path)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/",
        f"const SESSION={json.dumps(header)};"
        f"const TURNS={json.dumps(turns)};"
        f"const QUALITY={json.dumps(quality)};",
    )
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
```

The template's JS (~400 lines): build stacked series from `turns[i].totals.by_source`, render `<path>` polygons into an SVG, click a turn → populate the inspector table from `turns[i].blocks` sorted by tokens desc, compute stale burn as `Σ tokens×age`, draw event markers from `events` (including `server_edit`/`server_compaction`/`reversible_evict`, each styled from the shared `EVENT_STYLES` map) with `eventBadge()` badges in the timeline, and — if any turn carries `usage` — render the divergence panel as two `<path>` line charts (`totals.tokens` vs `usage.input_tokens`) sharing the same turn x-axis, with markers at server-side event turns.

---

## 12. CLI (`cli.py`)

```python
"""contextlens CLI."""

import argparse, json


def main():
    p = argparse.ArgumentParser(prog="contextlens")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="render ledger to standalone HTML")
    r.add_argument("ledger"); r.add_argument("-o", "--out",
                   default="contextlens-report.html")

    s = sub.add_parser("stats", help="print summary stats to stdout")
    s.add_argument("ledger")

    q = sub.add_parser("quality", help="score compaction events in a ledger")
    q.add_argument("ledger"); q.add_argument("--model", default=None)

    args = p.parse_args()
    if args.cmd == "report":
        from .report import render
        print(render(args.ledger, args.out))
    elif args.cmd == "stats":
        _stats(args.ledger)
    elif args.cmd == "quality":
        _quality(args.ledger, args.model)   # anthropic extra required


def _stats(path):
    from .report import load_ledger
    header, turns, _ = load_ledger(path)
    peak = max((t["totals"]["tokens"] for t in turns), default=0)
    comps = sum(1 for t in turns for e in t["events"] if e["type"] == "compaction")
    reversible = sum(
        e["tokens"] for t in turns for e in t["events"] if e["type"] == "reversible_evict")
    print(json.dumps({"session": header["label"], "turns": len(turns),
                      "peak_tokens": peak, "compactions": comps,
                      "reversible_saved_tokens": reversible}, indent=2))
```

`reversible_saved_tokens` is a separate line item from `compactions`/`evictions` deliberately — it's savings, not loss, and mixing it into the same tally would make a Headroom-heavy session look more lossy than it is.

---

## 13. Packaging (`pyproject.toml`) — yes, it's a real pip package

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "contextlens"
version = "0.1.0"
description = "Context window profiler for LLM agents: per-turn token ledger, eviction and compaction tracking, compaction quality scoring, standalone HTML reports."
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Datta Sai Krishna Somesula"}]
keywords = ["llm", "agents", "observability", "context-window", "profiler",
            "langgraph", "anthropic"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "Programming Language :: Python :: 3.10",
  "Topic :: Software Development :: Debuggers",
]
dependencies = []                       # core = stdlib only

[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
langchain = ["langchain-core>=0.3"]
tiktoken  = ["tiktoken>=0.7"]
dev       = ["pytest", "build", "twine"]

[project.scripts]
contextlens = "contextlens.cli:main"

[project.urls]
Homepage = "https://github.com/sackri10/contextlens"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
contextlens = ["report_template.html"]
```

**Publish workflow:**

```bash
pip install build twine
python -m build                        # → dist/contextlens-0.1.0-py3-none-any.whl + sdist
twine upload dist/*                    # needs a PyPI account + API token
```

Check the name is free on pypi.org first (search "contextlens"); if taken, fallbacks: `ctxlens`, `contextlens-agent`, `loopforge-contextlens` (nice tie-in to your existing brand — `pip install loopforge-contextlens`, imported as `contextlens`). Set up **trusted publishing** via GitHub Actions (`pypa/gh-action-pypi-publish`) so releases are `git tag v0.1.0 && git push --tags`.

---

## 14. Tests (`tests/test_profiler.py`) — the behaviors that must never regress

```python
from contextlens import ContextProfiler


def _turns(profiler):
    return profiler.records


def test_entry_and_eviction_detection(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    m1 = [{"role": "user", "content": "hello"}]
    m2 = m1 + [{"role": "assistant", "content": "hi"},
               {"role": "user", "content": "run the query"}]
    p.record_turn(m1); p.record_turn(m2)
    # drop the first user message → eviction
    p.record_turn(m2[1:])
    ev = [e for e in _turns(p)[2].events if e.type == "evicted"]
    assert ev and len(ev[0].block_ids) == 1


def test_compaction_detected(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    old = [{"role": "user", "content": f"msg {i} " + "x" * 200} for i in range(5)]
    p.record_turn(old)
    compacted = [{"role": "user",
                  "content": "[conversation summary] user asked about msgs 0-4"}]
    rec = p.record_turn(compacted)
    assert any(e.type == "compaction" for e in rec.events)


def test_age_tracking(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    msgs = [{"role": "system", "content": "you are helpful"}]
    for _ in range(3):
        p.record_turn(msgs)
    assert _turns(p)[-1].blocks[0].age == 2


def test_profiler_never_breaks_agent(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=False)
    assert p.record_turn(None) is None        # garbage in, no exception out
```

---

## 15. Demo scenario (`examples/demo_agent.py`)

A simulated 24-turn Text-to-SQL session — no API key needed — that produces the launch screenshot:

- Turns 1–8: conversation grows; two large `run_sql` tool results enter (3–4K tokens each)
- Turns 9–14: results go stale (stale-burn accumulates)
- Turn 15: compaction — 9 blocks evicted, one `[conversation summary]` block enters
- Turns 16–24: growth resumes
- Finishes by calling `report.render()` → `demo-report.html`

This doubles as an integration test and the GIF for your LinkedIn/Substack post.

---

## 16. Build order (three weekends)

| Milestone | Scope | Exit criterion |
|---|---|---|
| **W1 — core** | models, tokenizers, classify, profiler, tests, demo script | `pytest` green; demo JSONL correct by hand-inspection |
| **W2 — surface** | report template + renderer, CLI, Anthropic wrapper | `contextlens report demo.jsonl` opens a chart you'd screenshot |
| **W3 — moat + launch** | quality checker, LangChain handler, README, PyPI publish, profile your real Text-to-SQL pipeline | one real-pipeline report + article draft |
| **W4 — validation** | `wrap_anthropic_beta`, `server_edit`/`server_compaction` events, divergence chart, `examples/validation/` suite against every real compaction mechanism (SummarizationMiddleware, langmem, manual RemoveMessage, trim_messages, both Anthropic server-side betas), plus Headroom's `reversible_evict` and the `wrap_headroom_compress` adapter | all seven mechanisms in `examples/validation/README.md` produce exactly their expected event type; `pytest` covers the detection logic without needing network access |

**v1.1 backlog:** streaming wrapper, `rescore` with exact Anthropic counts, per-node LangGraph view, multi-session diffing (bridges toward idea #5, trajectory regression), **cache awareness** (`cache_read_input_tokens` from usage, shown alongside `server_edit`'s `cleared_input_tokens` — server-side clearing exists partly to preserve prompt-cache prefixes, so surfacing both tells the full cost story; promoted here from "nice to have" to "validated as needed" by the W4 examples).

---

## 17. Risks and mitigations

- **Per-block counts are estimates** → always display provider `usage` alongside; ship `rescore` for exactness. Honesty here is a feature, not a bug.
- **Compaction heuristic misses custom schemes** → classifier hooks + marker registration; document loudly. Validated in `examples/validation/ex3_manual_removemessage.py`: the same DIY summary produces zero `compaction` events without a hook and a correct one with a one-line hook registered.
- **Server-side context management is invisible to hash-diffing** → `wrap_anthropic_beta` reads the response instead of the request (§9.3); `server_edit`/`server_compaction` events plus the divergence chart (§11) make it visible again. This was the biggest gap the original design had — client-side-only diffing is silently wrong (not just incomplete) for any agent using Anthropic's beta context-management features.
- **Reversible compression could be mislabeled as lossy compaction** → `reversible_evict` (§4, §9.4) is a distinct event type with its own marker/badge in the report (§11); the risk isn't detection (Headroom reports exact numbers) but *misrepresentation* if it shared styling with `compaction`. Guarded by `test_reversible_evict_never_classified_as_compaction`.
- **Name collision on PyPI** → check before building brand; fallbacks listed in §13.
- **LangChain API churn** → pin `langchain-core>=0.3`, keep the handler ~40 lines so it's cheap to maintain; the raw-dict core is insulated by design.

---

*Next step when you're ready: I can generate the full working repo from this spec — every file above completed, tests passing, demo report rendered — as a zip you can `pip install -e .` immediately.*
