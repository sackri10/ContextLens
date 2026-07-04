"""ContextProfiler: turn message lists into a per-turn context ledger."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .models import ContextBlock, Event, SessionHeader, TurnRecord, _normalize_content, block_id
from .classify import classify, Classifier
from .tokenizers import TokenCounter, heuristic_counter

# Heuristic: a compaction = several blocks evicted AND a summary-ish block
# entering in the same turn, with net token reduction.
COMPACTION_MIN_EVICTED = 3


class ContextProfiler:
    def __init__(
        self,
        sink: str | Path | None = "contextwatch.jsonl",
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

        # ---- server-side context management (Anthropic beta) ----
        # Context editing and compaction can run server-side: the client
        # `messages` list never changes, so hash-diffing sees nothing. The
        # beta wrapper (integrations/anthropic_client.wrap_anthropic_beta)
        # detects these from the *response* and passes them through `meta`;
        # promote them to first-class events here so the report can show
        # them alongside client-side evictions/compactions.
        for edit in meta.get("server_edits", []) or []:
            events.append(Event(
                type="server_edit",
                tokens=edit.get("cleared_input_tokens") or 0,
                detail=(f"{edit.get('type', '?')}: cleared "
                        f"{edit.get('cleared_tool_uses')} tool uses server-side "
                        f"({edit.get('cleared_input_tokens')} tok)")))
        if meta.get("server_compaction"):
            events.append(Event(
                type="server_compaction",
                detail="stop_reason=compaction (server-side summary)"))

        # --- flush any events queued by wrap_headroom_compress() ---
        # Headroom's compress() runs outside of record_turn (it's called on
        # the raw tool output before it's appended to messages), so it can't
        # attach its `reversible_evict` event to a turn directly. It queues
        # the event on the profiler instead; the next record_turn() call
        # picks it up here.
        events.extend(getattr(self, "_pending_events", []))
        self._pending_events = []

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
