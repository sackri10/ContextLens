"""Headroom (https://github.com/chopratejas/headroom) -- reversible
compression, a third failure mode distinct from lossy `compaction` and
plain `evicted`.

Headroom compresses tool outputs/logs/files/RAG chunks via
`compress(messages, model=...)`, returning a `CompressResult` with exact
`tokens_before`/`tokens_after`, and caches the original so the model can
retrieve it later. Nothing is deleted or summarized away -- the profiler
records this as a `reversible_evict` event, never as `compaction`.

Expected ledger signature:
  - No `evicted`/`compaction` events from the raw tool output (Headroom
    compresses it before it's ever appended to `messages`, so ContextLens
    never even sees the uncompressed version as a block)
  - A `reversible_evict` event on every turn where compress() saved tokens,
    with `tokens` exactly equal to `tokens_before - tokens_after`
  - The report's composition stream shows a teal `↺` marker (never the rose
    diamond used for compaction) at each such turn
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

import anthropic
from headroom import compress

from contextlens import ContextProfiler
from contextlens.integrations.headroom_adapter import wrap_headroom_compress
from contextlens.report import render

load_dotenv()

LABEL = "ex7-headroom"

TOOLS = [{
    "name": "run_sql",
    "description": "Run a SQL query against the warehouse.",
    "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]},
}]


def main():
    profiler = ContextProfiler(f"{LABEL}.jsonl", label="headroom-ccr")
    headroom_compress = wrap_headroom_compress(compress, profiler)
    client = anthropic.Anthropic()

    messages = [{"role": "user",
                 "content": "Query top 500 customers by revenue, then tell me the top 3."}]

    for _ in range(6):
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            break
        # Duplicate-heavy rows (like a real paginated SQL result: repeating
        # customers/regions/status across pages) -- Headroom's SmartCrusher
        # dedupes this; 500 all-unique tuples give it nothing to compress.
        raw_results = [{"type": "tool_result", "tool_use_id": t.id,
                        "content": json.dumps([
                            {"id": i % 20, "customer": f"Customer {i % 20}",
                             "region": "US-WEST", "status": "active",
                             "amount": (i % 20) * 37}
                            for i in range(500)])}
                       for t in tool_uses]
        # Headroom compresses the raw tool output before it joins context
        compressed = headroom_compress(raw_results, model="claude-sonnet-4-6")
        messages.append({"role": "user", "content": compressed.messages})

        # ContextLens profiles what the model actually received this turn
        profiler.record_turn(messages, model="claude-sonnet-4-6",
                              usage={"input_tokens": resp.usage.input_tokens,
                                     "output_tokens": resp.usage.output_tokens})

    profiler.close()
    print(render(f"{LABEL}.jsonl", f"{LABEL}-report.html"))


if __name__ == "__main__":
    main()
