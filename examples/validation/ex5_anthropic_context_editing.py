"""Anthropic server-side context editing (clear_tool_uses_20250919).

Enabled via the `context-management-2025-06-27` beta header, this strategy
clears old tool results server-side when context grows past a trigger,
replacing each cleared result with placeholder text; the response reports
what happened in `context_management.applied_edits` (tool uses cleared,
input tokens cleared).

The design implication: the client message list is untouched, so
ContextLens's hash-diffing sees nothing. `wrap_anthropic_beta` (see
contextlens/integrations/anthropic_client.py) reads the response instead
and surfaces it as a `server_edit` event.

Expected ledger signature:
  - Client-side blocks: no evictions (the list only grows) -- correct!
  - `server_edit` events with cleared token counts once input passes ~8K
  - The gap between `totals.tokens` (client-side estimate, keeps growing)
    and `usage.input_tokens` (drops after each clear) is drawn as the
    "client estimate vs. provider usage" divergence panel in the report --
    that gap IS the visualization of server-side clearing.
"""

from __future__ import annotations

import anthropic

from contextlens import ContextProfiler
from contextlens.integrations.anthropic_client import wrap_anthropic_beta
from contextlens.report import render

LABEL = "ex5-editing"

TOOLS = [{
    "name": "fetch_report",
    "description": "Fetch a section of the quarterly report.",
    "input_schema": {"type": "object",
                      "properties": {"section": {"type": "string"}},
                      "required": ["section"]},
}]

CONTEXT_MGMT = {"edits": [{
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 8000},   # low, to fire fast
    "keep": {"type": "tool_uses", "value": 2},
    "clear_at_least": {"type": "input_tokens", "value": 2000},
}]}


def main():
    profiler = ContextProfiler(f"{LABEL}.jsonl", label="anthropic-clear-tool-uses")
    client = wrap_anthropic_beta(anthropic.Anthropic(), profiler)

    messages = [{"role": "user",
                 "content": "Fetch sections A, B, C, D one at a time, then compare all four."}]

    for _ in range(12):
        resp = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            betas=["context-management-2025-06-27"],
            context_management=CONTEXT_MGMT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            break
        results = [{"type": "tool_result", "tool_use_id": t.id,
                    "content": f"SECTION {t.input.get('section', '?')}: "
                               + ("revenue row; " * 600)}          # big payload
                   for t in tool_uses]
        messages.append({"role": "user", "content": results})

    profiler.close()
    print(render(f"{LABEL}.jsonl", f"{LABEL}-report.html"))


if __name__ == "__main__":
    main()
