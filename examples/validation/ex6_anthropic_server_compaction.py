"""Anthropic server-side compaction (compact_20260112).

Server-side compaction summarizes earlier conversation on the server so it
can continue past window limits (beta header `compact-2026-01-12`);
Anthropic recommends it over the deprecated SDK `compaction_control`. With
`pause_after_compaction`, the API returns `stop_reason: "compaction"` so a
harness can inspect the summary before continuing.

Expected: a `server_compaction` event on the compaction turn;
`usage.input_tokens` drops sharply while the client-side `totals.tokens`
keeps climbing -- visible as the divergence panel in the report. The recall
probe ("metric_1's exact value") is a one-shot compaction quality check:
early chunks' precise figures are exactly the kind of subtle-but-critical
detail compaction risks losing, so this script logs whether the answer
survives.
"""

from __future__ import annotations

import anthropic

from contextlens import ContextProfiler
from contextlens.integrations.anthropic_client import wrap_anthropic_beta
from contextlens.report import render

LABEL = "ex6-compact"

CONTEXT_MGMT = {"edits": [{
    "type": "compact_20260112",
    "trigger": {"type": "input_tokens", "value": 20000},  # low for the demo
    # "pause_after_compaction": True,   # enable to intercept the summary
}]}


def main():
    profiler = ContextProfiler(f"{LABEL}.jsonl", label="anthropic-server-compaction")
    client = wrap_anthropic_beta(anthropic.Anthropic(), profiler)

    messages = [{"role": "user",
                 "content": "Let's review a long document. I'll paste chunks; "
                            "track key figures across all chunks."}]

    for i in range(10):
        messages.append({"role": "user",
                          "content": f"CHUNK {i}: metric_{i} = {i * 111}. "
                                     + ("filler analysis text; " * 500)})
        resp = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            betas=["compact-2026-01-12"],
            context_management=CONTEXT_MGMT,
            messages=messages,
        )
        if resp.stop_reason == "compaction":
            print(f"turn {i}: server compacted -- summary now anchors the thread")
        messages.append({"role": "assistant", "content": resp.content})

    # Recall probe for quality checking:
    messages.append({"role": "user", "content": "What was metric_1's exact value?"})
    resp = client.beta.messages.create(
        model="claude-sonnet-4-6", max_tokens=256,
        betas=["compact-2026-01-12"], context_management=CONTEXT_MGMT,
        messages=messages)
    print("recall answer:", resp.content[0].text)

    profiler.close()
    print(render(f"{LABEL}.jsonl", f"{LABEL}-report.html"))


if __name__ == "__main__":
    main()
