"""Shared harness for the validation examples in this directory.

Every ex*.py script wires a ContextProfiler + ContextWatchCallbackHandler
through make_profiler(), runs its own agent/graph, then calls finish() to
close the ledger and render the report.
"""

from __future__ import annotations

from contextwatch import ContextProfiler
from contextwatch.integrations.langchain_handler import ContextWatchCallbackHandler
from contextwatch.report import render


def make_profiler(label: str):
    profiler = ContextProfiler(f"{label}.jsonl", label=label)
    handler = ContextWatchCallbackHandler(profiler)
    return profiler, handler


def finish(profiler, label: str):
    profiler.close()
    print("report:", render(f"{label}.jsonl", f"{label}-report.html"))
